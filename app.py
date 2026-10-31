import base64
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
import threading
import time
import webbrowser
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
import uvicorn
from PIL import Image
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
BUNDLE_DIR = APP_DIR / "model_bundle"
MODEL_DIR = BUNDLE_DIR / "models"
CONFIG_PATH = BUNDLE_DIR / "fusion_config.json"
CALIBRATOR_PATH = BUNDLE_DIR / "calibrators.joblib"

MODEL_FILES = {
    "EfficientNetV2B0": "efficientnetv2b0.keras",
    "ResNet50": "resnet50.keras",
    "DenseNet121": "densenet121.keras",
    "VGG16": "vgg16.keras",
    "Xception": "xception.keras",
    "CustomAttentionCNN": "custom_attention_cnn.keras",
}

_ENSEMBLE_LOCK = threading.Lock()
_ENSEMBLE_CACHE = {
    "loaded": False,
    "models": None,
    "calibrators": None,
    "config": None,
    "error": None,
}


def load_ensemble():
    """Load and cache the six trained detectors, Platt calibrators, and fusion config."""
    with _ENSEMBLE_LOCK:
        if _ENSEMBLE_CACHE["loaded"] and _ENSEMBLE_CACHE["models"] is not None:
            return (
                _ENSEMBLE_CACHE["models"],
                _ENSEMBLE_CACHE["calibrators"],
                _ENSEMBLE_CACHE["config"],
            )

        if not CONFIG_PATH.exists():
            raise FileNotFoundError(
                "model_bundle/fusion_config.json is missing. "
                "Run the Colab export cell and copy the generated model bundle."
            )
        if not CALIBRATOR_PATH.exists():
            raise FileNotFoundError(
                "model_bundle/calibrators.joblib is missing. "
                "Run the Colab export cell and copy the generated model bundle."
            )

        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        calibrators = joblib.load(CALIBRATOR_PATH)

        models = {}
        missing = []
        for model_name, filename in MODEL_FILES.items():
            model_path = MODEL_DIR / filename
            if not model_path.exists():
                missing.append(str(model_path.relative_to(APP_DIR)))
                continue
            models[model_name] = tf.keras.models.load_model(
                model_path, compile=False
            )

        if missing:
            raise FileNotFoundError(
                "The following trained model files are missing:\n- "
                + "\n- ".join(missing)
            )

        expected = config.get("model_names", list(MODEL_FILES))
        missing_calibrators = [n for n in expected if n not in calibrators]
        if missing_calibrators:
            raise ValueError(
                "Missing calibration parameters for: "
                + ", ".join(missing_calibrators)
            )

        _ENSEMBLE_CACHE["loaded"] = True
        _ENSEMBLE_CACHE["models"] = models
        _ENSEMBLE_CACHE["calibrators"] = calibrators
        _ENSEMBLE_CACHE["config"] = config
        _ENSEMBLE_CACHE["error"] = None

        return models, calibrators, config


def platt_calibrate(fake_probability, calibrator):
    """Apply the same Platt scaling used in the Colab notebook."""
    p = np.clip(np.asarray(fake_probability, dtype=np.float64), 1e-7, 1 - 1e-7)
    logits = np.log(p / (1.0 - p))
    slope = float(calibrator["slope"])
    intercept = float(calibrator["intercept"])
    z = np.clip(slope * logits + intercept, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def predict_image(image, models, calibrators, config):
    t0 = time.perf_counter()
    image = image.convert("RGB").resize((224, 224))
    # Match the notebook input pipeline: RGB float32 pixel values, 0..255.
    array = np.asarray(image, dtype=np.float32)[None, ...]

    names = list(models.keys())
    raw_fake = {}
    calibrated_fake = {}

    for name, model in models.items():
        p_real = float(np.asarray(model.predict(array, verbose=0)).reshape(-1)[0])
        p_fake = 1.0 - p_real
        raw_fake[name] = p_fake
        calibrated_fake[name] = float(
            platt_calibrate([p_fake], calibrators[name])[0]
        )

    global_weights = config["global_weights"]
    beta = float(config["beta"])
    threshold = float(config["adaptive_threshold"])

    probabilities = np.asarray(
        [calibrated_fake[n] for n in names], dtype=np.float64
    )
    confidence = np.abs(probabilities - 0.5) * 2.0
    base_weights = np.asarray(
        [float(global_weights[n]) for n in names], dtype=np.float64
    )

    logits = np.log(np.maximum(base_weights, 1e-12)) + beta * confidence
    weights = np.exp(logits - np.max(logits))
    weights /= np.sum(weights)

    p_fake_final = float(np.sum(weights * probabilities))
    p_real_final = 1.0 - p_fake_final
    label = "FAKE" if p_fake_final >= threshold else "REAL"
    latency_ms = int(round((time.perf_counter() - t0) * 1000))

    return {
        "prediction": label,
        "fake_probability": p_fake_final,
        "real_probability": p_real_final,
        "threshold": threshold,
        "raw_fake": raw_fake,
        "calibrated_fake": calibrated_fake,
        "adaptive_weights": dict(zip(names, weights.tolist())),
        "latency_ms": latency_ms,
    }


def make_pdf(image, filename, result, raw_bytes=None, config=None):
    """Generate an enterprise-grade forensic dossier PDF report matching the FusionShield AI UI."""
    from reportlab.graphics.shapes import Drawing, Line, Rect, String
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import (
        Flowable,
        Image as RLImage,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    if config is None:
        config = _ENSEMBLE_CACHE.get("config") or {}
    global_weights = config.get("global_weights", {})
    beta = float(config.get("beta", 2.0))

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=14 * mm,
        leftMargin=14 * mm,
        topMargin=13 * mm,
        bottomMargin=14 * mm,
        title=f"FusionShield AI Forensic Report - {filename}",
        author="FusionShield AI",
    )

    content_w = 182 * mm  # 210mm - 28mm margins
    styles = getSampleStyleSheet()

    # Palette
    c_navy = colors.HexColor("#0F172A")
    c_slate_dark = colors.HexColor("#1E293B")
    c_slate = colors.HexColor("#475569")
    c_muted = colors.HexColor("#64748B")
    c_light = colors.HexColor("#94A3B8")
    c_border = colors.HexColor("#E2E8F0")
    c_bg_subtle = colors.HexColor("#F8FAFC")

    c_emerald_bg = colors.HexColor("#ECFDF5")
    c_emerald_border = colors.HexColor("#A7F3D0")
    c_emerald_text = colors.HexColor("#065F46")
    c_emerald_accent = colors.HexColor("#10B981")

    c_crimson_bg = colors.HexColor("#FEF2F2")
    c_crimson_border = colors.HexColor("#FECACA")
    c_crimson_text = colors.HexColor("#991B1B")
    c_crimson_accent = colors.HexColor("#EF4444")
    c_indigo = colors.HexColor("#4F46E5")

    is_fake = result["prediction"] == "FAKE"
    p_fake = float(result["fake_probability"])
    p_real = float(result["real_probability"])
    threshold = float(result["threshold"])
    latency_ms = result.get("latency_ms", 142)

    # Calculate SHA-256 & dimensions
    orig_w, orig_h = image.size
    if raw_bytes:
        sha256_hex = hashlib.sha256(raw_bytes).hexdigest()
        file_kb = f"{len(raw_bytes) / 1024:.1f} KB"
    else:
        tmp_b = io.BytesIO()
        image.save(tmp_b, format="PNG")
        raw_b = tmp_b.getvalue()
        sha256_hex = hashlib.sha256(raw_b).hexdigest()
        file_kb = f"{len(raw_b) / 1024:.1f} KB"

    report_id = f"FS-{sha256_hex[:8].upper()}-{int(time.time()) % 100000:05d}"
    ist_tz = timezone(timedelta(hours=5, minutes=30))
    ist_now = datetime.now(ist_tz).strftime("%Y-%m-%d %H:%M:%S IST")

    detector_names = list(result["calibrated_fake"].keys())
    total_heads = len(detector_names)
    agreeing_heads = sum(
        1
        for n in detector_names
        if (result["calibrated_fake"][n] >= threshold) == is_fake
    )
    margin = abs(p_fake - threshold)
    conf_tier = "High" if margin >= 0.25 else ("Moderate" if margin >= 0.10 else "Borderline")

    # Paragraph Styles
    st_header_eyebrow = ParagraphStyle(
        "HeaderEyebrow",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#059669"),
    )
    st_header_title = ParagraphStyle(
        "HeaderTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12.5,
        leading=16,
        textColor=colors.HexColor("#064E3B"),
    )
    st_header_sub = ParagraphStyle(
        "HeaderSub",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.8,
        leading=10.5,
        textColor=colors.HexColor("#047857"),
    )
    st_header_meta = ParagraphStyle(
        "HeaderMeta",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=7.5,
        leading=11.5,
        alignment=TA_RIGHT,
        textColor=colors.HexColor("#065F46"),
    )
    st_sec_eyebrow = ParagraphStyle(
        "SecEyebrow",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7,
        leading=9,
        textColor=c_muted,
    )
    st_sec_title = ParagraphStyle(
        "SecTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=c_navy,
    )
    st_meta_label = ParagraphStyle(
        "MetaLabel",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=6.8,
        leading=9,
        textColor=c_muted,
    )
    st_meta_val = ParagraphStyle(
        "MetaVal",
        parent=styles["Normal"],
        fontName="Courier-Bold",
        fontSize=8,
        leading=10.5,
        textColor=c_navy,
    )
    st_cell_arch = ParagraphStyle(
        "CellArch",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=c_navy,
    )
    st_cell_mono = ParagraphStyle(
        "CellMono",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=8,
        leading=10,
        textColor=c_slate_dark,
    )
    st_cell_mono_bold = ParagraphStyle(
        "CellMonoBold",
        parent=styles["Normal"],
        fontName="Courier-Bold",
        fontSize=8,
        leading=10,
        textColor=c_navy,
    )

    def make_progress_bar(pct_0_to_1, width_mm, height_mm, fill_color, mark_ratio=None):
        w = width_mm * mm
        h = height_mm * mm
        d = Drawing(w, h + 4)
        # Background track
        d.add(
            Rect(
                0,
                2,
                w,
                h,
                rx=h / 2.0,
                ry=h / 2.0,
                fillColor=colors.HexColor("#E2E8F0"),
                strokeColor=None,
            )
        )
        # Active fill
        fill_w = max(2.0, min(w, w * float(pct_0_to_1)))
        d.add(
            Rect(
                0,
                2,
                fill_w,
                h,
                rx=h / 2.0,
                ry=h / 2.0,
                fillColor=fill_color,
                strokeColor=None,
            )
        )
        if mark_ratio is not None:
            mx = max(1.0, min(w - 1.0, w * float(mark_ratio)))
            d.add(
                Line(
                    mx,
                    0,
                    mx,
                    h + 4,
                    strokeColor=c_navy,
                    strokeWidth=1.4,
                )
            )
        return d

    story = []

    # ==================== 1. TOP WHITE & LIGHT-GREEN GRADIENT BANNER WITH LOGO ====================
    class GradientHeaderBanner(Flowable):
        def __init__(self, inner_table, width_pt):
            super().__init__()
            self.inner_table = inner_table
            self.width = width_pt
            _, self.height = self.inner_table.wrap(self.width, 1000)

        def wrap(self, availWidth, availHeight):
            _, self.height = self.inner_table.wrap(self.width, availHeight)
            return self.width, self.height

        def draw(self):
            c = self.canv
            c.saveState()
            # Rounded clipping path for smooth white -> light emerald gradient
            path = c.beginPath()
            path.roundRect(0, 0, self.width, self.height, 7)
            c.clipPath(path, stroke=0, fill=0)

            steps = 48
            # Smooth horizontal gradient from pure white (#FFFFFF) to light mint/emerald (#D1FAE5)
            r1, g1, b1 = 1.0, 1.0, 1.0
            r2, g2, b2 = 209 / 255.0, 250 / 255.0, 229 / 255.0
            step_w = self.width / float(steps)
            for i in range(steps):
                t = i / float(steps - 1)
                c.setFillColorRGB(
                    r1 + (r2 - r1) * t,
                    g1 + (g2 - g1) * t,
                    b1 + (b2 - b1) * t,
                )
                c.rect(i * step_w, 0, step_w + 0.8, self.height, stroke=0, fill=1)

            # Subtle top/bottom accent border
            c.setStrokeColor(colors.HexColor("#6EE7B7"))
            c.setLineWidth(1.0)
            c.roundRect(0, 0, self.width, self.height, 7, stroke=1, fill=0)
            c.restoreState()

            self.inner_table.drawOn(c, 0, 0)

    logo_path = STATIC_DIR / "logo.png"
    if not logo_path.exists():
        logo_path = APP_DIR / "FusionShield AI Green Shield Logo.png"

    if logo_path.exists():
        logo_flowable = RLImage(str(logo_path), width=17 * mm, height=17 * mm)
    else:
        logo_flowable = Spacer(17 * mm, 17 * mm)

    banner_center = [
        Paragraph("FUSIONSHIELD AI", st_header_eyebrow),
        Spacer(1, 1 * mm),
        Paragraph(
            "A Heterogeneous Ensemble Framework for Detecting Deepfake and AI-Generated Images",
            st_header_title,
        ),
        Spacer(1, 1 * mm),
        Paragraph(
            "Six-Detector Convolutional Ensemble • Platt Logit Calibration &amp; Adaptive Fusion",
            st_header_sub,
        ),
    ]
    banner_right = [
        Paragraph(
            f'<font color="#047857"><b>REPORT ID:</b></font> <font color="#0F172A"><b>{report_id}</b></font>',
            st_header_meta,
        ),
        Paragraph(
            f'<font color="#047857"><b>TIMESTAMP:</b></font> <font color="#0F172A">{ist_now}</font>',
            st_header_meta,
        ),
        Paragraph(
            '<font color="#047857"><b>ENSEMBLE:</b></font> <font color="#059669"><b>6 Active CNNs</b></font>',
            st_header_meta,
        ),
        Paragraph(
            f'<font color="#047857"><b>LATENCY:</b></font> <font color="#0F172A">{latency_ms} ms</font>',
            st_header_meta,
        ),
    ]
    banner_inner_tbl = Table(
        [[logo_flowable, banner_center, banner_right]],
        colWidths=[22 * mm, 96 * mm, 64 * mm],
    )
    banner_inner_tbl.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (0, 0), "CENTER"),
                ("LEFTPADDING", (0, 0), (0, 0), 8),
                ("RIGHTPADDING", (0, 0), (0, 0), 4),
                ("LEFTPADDING", (1, 0), (1, 0), 4),
                ("RIGHTPADDING", (1, 0), (1, 0), 8),
                ("LEFTPADDING", (2, 0), (2, 0), 6),
                ("RIGHTPADDING", (2, 0), (2, 0), 11),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("LINEBEFORE", (2, 0), (2, 0), 0.6, colors.HexColor("#A7F3D0")),
            ]
        )
    )
    story.append(GradientHeaderBanner(banner_inner_tbl, content_w))
    story.append(Spacer(1, 5 * mm))

    # ==================== 2. SOURCE EVIDENCE & TELEMETRY PANEL ====================
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="PNG")
    image_buffer.seek(0)
    reader = ImageReader(image_buffer)
    iw, ih = reader.getSize()
    max_w, max_h = 56 * mm, 44 * mm
    scale = min(max_w / iw, max_h / ih)
    rl_img = RLImage(image_buffer, width=iw * scale, height=ih * scale)

    img_box = Table([[rl_img]], colWidths=[62 * mm])
    img_box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), c_bg_subtle),
                ("BOX", (0, 0), (-1, -1), 0.6, c_border),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("ROUNDEDCORNERS", [5, 5, 5, 5]),
            ]
        )
    )

    short_name = filename if len(filename) <= 34 else filename[:31] + "..."
    meta_rows = [
        [
            Paragraph("SOURCE TARGET FILE", st_meta_label),
            Paragraph("ORIGINAL RESOLUTION", st_meta_label),
        ],
        [
            Paragraph(short_name, st_meta_val),
            Paragraph(f"{orig_w} x {orig_h} px ({file_kb})", st_meta_val),
        ],
        [
            Paragraph("SHA-256 DIGITAL FINGERPRINT", st_meta_label),
            Paragraph("INPUT TENSOR PIPELINE", st_meta_label),
        ],
        [
            Paragraph(f"{sha256_hex[:24]}...", st_meta_val),
            Paragraph("[224, 224] RGB Float32", st_meta_val),
        ],
        [
            Paragraph("DECISION THRESHOLD (EQUI-COST)", st_meta_label),
            Paragraph("PLATT SCALING (BETA)", st_meta_label),
        ],
        [
            Paragraph(f"{threshold:.6f}  (Output: P(FAKE))", st_meta_val),
            Paragraph(f"Beta = {beta:.2f}  |  Calibrated", st_meta_val),
        ],
    ]
    meta_tbl = Table(meta_rows, colWidths=[62 * mm, 54 * mm])
    meta_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), c_bg_subtle),
                ("BOX", (0, 0), (-1, -1), 0.6, c_border),
                ("INNERGRID", (0, 0), (-1, -1), 0.3, c_border),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
                ("ROUNDEDCORNERS", [5, 5, 5, 5]),
            ]
        )
    )

    evidence_layout = Table(
        [[img_box, meta_tbl]],
        colWidths=[65 * mm, 117 * mm],
    )
    evidence_layout.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    story.append(evidence_layout)
    story.append(Spacer(1, 5.5 * mm))

    # ==================== 3. FORENSIC SYNTHESIS VERDICT ====================
    story.append(Paragraph("FORENSIC SYNTHESIS VERDICT", st_sec_eyebrow))
    story.append(
        Paragraph("Ensemble Consensus &amp; Probabilistic Synthesis", st_sec_title)
    )
    story.append(Spacer(1, 2.5 * mm))

    verdict_bg = c_crimson_bg if is_fake else c_emerald_bg
    verdict_bd = c_crimson_border if is_fake else c_emerald_border
    verdict_fg = c_crimson_text if is_fake else c_emerald_text
    verdict_ac = c_crimson_accent if is_fake else c_emerald_accent

    badge_text = "SYNTHETIC / FAKE" if is_fake else "AUTHENTIC ESTIMATE"
    headline_text = (
        "FAKE — Manipulated Estimate" if is_fake else "REAL — Authentic Estimate"
    )

    st_v_badge = ParagraphStyle(
        "VBadge",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7,
        leading=9,
        textColor=verdict_fg,
    )
    st_v_head = ParagraphStyle(
        "VHead",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=c_navy,
    )
    st_v_sub = ParagraphStyle(
        "VSub",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.8,
        leading=10.5,
        textColor=verdict_fg,
    )

    card_verdict = Table(
        [
            [
                Paragraph(f"<b>[ {badge_text} ]</b>", st_v_badge),
            ],
            [
                Paragraph(headline_text, st_v_head),
            ],
            [
                Paragraph(
                    f"Consensus achieved across {agreeing_heads} of {total_heads} detector heads "
                    f"(Confidence: {conf_tier})",
                    st_v_sub,
                ),
            ],
        ],
        colWidths=[64 * mm],
    )
    card_verdict.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), verdict_bg),
                ("BOX", (0, 0), (-1, -1), 0.9, verdict_bd),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ROUNDEDCORNERS", [5, 5, 5, 5]),
            ]
        )
    )

    st_prob_num_fake = ParagraphStyle(
        "ProbNumFake",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=c_crimson_text if is_fake else c_navy,
    )
    st_prob_num_real = ParagraphStyle(
        "ProbNumReal",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=c_emerald_text if not is_fake else c_navy,
    )

    card_fake_prob = Table(
        [
            [Paragraph("FAKE PROBABILITY", st_meta_label)],
            [Paragraph(f"{p_fake * 100:.2f}%", st_prob_num_fake)],
            [make_progress_bar(p_fake, 29, 2.2, c_crimson_accent)],
        ],
        colWidths=[35 * mm],
    )
    card_fake_prob.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), c_bg_subtle),
                ("BOX", (0, 0), (-1, -1), 0.6, c_border),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 4.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
                ("ROUNDEDCORNERS", [5, 5, 5, 5]),
            ]
        )
    )

    card_real_prob = Table(
        [
            [Paragraph("REAL PROBABILITY", st_meta_label)],
            [Paragraph(f"{p_real * 100:.2f}%", st_prob_num_real)],
            [make_progress_bar(p_real, 29, 2.2, c_emerald_accent)],
        ],
        colWidths=[35 * mm],
    )
    card_real_prob.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), c_bg_subtle),
                ("BOX", (0, 0), (-1, -1), 0.6, c_border),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 4.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
                ("ROUNDEDCORNERS", [5, 5, 5, 5]),
            ]
        )
    )

    st_scale_note = ParagraphStyle(
        "ScaleNote",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=6.5,
        leading=8.5,
        textColor=c_muted,
    )
    card_scale = Table(
        [
            [
                Paragraph(
                    f"ADAPTIVE FUSION P(FAKE): <b>{p_fake:.6f}</b>",
                    st_meta_label,
                )
            ],
            [
                make_progress_bar(
                    p_fake,
                    36,
                    2.6,
                    verdict_ac,
                    mark_ratio=threshold,
                )
            ],
            [
                Paragraph(
                    f"0.0 (Real) | Thresh: {threshold:.3f} | 1.0 (Fake)",
                    st_scale_note,
                )
            ],
        ],
        colWidths=[42 * mm],
    )
    card_scale.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), c_bg_subtle),
                ("BOX", (0, 0), (-1, -1), 0.6, c_border),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
                ("ROUNDEDCORNERS", [5, 5, 5, 5]),
            ]
        )
    )

    verdict_strip = Table(
        [[card_verdict, card_fake_prob, card_real_prob, card_scale]],
        colWidths=[66 * mm, 37 * mm, 37 * mm, 42 * mm],
    )
    verdict_strip.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    story.append(verdict_strip)
    story.append(Spacer(1, 6 * mm))

    # ==================== 4. INDIVIDUAL DETECTOR ANALYSIS TABLE ====================
    story.append(Paragraph("CROSS-ARCHITECTURE BREAKDOWN", st_sec_eyebrow))
    story.append(
        Paragraph(
            "Individual Detector Analysis (Raw, Platt-Calibrated &amp; Adaptive Fusion Weights)",
            st_sec_title,
        )
    )
    story.append(Spacer(1, 2.5 * mm))

    st_th = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=6.8,
        leading=8.5,
        textColor=colors.white,
    )

    table_rows = [
        [
            Paragraph("DETECTOR<br/>ARCHITECTURE", st_th),
            Paragraph("RAW<br/>P(REAL)", st_th),
            Paragraph("RAW<br/>P(FAKE)", st_th),
            Paragraph("CALIBRATED<br/>P(FAKE)", st_th),
            Paragraph("VISUAL<br/>P(FAKE)", st_th),
            Paragraph("BASE<br/>WEIGHT", st_th),
            Paragraph("ADAPTIVE<br/>WEIGHT", st_th),
            Paragraph("DETECTOR<br/>VOTE", st_th),
        ]
    ]

    row_styles = [
        ("BACKGROUND", (0, 0), (-1, 0), c_navy),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, c_border),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, 0), 6),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("TOPPADDING", (0, 1), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
    ]

    for idx, name in enumerate(detector_names, start=1):
        cal_f = float(result["calibrated_fake"][name])
        raw_f = float(result.get("raw_fake", {}).get(name, cal_f))
        raw_r = 1.0 - raw_f
        ad_w = float(result["adaptive_weights"][name])
        base_w = float(global_weights.get(name, 1.0 / max(1, total_heads)))
        vote_fake = cal_f >= threshold

        vote_label = "FAKE" if vote_fake else "REAL"
        vote_fg = "#991B1B" if vote_fake else "#065F46"
        vote_bg = c_crimson_bg if vote_fake else c_emerald_bg
        bar_col = c_crimson_accent if vote_fake else c_emerald_accent

        st_vote = ParagraphStyle(
            f"Vote_{idx}",
            parent=styles["Normal"],
            fontName="Courier-Bold",
            fontSize=7.5,
            leading=9,
            alignment=TA_CENTER,
            textColor=colors.HexColor(vote_fg),
        )

        table_rows.append(
            [
                Paragraph(name, st_cell_arch),
                Paragraph(f"{raw_r * 100:.2f}%", st_cell_mono),
                Paragraph(f"{raw_f * 100:.2f}%", st_cell_mono),
                Paragraph(f"{cal_f * 100:.2f}%", st_cell_mono_bold),
                make_progress_bar(cal_f, 20, 2.0, bar_col),
                Paragraph(f"{base_w * 100:.2f}%", st_cell_mono),
                Paragraph(f"{ad_w * 100:.2f}%", st_cell_mono_bold),
                Paragraph(vote_label, st_vote),
            ]
        )

        bg_row = colors.white if idx % 2 == 1 else c_bg_subtle
        row_styles.append(("BACKGROUND", (0, idx), (6, idx), bg_row))
        row_styles.append(("BACKGROUND", (7, idx), (7, idx), vote_bg))

    det_table = Table(
        table_rows,
        colWidths=[
            37 * mm,
            19 * mm,
            19 * mm,
            22 * mm,
            24 * mm,
            19 * mm,
            22 * mm,
            20 * mm,
        ],
        repeatRows=1,
    )
    det_table.setStyle(TableStyle(row_styles))
    story.append(det_table)
    story.append(Spacer(1, 6 * mm))

    # ==================== 5. METHODOLOGY & FORENSIC DISCLAIMER ====================
    st_foot_title = ParagraphStyle(
        "FootTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=c_navy,
    )
    st_foot_body = ParagraphStyle(
        "FootBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=10.5,
        textColor=c_slate,
    )

    disclaimer_box = Table(
        [
            [
                Paragraph(
                    "METHODOLOGY &amp; EVIDENTIARY LIMITATION NOTICE",
                    st_foot_title,
                )
            ],
            [
                Paragraph(
                    "1. <b>Adaptive Ensemble Fusion:</b> Six CNN detectors independently analyse the input image and generate raw P(REAL) predictions, which are converted into P(FAKE) probabilities and calibrated using validation-fitted Platt scaling. The fusion engine combines global detector reliability with image-specific confidence to dynamically calculate detector weights and produce the final weighted P(FAKE) score.<br/>"
"2. <b>Forensic Disclaimer:</b> This report presents an experimental deep-learning-based statistical estimate and does not constitute definitive proof of image authenticity or manipulation. Detection performance may vary across unseen AI image generators, manipulation techniques, image transformations, post-processing and compression. Probability scores and model predictions should therefore be interpreted as supporting indicators rather than absolute conclusions.",
                    st_foot_body,
                )
            ],
        ],
        colWidths=[content_w],
    )
    disclaimer_box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), c_bg_subtle),
                ("BOX", (0, 0), (-1, -1), 0.6, c_border),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("ROUNDEDCORNERS", [5, 5, 5, 5]),
            ]
        )
    )
    story.append(disclaimer_box)

    def draw_page_footer(canvas, doc_obj):
        canvas.saveState()
        canvas.setStrokeColor(c_border)
        canvas.setLineWidth(0.5)
        canvas.line(14 * mm, 10 * mm, 196 * mm, 10 * mm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(c_muted)
        canvas.drawString(
            14 * mm,
            6.5 * mm,
            f"FusionShield AI - ",
        )
        canvas.drawRightString(
            196 * mm,
            6.5 * mm,
            f"Page {doc_obj.page}",
        )
        canvas.restoreState()

    doc.build(story, onFirstPage=draw_page_footer, onLaterPages=draw_page_footer)
    buffer.seek(0)
    return buffer.getvalue()


async def homepage(request):
    return FileResponse(STATIC_DIR / "index.html")


async def pwa_manifest(request):
    return FileResponse(
        STATIC_DIR / "manifest.json",
        media_type="application/manifest+json",
    )


async def pwa_service_worker(request):
    return FileResponse(
        STATIC_DIR / "sw.js",
        media_type="application/javascript",
        headers={"Service-Worker-Allowed": "/"},
    )


async def api_status(request):
    try:
        models, _, config = load_ensemble()
        return JSONResponse(
            {
                "loaded": True,
                "detector_count": len(models),
                "image_size": config.get("image_size", [224, 224]),
                "model_output": config.get("model_output", "P(REAL)"),
                "fusion_output": config.get("fusion_output", "P(FAKE)"),
                "adaptive_threshold": float(config.get("adaptive_threshold", 0.498)),
                "beta": float(config.get("beta", 2.0)),
                "global_weights": config.get("global_weights", {}),
                "model_names": list(models.keys()),
            }
        )
    except Exception as exc:
        return JSONResponse(
            {
                "loaded": False,
                "detector_count": 0,
                "error": str(exc),
            },
            status_code=500,
        )


async def api_predict(request):
    try:
        models, calibrators, config = load_ensemble()
    except Exception as exc:
        return JSONResponse(
            {"error": f"The trained model bundle could not be loaded: {exc}"},
            status_code=500,
        )

    form = await request.form()
    upload = form.get("file")
    if upload is None:
        return JSONResponse({"error": "No image file provided."}, status_code=400)

    filename = getattr(upload, "filename", "uploaded_image.png") or "uploaded_image.png"
    raw_bytes = await upload.read()

    try:
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception:
        return JSONResponse(
            {"error": "The uploaded file could not be decoded as an image."},
            status_code=400,
        )

    result = predict_image(image, models, calibrators, config)
    pdf_bytes = make_pdf(image, filename, result, raw_bytes=raw_bytes, config=config)
    safe_name = Path(filename).stem.replace(" ", "_")

    result["pdf_base64"] = base64.b64encode(pdf_bytes).decode("ascii")
    result["pdf_filename"] = f"FusionShieldAI_{safe_name}_report.pdf"
    return JSONResponse(result)


async def api_report(request):
    try:
        models, calibrators, config = load_ensemble()
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

    form = await request.form()
    upload = form.get("file")
    if upload is None:
        return JSONResponse({"error": "No image file provided."}, status_code=400)

    filename = getattr(upload, "filename", "uploaded_image.png") or "uploaded_image.png"
    raw_bytes = await upload.read()
    try:
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception:
        return JSONResponse(
            {"error": "The uploaded file could not be decoded as an image."},
            status_code=400,
        )

    result = predict_image(image, models, calibrators, config)
    pdf_bytes = make_pdf(image, filename, result, raw_bytes=raw_bytes, config=config)
    safe_name = Path(filename).stem.replace(" ", "_")
    headers = {
        "Content-Disposition": f'attachment; filename="FusionShieldAI_{safe_name}_report.pdf"'
    }
    return Response(pdf_bytes, media_type="application/pdf", headers=headers)


@asynccontextmanager
async def lifespan(app):
    try:
        print("Loading six trained detectors into memory...")
        load_ensemble()
        print("FusionShield AI ensemble loaded and ready.")
    except Exception as exc:
        print(f"Warning: Could not pre-load model bundle on startup: {exc}")
    yield


routes = [
    Route("/", homepage),
    Route("/manifest.json", pwa_manifest, methods=["GET"]),
    Route("/sw.js", pwa_service_worker, methods=["GET"]),
    Route("/api/status", api_status, methods=["GET"]),
    Route("/api/predict", api_predict, methods=["POST"]),
    Route("/api/report", api_report, methods=["POST"]),
    Mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static"),
]

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

app = Starlette(
    debug=False,
    routes=routes,
    lifespan=lifespan,
    middleware=middleware,
)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "127.0.0.1")
    url = f"http://{host}:{port}"
    print(f"\n[FusionShield AI] Web Portal starting at: http://localhost:{port} ({url})\n")

    if os.environ.get("NO_BROWSER") != "1":
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    uvicorn.run(app, host=host, port=port, log_level="info")
