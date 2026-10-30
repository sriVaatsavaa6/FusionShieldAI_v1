# ============================================================
# FUSIONSHEILD AI — EXPORT TRAINED MODELS FOR STREAMLIT
# Run this cell in the SAME Colab notebook/runtime where:
# detectors, calibrators, global_weights, BETA and
# method_thresholds are already defined.
# ============================================================

import os, json, shutil, zipfile
from pathlib import Path
import joblib

EXPORT_ROOT = Path("/content/FusionShieldAI_Streamlit_Export")
MODEL_DIR = EXPORT_ROOT / "model_bundle" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Map the exact detector dictionary keys to portable filenames.
MODEL_FILES = {
    "EfficientNetV2B0": "efficientnetv2b0.keras",
    "ResNet50": "resnet50.keras",
    "DenseNet121": "densenet121.keras",
    "VGG16": "vgg16.keras",
    "Xception": "xception.keras",
    "CustomAttentionCNN": "custom_attention_cnn.keras",
}

# Save all six already-trained models. This does NOT retrain them.
for model_name, filename in MODEL_FILES.items():
    if model_name not in detectors:
        raise KeyError(f"Detector missing from `detectors`: {model_name}")

    save_path = MODEL_DIR / filename
    detectors[model_name].save(save_path)
    print("Saved:", model_name, "->", save_path)

# Save the same Platt calibration parameters used in the notebook.
for model_name in MODEL_FILES:
    if model_name not in calibrators:
        raise KeyError(f"Calibrator missing: {model_name}")
    if not all(k in calibrators[model_name] for k in ("slope", "intercept")):
        raise ValueError(
            f"{model_name} calibrator is not the expected Platt "
            "{'slope': ..., 'intercept': ...} format."
        )

joblib.dump(
    {name: calibrators[name] for name in MODEL_FILES},
    EXPORT_ROOT / "model_bundle" / "calibrators.joblib"
)

# Store exactly the learned fusion settings and selected validation threshold.
adaptive_threshold = float(method_thresholds["Adaptive Fusion"])

fusion_config = {
    "project": "FusionShield AI",
    "image_size": [224, 224],
    "model_names": list(MODEL_FILES.keys()),
    "model_output": "P(REAL)",
    "fusion_output": "P(FAKE)",
    "class_mapping": {"0": "Fake", "1": "Real"},
    "global_weights": {
        name: float(global_weights[name]) for name in MODEL_FILES
    },
    "alpha": float(ALPHA),
    "beta": float(BETA),
    "adaptive_threshold": adaptive_threshold,
    "threshold_source": "Validation-only accuracy optimization",
    "tensorflow_version": tf.__version__
}

with open(
    EXPORT_ROOT / "model_bundle" / "fusion_config.json",
    "w"
) as f:
    json.dump(fusion_config, f, indent=2)

# Create one portable ZIP containing all model weights and fusion artifacts.
zip_path = "/content/FusionShieldAI_model_bundle.zip"
if os.path.exists(zip_path):
    os.remove(zip_path)

shutil.make_archive(
    "/content/FusionShieldAI_model_bundle",
    "zip",
    root_dir=EXPORT_ROOT,
    base_dir="model_bundle"
)

print("\nExport complete:", zip_path)
print("Bundle size (MB):", round(os.path.getsize(zip_path) / (1024**2), 2))

from google.colab import files
files.download(zip_path)
