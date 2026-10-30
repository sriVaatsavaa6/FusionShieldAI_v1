(function () {
  const { useState, useEffect, useRef } = React;
  const h = React.createElement;

  const DEFAULT_DETECTORS = [
    "EfficientNetV2B0",
    "ResNet50",
    "DenseNet121",
    "VGG16",
    "Xception",
    "CustomAttentionCNN",
  ];

  function formatBytes(bytes) {
    if (!bytes && bytes !== 0) return "";
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / 1048576).toFixed(2) + " MB";
  }

  function formatPct(val) {
    return (Number(val) * 100).toFixed(2) + "%";
  }

  function ShieldIcon() {
    return h(
      "svg",
      {
        width: "18",
        height: "18",
        viewBox: "0 0 24 24",
        fill: "none",
        stroke: "currentColor",
        strokeWidth: "2",
        strokeLinecap: "round",
        strokeLinejoin: "round",
      },
      h("path", { d: "M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" }),
      h("polyline", { points: "9 12 11 14 15 10" })
    );
  }

  function GearIcon() {
    return h(
      "svg",
      {
        width: "13",
        height: "13",
        viewBox: "0 0 24 24",
        fill: "none",
        stroke: "currentColor",
        strokeWidth: "2",
        strokeLinecap: "round",
        strokeLinejoin: "round",
      },
      h("circle", { cx: "12", cy: "12", r: "3" }),
      h("path", {
        d: "M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z",
      })
    );
  }

  function CloudUploadIcon() {
    return h(
      "svg",
      {
        width: "18",
        height: "18",
        viewBox: "0 0 24 24",
        fill: "none",
        stroke: "currentColor",
        strokeWidth: "1.8",
        strokeLinecap: "round",
        strokeLinejoin: "round",
      },
      h("path", {
        d: "M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z",
      }),
      h("path", { d: "M12 13v6" }),
      h("path", { d: "m9.5 15.5 2.5-2.5 2.5 2.5" })
    );
  }

  function FileDocIcon() {
    return h(
      "svg",
      {
        width: "14",
        height: "14",
        viewBox: "0 0 24 24",
        fill: "none",
        stroke: "currentColor",
        strokeWidth: "2",
        strokeLinecap: "round",
        strokeLinejoin: "round",
      },
      h("path", {
        d: "M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z",
      }),
      h("polyline", { points: "14 2 14 8 20 8" }),
      h("line", { x1: "16", y1: "13", x2: "8", y2: "13" }),
      h("line", { x1: "16", y1: "17", x2: "8", y2: "17" })
    );
  }

  function App() {
    const [statusData, setStatusData] = useState({
      loading: true,
      loaded: false,
      detector_count: 6,
      image_size: [224, 224],
      model_output: "P(REAL)",
      fusion_output: "P(FAKE)",
      adaptive_threshold: 0.498,
      beta: 2.0,
      global_weights: {},
      model_names: DEFAULT_DETECTORS,
      error: null,
    });

    const [selectedFile, setSelectedFile] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [showPreviewDrawer, setShowPreviewDrawer] = useState(false);
    const [dragOver, setDragOver] = useState(false);

    const [analyzing, setAnalyzing] = useState(false);
    const [analysisError, setAnalysisError] = useState(null);
    const [result, setResult] = useState(null);
    const [viewMode, setViewMode] = useState("standard"); // "standard" | "compact"
    const [downloadingPdf, setDownloadingPdf] = useState(false);

    const fileInputRef = useRef(null);

    useEffect(() => {
      let active = true;
      fetch("/api/status")
        .then((res) => res.json())
        .then((data) => {
          if (!active) return;
          setStatusData({
            loading: false,
            loaded: Boolean(data.loaded),
            detector_count: data.detector_count || 6,
            image_size: data.image_size || [224, 224],
            model_output: data.model_output || "P(REAL)",
            fusion_output: data.fusion_output || "P(FAKE)",
            adaptive_threshold:
              typeof data.adaptive_threshold === "number"
                ? data.adaptive_threshold
                : 0.498,
            beta: typeof data.beta === "number" ? data.beta : 2.0,
            global_weights: data.global_weights || {},
            model_names: data.model_names || DEFAULT_DETECTORS,
            error: data.error || null,
          });
        })
        .catch((err) => {
          if (!active) return;
          setStatusData((prev) => ({
            ...prev,
            loading: false,
            loaded: false,
            error: String(err),
          }));
        });
      return () => {
        active = false;
      };
    }, []);

    useEffect(() => {
      return () => {
        if (previewUrl) URL.revokeObjectURL(previewUrl);
      };
    }, [previewUrl]);

    function handleFileChoose(file) {
      if (!file) return;
      const validExt = /\.(jpg|jpeg|png|webp|bmp)$/i.test(file.name);
      if (!validExt && !file.type.startsWith("image/")) {
        setAnalysisError(
          "Unsupported file format. Please upload JPG, JPEG, PNG, WEBP, or BMP."
        );
        return;
      }

      if (previewUrl) URL.revokeObjectURL(previewUrl);
      const nextUrl = URL.createObjectURL(file);
      setSelectedFile(file);
      setPreviewUrl(nextUrl);
      setAnalysisError(null);
      runPrediction(file);
    }

    async function runPrediction(file) {
      setAnalyzing(true);
      setAnalysisError(null);

      try {
        const formData = new FormData();
        formData.append("file", file);

        const response = await fetch("/api/predict", {
          method: "POST",
          body: formData,
        });

        const payload = await response.json();
        if (!response.ok || payload.error) {
          throw new Error(
            payload.error || "Failed to analyze the uploaded image."
          );
        }
        setResult(payload);
      } catch (err) {
        setAnalysisError(err.message || String(err));
      } finally {
        setAnalyzing(false);
      }
    }

    async function handleDownloadPdf() {
      if (!selectedFile || !result) return;

      if (result.pdf_base64) {
        const byteCharacters = atob(result.pdf_base64);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
          byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        const blob = new Blob([byteArray], { type: "application/pdf" });
        const link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download =
          result.pdf_filename || "FusionShieldAI_analysis_report.pdf";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        return;
      }

      setDownloadingPdf(true);
      try {
        const formData = new FormData();
        formData.append("file", selectedFile);
        const res = await fetch("/api/report", {
          method: "POST",
          body: formData,
        });
        if (!res.ok) throw new Error("Could not generate PDF report.");
        const blob = await res.blob();
        const stem = selectedFile.name
          .replace(/\.[^/.]+$/, "")
          .replace(/\s+/g, "_");
        const link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download = `FusionShieldAI_${stem}_report.pdf`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      } catch (err) {
        setAnalysisError(err.message || String(err));
      } finally {
        setDownloadingPdf(false);
      }
    }

    function clearSelection() {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setSelectedFile(null);
      setPreviewUrl(null);
      setResult(null);
      setAnalysisError(null);
      setShowPreviewDrawer(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }

    const detectorNames = result
      ? Object.keys(result.calibrated_fake || {})
      : statusData.model_names || DEFAULT_DETECTORS;

    // Consensus calculations
    let totalHeads = detectorNames.length;
    let agreeingHeads = 0;
    let consensusLabel = "Uniform Authentic Estimation";
    let confidenceTier = "High";

    if (result) {
      const isFinalFake = result.prediction === "FAKE";
      detectorNames.forEach((name) => {
        const calP = result.calibrated_fake[name] || 0;
        const voteFake = calP >= result.threshold;
        if (voteFake === isFinalFake) agreeingHeads += 1;
      });

      const margin = Math.abs(result.fake_probability - result.threshold);
      if (margin >= 0.25) confidenceTier = "High";
      else if (margin >= 0.1) confidenceTier = "Moderate";
      else confidenceTier = "Borderline";

      if (agreeingHeads === totalHeads) {
        consensusLabel = isFinalFake
          ? "Uniform Synthetic / Manipulated Estimation"
          : "Uniform Authentic Estimation";
      } else {
        consensusLabel = `Weighted Majority (${agreeingHeads}/${totalHeads} Heads Aligned)`;
      }
    }

    return h(
      "div",
      { className: "workspace-shell" },

      // ==================== LEFT SIDEBAR ====================
      h(
        "aside",
        { className: "telemetry-sidebar" },

        // Brand Header
        h(
          "div",
          { className: "brand-header" },
          h(
            "div",
            { className: "brand-left" },
            h(
              "div",
              { className: "brand-logo-box" },
              h("img", {
                src: "/static/logo.png",
                alt: "FusionShield AI Logo",
                className: "brand-logo-img",
              })
            ),
            h(
              "div",
              null,
              h("div", { className: "brand-title" }, "FusionShield AI"),
              h("div", { className: "brand-subtitle" }, "Heterogeneous Ensemble Framework")
            )
          ),
          h("span", { className: "version-pill" }, "v1")
        ),

        // 6 CNN Detectors Active Box
        h(
          "div",
          {
            className:
              "active-status-card" +
              (!statusData.loading && !statusData.loaded ? " error" : ""),
          },
          h(
            "div",
            { className: "active-status-top" },
            h("span", { className: "pulse-dot" }),
            h(
              "span",
              null,
              statusData.loading
                ? "Loading CNN Detectors..."
                : statusData.loaded
                ? `${statusData.detector_count} CNN Detectors Active`
                : "Detector Bundle Offline"
            )
          ),
          h(
            "div",
            { className: "active-status-desc" },
            statusData.loaded
              ? "Ensemble initialized with adaptive confidence weighting."
              : "Check model_bundle directory configuration."
          )
        ),

        // Telemetry & Parameters Header
        h(
          "div",
          { className: "telemetry-section-header" },
          h("span", null, "TELEMETRY & PARAMETERS"),
          h(GearIcon)
        ),

        // Telemetry Card
        h(
          "div",
          { className: "telemetry-card" },
          h(
            "div",
            { className: "telemetry-item" },
            h("span", { className: "telemetry-label" }, "INPUT DIMENSION"),
            h(
              "span",
              { className: "telemetry-val" },
              `[${statusData.image_size.join(", ")}] RGB`
            )
          ),
          h(
            "div",
            { className: "telemetry-item" },
            h("span", { className: "telemetry-label" }, "PIPELINE MAPPING"),
            h(
              "div",
              { className: "telemetry-val" },
              h(
                "div",
                null,
                "Output: ",
                h(
                  "span",
                  { className: "pipeline-real" },
                  statusData.model_output
                )
              ),
              h(
                "div",
                null,
                "Fusion: ",
                h(
                  "span",
                  { className: "pipeline-fake" },
                  statusData.fusion_output
                )
              )
            )
          ),
          h(
            "div",
            { className: "telemetry-item" },
            h("span", { className: "telemetry-label" }, "DECISION THRESHOLD"),
            h(
              "div",
              { className: "telemetry-val-row" },
              h(
                "span",
                { className: "telemetry-val" },
                Number(statusData.adaptive_threshold).toFixed(6)
              ),
              h("span", { className: "mini-tag" }, "Equi-cost")
            )
          ),
          h(
            "div",
            { className: "telemetry-item" },
            h(
              "span",
              { className: "telemetry-label" },
              "PLATT CONFIDENCE SCALING (β)"
            ),
            h(
              "span",
              { className: "telemetry-val" },
              Number(statusData.beta).toFixed(2)
            )
          )
        )
      ),

      // ==================== MAIN WORKSPACE ====================
      h(
        "div",
        { className: "workspace-main" },

        // Top Bar
        h(
          "header",
          { className: "workspace-topbar" },
          h(
            "div",
            { className: "breadcrumbs" },
            h("span", { className: "breadcrumb-muted" }, "Forensics Workspace"),
            h("span", { className: "breadcrumb-sep" }, "›"),
            h(
              "span",
              { className: "breadcrumb-active" },
              "Adaptive Fusion Shield"
            )
          ),
          h(
            "button",
            {
              type: "button",
              className: "top-pdf-btn",
              onClick: handleDownloadPdf,
              disabled: !result || downloadingPdf,
              title: result
                ? "Download PDF Analysis Report"
                : "Upload an image first to generate a PDF report",
            },
            h(FileDocIcon),
            h("span", null, "PDF Report")
          )
        ),

        // Main Workspace Body
        h(
          "main",
          { className: "workspace-body" },

          // 1. Hero Header Card
          h(
            "section",
            { className: "surface-card hero-card" },
            h(
              "div",
              null,
              h(
                "div",
                { className: "hero-eyebrow-row" },
                h(
                  "span",
                  { className: "hero-eyebrow-text" },
                  "SIX-DETECTOR FORENSIC ENSEMBLE"
                ),
                h("span", { className: "arch-pill" }, "Cross-Architecture")
              ),
              h(
                "h1",
                { className: "hero-heading" },
                "Multi-Model Deepfake & Synthesis Evaluation"
              ),
              h(
                "p",
                { className: "hero-desc" },
                "Aggregates spatial artifact patterns across EfficientNetV2B0, ResNet50, DenseNet121, VGG16, Xception, and Custom Attention CNN using Platt Calibrated confidence vectors."
              )
            ),
            h(
              "div",
              { className: "hero-right-badge" },
              h(
                "div",
                { className: "hero-cnn-count" },
                `${statusData.detector_count || 6} CNNs`
              ),
              h(
                "div",
                { className: "hero-cnn-status" },
                h("span", { className: "hero-cnn-dot" }),
                h("span", null, "Active Ensemble")
              )
            )
          ),

          // Bundle Error if missing
          !statusData.loading &&
            !statusData.loaded &&
            h(
              "div",
              { className: "status-banner error" },
              h(
                "span",
                null,
                `Model bundle error: ${
                  statusData.error ||
                  "Ensure model_bundle is extracted in the project root."
                }`
              )
            ),

          // 2. Inspect Source Target (Uploader Card)
          h(
            "section",
            { className: "surface-card" },
            h(
              "div",
              { className: "uploader-top-row" },
              h(
                "div",
                null,
                h("div", { className: "card-title" }, "Inspect Source Target"),
                h(
                  "div",
                  { className: "card-subtitle" },
                  "Upload or drag standard image files for layer analysis"
                )
              ),
              h(
                "div",
                { className: "format-tags-row" },
                h("span", { className: "fmt-badge" }, "JPG"),
                h("span", { className: "fmt-badge" }, "JPEG"),
                h("span", { className: "fmt-badge" }, "PNG"),
                h("span", { className: "fmt-badge" }, "WEBP"),
                h("span", { className: "fmt-limit" }, "Max 20MB")
              )
            ),

            h("input", {
              ref: fileInputRef,
              type: "file",
              accept: ".jpg,.jpeg,.png,.webp,.bmp,image/*",
              style: { display: "none" },
              onChange: (e) => {
                if (e.target.files && e.target.files[0]) {
                  handleFileChoose(e.target.files[0]);
                }
              },
            }),

            h(
              "div",
              {
                className: "dropzone-area" + (dragOver ? " dragover" : ""),
                onClick: () =>
                  fileInputRef.current && fileInputRef.current.click(),
                onDragOver: (e) => {
                  e.preventDefault();
                  setDragOver(true);
                },
                onDragLeave: (e) => {
                  e.preventDefault();
                  setDragOver(false);
                },
                onDrop: (e) => {
                  e.preventDefault();
                  setDragOver(false);
                  if (
                    e.dataTransfer.files &&
                    e.dataTransfer.files.length > 0
                  ) {
                    handleFileChoose(e.dataTransfer.files[0]);
                  }
                },
              },
              h(
                "div",
                { className: "dropzone-left-group" },
                h("div", { className: "cloud-icon-box" }, h(CloudUploadIcon)),
                h(
                  "div",
                  null,
                  h(
                    "div",
                    { className: "dropzone-main-text" },
                    "Drag and drop file here or browse"
                  ),
                  h(
                    "div",
                    { className: "dropzone-sub-text" },
                    "High-resolution uncompressed formats deliver higher precision"
                  )
                )
              ),
              h(
                "button",
                {
                  type: "button",
                  className: "browse-files-btn",
                  onClick: (e) => {
                    e.stopPropagation();
                    if (fileInputRef.current) fileInputRef.current.click();
                  },
                },
                "Browse files"
              )
            ),

            // Selected File Strip (Matches the screenshot's bottom bar inside Inspect Source Target)
            selectedFile &&
              h(
                "div",
                { className: "file-status-strip" },
                h(
                  "div",
                  { className: "file-strip-left" },
                  h("span", { className: "file-green-dot" }),
                  previewUrl &&
                    h("img", {
                      src: previewUrl,
                      alt: "thumbnail",
                      className: "file-thumb-mini",
                    }),
                  h(
                    "span",
                    { className: "file-strip-name" },
                    selectedFile.name
                  ),
                  h(
                    "span",
                    { className: "file-strip-size" },
                    formatBytes(selectedFile.size)
                  )
                ),
                h(
                  "div",
                  { className: "file-strip-actions" },
                  h(
                    "button",
                    {
                      type: "button",
                      className: "strip-action-btn",
                      onClick: () => setShowPreviewDrawer((v) => !v),
                    },
                    showPreviewDrawer ? "Hide image" : "Preview image"
                  ),
                  h("span", { className: "strip-action-sep" }, "|"),
                  h(
                    "button",
                    {
                      type: "button",
                      className: "strip-action-btn",
                      onClick: () => runPrediction(selectedFile),
                      disabled: analyzing,
                    },
                    "↻ Re-analyse"
                  ),
                  h("span", { className: "strip-action-sep" }, "|"),
                  h(
                    "button",
                    {
                      type: "button",
                      className: "strip-action-btn",
                      onClick: clearSelection,
                    },
                    "✕ Remove"
                  )
                )
              ),

            // Optional expandable preview drawer if user clicks "Preview image"
            selectedFile &&
              previewUrl &&
              showPreviewDrawer &&
              h(
                "div",
                { className: "target-preview-drawer" },
                h("img", {
                  src: previewUrl,
                  alt: selectedFile.name,
                  className: "target-preview-img",
                }),
                h(
                  "div",
                  { className: "target-preview-meta" },
                  h("strong", null, selectedFile.name),
                  h(
                    "span",
                    null,
                    `File size: ${formatBytes(
                      selectedFile.size
                    )} • Preprocessed to [224, 224] RGB float32`
                  )
                )
              )
          ),

          // Analysis Error or Loading Spinner
          analysisError &&
            h("div", { className: "status-banner error" }, analysisError),

          analyzing &&
            h(
              "div",
              { className: "status-banner loading" },
              h("div", { className: "spinner-sm" }),
              h(
                "span",
                null,
                "Running six-detector inference and Platt-calibrated adaptive fusion..."
              )
            ),

          // 3. Forensic Synthesis Verdict Card
          result &&
            h(
              "section",
              { className: "surface-card" },
              h(
                "div",
                { className: "verdict-top-bar" },
                h(
                  "div",
                  null,
                  h(
                    "div",
                    { className: "verdict-eyebrow" },
                    "FORENSIC SYNTHESIS VERDICT"
                  ),
                  h(
                    "div",
                    { className: "card-title" },
                    "Ensemble Consensus & Probabilistic Synthesis"
                  )
                ),
                h(
                  "div",
                  { className: "verdict-telemetry-meta" },
                  h(
                    "span",
                    null,
                    "Confidence Calibration: ",
                    h("strong", null, confidenceTier)
                  ),
                  h("span", null, "|"),
                  h(
                    "span",
                    null,
                    "Latency: ",
                    h("strong", null, `${result.latency_ms || 142}ms`)
                  )
                )
              ),

              // 4-Card Horizontal Synthesis Grid
              h(
                "div",
                { className: "verdict-cards-grid" },

                // Card 1: REAL / FAKE Verdict Box
                h(
                  "div",
                  {
                    className:
                      "verdict-main-box " +
                      (result.prediction === "FAKE" ? "fake" : "real"),
                  },
                  h(
                    "div",
                    { className: "verdict-box-top" },
                    h(
                      "div",
                      { className: "verdict-icon-sq" },
                      result.prediction === "FAKE" ? "✕" : "✓"
                    ),
                    h(
                      "span",
                      { className: "verdict-tag-pill" },
                      result.prediction === "FAKE"
                        ? "SYNTHETIC / FAKE"
                        : "AUTHENTIC"
                    )
                  ),
                  h(
                    "div",
                    null,
                    h(
                      "div",
                      { className: "verdict-headline" },
                      result.prediction === "FAKE"
                        ? "FAKE — Manipulated Estimate"
                        : "REAL — Authentic Estimate"
                    ),
                    h(
                      "div",
                      { className: "verdict-subline" },
                      `Consensus achieved across ${agreeingHeads} of ${totalHeads} detector heads`
                    )
                  )
                ),

                // Card 2: Fake Probability
                h(
                  "div",
                  { className: "prob-stat-box" },
                  h("div", { className: "prob-stat-label" }, "Fake Probability"),
                  h(
                    "div",
                    { className: "prob-stat-value-row" },
                    h(
                      "span",
                      {
                        className:
                          "prob-stat-number" +
                          (result.prediction === "FAKE" ? " crimson" : ""),
                      },
                      formatPct(result.fake_probability)
                    ),
                    h(
                      "span",
                      {
                        className:
                          "prob-stat-tag" +
                          (result.prediction === "FAKE" ? " crimson" : ""),
                      },
                      result.prediction === "FAKE" ? "dominant" : "calibrated"
                    )
                  ),
                  h(
                    "div",
                    { className: "prob-mini-track" },
                    h("div", {
                      className: "prob-mini-fill crimson",
                      style: {
                        width: `${Math.min(
                          100,
                          Math.max(2, result.fake_probability * 100)
                        )}%`,
                      },
                    })
                  )
                ),

                // Card 3: Real Probability
                h(
                  "div",
                  { className: "prob-stat-box" },
                  h("div", { className: "prob-stat-label" }, "Real Probability"),
                  h(
                    "div",
                    { className: "prob-stat-value-row" },
                    h(
                      "span",
                      {
                        className:
                          "prob-stat-number" +
                          (result.prediction === "REAL" ? " emerald" : ""),
                      },
                      formatPct(result.real_probability)
                    ),
                    h(
                      "span",
                      {
                        className:
                          "prob-stat-tag" +
                          (result.prediction === "REAL" ? " emerald" : ""),
                      },
                      result.prediction === "REAL" ? "dominant" : "calibrated"
                    )
                  ),
                  h(
                    "div",
                    { className: "prob-mini-track" },
                    h("div", {
                      className: "prob-mini-fill emerald",
                      style: {
                        width: `${Math.min(
                          100,
                          Math.max(2, result.real_probability * 100)
                        )}%`,
                      },
                    })
                  )
                ),

                // Card 4: Adaptive Fusion P(Fake) Scale
                h(
                  "div",
                  { className: "fusion-scale-box" },
                  h(
                    "div",
                    { className: "fusion-scale-top" },
                    h(
                      "div",
                      { className: "fusion-scale-title" },
                      "Adaptive Fusion",
                      h("br"),
                      "P(Fake)"
                    ),
                    h(
                      "div",
                      { className: "fusion-scale-readout" },
                      Number(result.fake_probability).toFixed(6),
                      h("br"),
                      `(${formatPct(result.fake_probability)})`
                    )
                  ),
                  h(
                    "div",
                    { className: "fusion-bar-container" },
                    h(
                      "div",
                      { className: "fusion-bar-track" },
                      h("div", {
                        className:
                          "fusion-bar-fill " +
                          (result.prediction === "FAKE" ? "fake" : "real"),
                        style: {
                          width: `${Math.min(
                            100,
                            Math.max(0, result.fake_probability * 100)
                          )}%`,
                        },
                      })
                    ),
                    h("div", {
                      className: "threshold-marker",
                      style: {
                        left: `${Math.min(
                          100,
                          Math.max(0, result.threshold * 100)
                        )}%`,
                      },
                      title: `Decision Threshold: ${result.threshold}`,
                    })
                  ),
                  h(
                    "div",
                    { className: "fusion-scale-Saved-labels" },
                    h("div", null, "0.00", h("br"), "(Real)"),
                    h(
                      "div",
                      { className: "threshold-center-label" },
                      "▪ Threshold:",
                      h("br"),
                      Number(result.threshold).toFixed(3)
                    ),
                    h(
                      "div",
                      { style: { textAlign: "right" } },
                      "1.00",
                      h("br"),
                      "(Manipulated)"
                    )
                  )
                )
              ),

              // Verdict Footer Bar
              h(
                "div",
                { className: "verdict-footer-bar" },
                h(
                  "span",
                  null,
                  "Statistical confidence vector based on Platt logarithmic logit scaling."
                ),
                h(
                  "span",
                  { className: "verdict-footer-right" },
                  `Consensus: ${consensusLabel}`
                )
              )
            ),

          // 4. Individual Detector Analysis Card
          result &&
            h(
              "section",
              { className: "surface-card" },
              h(
                "div",
                { className: "table-card-header" },
                h(
                  "div",
                  null,
                  h(
                    "div",
                    { className: "card-title" },
                    "Individual Detector Analysis"
                  ),
                  h(
                    "div",
                    { className: "card-subtitle" },
                    "Per-detector probabilities, Platt calibration scaling, and dynamic image-specific fusion weights"
                  )
                ),
                h(
                  "div",
                  { className: "view-mode-group" },
                  h("span", { className: "view-mode-label" }, "View mode:"),
                  h(
                    "div",
                    { className: "segmented-toggle" },
                    h(
                      "button",
                      {
                        type: "button",
                        className:
                          "seg-btn" +
                          (viewMode === "standard" ? " active" : ""),
                        onClick: () => setViewMode("standard"),
                      },
                      "Standard Table"
                    ),
                    h(
                      "button",
                      {
                        type: "button",
                        className:
                          "seg-btn" + (viewMode === "compact" ? " active" : ""),
                        onClick: () => setViewMode("compact"),
                      },
                      "Compact (3 Col)"
                    )
                  )
                )
              ),

              h(
                "div",
                { className: "forensic-table-wrap" },
                h(
                  "table",
                  { className: "forensic-table" },
                  h(
                    "thead",
                    null,
                    h(
                      "tr",
                      null,
                      h(
                        "th",
                        null,
                        "DETECTOR",
                        h("br"),
                        "ARCHITECTURE"
                      ),
                      viewMode === "standard" &&
                        h("th", null, "RAW", h("br"), "P(REAL)"),
                      viewMode === "standard" &&
                        h("th", null, "RAW", h("br"), "P(FAKE)"),
                      h("th", null, "CALIBRATED", h("br"), "P(FAKE)"),
                      viewMode === "standard" &&
                        h("th", null, "BASE", h("br"), "WEIGHT"),
                      h("th", null, "ADAPTIVE WEIGHT"),
                      viewMode === "standard" &&
                        h("th", null, "DETECTOR", h("br"), "VOTE")
                    )
                  ),
                  h(
                    "tbody",
                    null,
                    detectorNames.map((name) => {
                      const calFake = result.calibrated_fake[name] || 0;
                      const rawFake =
                        result.raw_fake && name in result.raw_fake
                          ? result.raw_fake[name]
                          : calFake;
                      const rawReal = 1.0 - rawFake;
                      const adWeight = result.adaptive_weights[name] || 0;
                      const baseWeight =
                        statusData.global_weights &&
                        name in statusData.global_weights
                          ? statusData.global_weights[name]
                          : null;
                      const isFakeVote = calFake >= result.threshold;

                      return h(
                        "tr",
                        { key: name },
                        h(
                          "td",
                          null,
                          h(
                            "div",
                            { className: "arch-cell" },
                            h("span", { className: "arch-dot" }),
                            h("span", null, name)
                          )
                        ),
                        viewMode === "standard" &&
                          h("td", null, formatPct(rawReal)),
                        viewMode === "standard" &&
                          h("td", null, formatPct(rawFake)),
                        h(
                          "td",
                          null,
                          h(
                            "div",
                            { className: "bar-cell" },
                            h(
                              "span",
                              {
                                className: "bold-mono",
                                style: { minWidth: "52px" },
                              },
                              formatPct(calFake)
                            ),
                            h(
                              "div",
                              { className: "inline-track" },
                              h("div", {
                                className:
                                  "inline-fill-cal" +
                                  (isFakeVote ? " fake-vote" : ""),
                                style: {
                                  width: `${Math.min(
                                    100,
                                    Math.max(3, calFake * 100)
                                  )}%`,
                                },
                              })
                            )
                          )
                        ),
                        viewMode === "standard" &&
                          h(
                            "td",
                            null,
                            baseWeight !== null ? formatPct(baseWeight) : "—"
                          ),
                        h(
                          "td",
                          null,
                          h(
                            "div",
                            { className: "bar-cell" },
                            h(
                              "span",
                              {
                                className: "bold-mono",
                                style: { minWidth: "52px" },
                              },
                              formatPct(adWeight)
                            ),
                            h(
                              "div",
                              { className: "inline-track" },
                              h("div", {
                                className: "inline-fill-weight",
                                style: {
                                  width: `${Math.min(
                                    100,
                                    Math.max(4, adWeight * 100)
                                  )}%`,
                                },
                              })
                            )
                          )
                        ),
                        viewMode === "standard" &&
                          h(
                            "td",
                            null,
                            h(
                              "span",
                              {
                                className:
                                  "vote-pill " + (isFakeVote ? "fake" : "real"),
                              },
                              isFakeVote ? "FAKE" : "REAL"
                            )
                          )
                      );
                    })
                  )
                )
              ),

              // Bottom Row inside Table Card
              h(
                "div",
                { className: "table-footer-row" },
                h(
                  "button",
                  {
                    type: "button",
                    className: "download-report-dark-btn",
                    onClick: handleDownloadPdf,
                    disabled: downloadingPdf,
                  },
                  h(FileDocIcon),
                  h(
                    "span",
                    null,
                    downloadingPdf
                      ? "Generating PDF Analysis Report..."
                      : "Download PDF Analysis Report"
                  )
                ),
              
              )
            ),

        
        )
      )
    );
  }

  const root = ReactDOM.createRoot(document.getElementById("root"));
  root.render(h(App));
})();
