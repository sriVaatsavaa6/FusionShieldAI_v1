---
title: FusionShield AI
emoji: 🛡️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# FusionShield AI — Web Portal (React + HTML/CSS/JS + Python Backend)

This project launches a local web interface (built with React, modern HTML5/CSS3, and a Starlette/Uvicorn API server) for your **already-trained** six-detector FusionShield AI ensemble. It replaces Streamlit while preserving the exact forensic UI layout, probabilities, Platt calibration, adaptive weights, predictions, and PDF report generation.

## Important: export your trained model bundle first

The source code ZIP intentionally does not contain your trained neural-network weights because those files are created in your Colab runtime and can be very large.

1. Open the Colab notebook in which the six trained models and final fusion values are available.
2. Make sure these variables exist:
   `detectors`, `calibrators`, `global_weights`, `ALPHA`, `BETA`, `method_thresholds`.
3. Open `export_models_from_colab.py` from this project, copy its complete contents into a new Colab cell, and run it.
4. Colab downloads `FusionShieldAI_model_bundle.zip`.
5. Extract that ZIP into this project's root folder. The final layout must be:

   FusionShieldAI/
   - app.py
   - requirements.txt
   - README.md
   - static/
     - index.html
     - styles.css
     - app.js
   - model_bundle/
     - fusion_config.json
     - calibrators.joblib
     - models/
       - efficientnetv2b0.keras
       - resnet50.keras
       - densenet121.keras
       - vgg16.keras
       - xception.keras
       - custom_attention_cnn.keras

## Run on Windows

Install Python 3.10+ (64-bit). Open Command Prompt or PowerShell in this project folder.

Create and activate a virtual environment (if not already created):

```bash
py -3.10 -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Launch the web portal:

```bash
python app.py
```

The server will start at:

`http://127.0.0.1:8000`

It will automatically open in your browser. Upload any JPG/JPEG/PNG/WEBP/BMP image to run the six-detector analysis and download the PDF forensic report.

## Run on macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

## What the app does

- Loads six saved Keras `.keras` models (`EfficientNetV2B0`, `ResNet50`, `DenseNet121`, `VGG16`, `Xception`, `CustomAttentionCNN`).
- Resizes uploaded RGB images to 224 × 224.
- Passes float32 RGB pixel values in the 0–255 range, matching the notebook dataset pipeline. Model-specific preprocessing is embedded in the models.
- Reads each model output as P(REAL), converts it to P(FAKE), and applies the saved Platt calibration.
- Calculates confidence as `abs(P_fake - 0.5) * 2`.
- Applies the saved global weights and beta to calculate image-specific adaptive weights.
- Produces the weighted Adaptive Fusion P(FAKE) score.
- Applies the saved validation-selected Adaptive Fusion threshold.
- Displays individual detector outputs, raw & calibrated probabilities, and adaptive weights.
- Generates a downloadable PDF report.

## Research limitation

FusionShield AI outputs are experimental model estimates, not forensic proof. The system's measured test metrics do not guarantee equivalent performance on unseen generators, edited images, or heavily compressed images.
