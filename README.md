# 🧠 Alzheimer's Disease Detection using Vision Transformers

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-Optimized-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Render](https://img.shields.io/badge/Deployment-Render-black?logo=render)

</p>

A production-ready deep learning application for **Alzheimer's Disease Stage Classification** from brain MRI images using **Vision Transformers (ViT/DeiT/BEiT)**.

The project includes:

- 🔬 Research prototype built with **PyTorch + Gradio**
- ⚡ Optimized **FastAPI + ONNX Runtime** inference server
- 🌐 Lightweight deployment using **Render**
- 🧠 Quantized model for efficient CPU inference

---

# 📸 Application Preview

<p align="center">
<img src="assets/screenshot2.png" width="95%">
</p>

---

# 📌 Overview

This project classifies brain MRI scans into four Alzheimer's disease stages using pretrained Vision Transformer architectures.

### Predicted Classes

| Class | Description |
|--------|-------------|
| 🟢 Non Demented | Healthy Brain |
| 🟡 Very Mild Dementia | Early Cognitive Impairment |
| 🟠 Mild Dementia | Alzheimer's Disease |
| 🔴 Moderate Dementia | Advanced Alzheimer's |

The application automatically preprocesses uploaded MRI images and returns:

- Predicted stage
- Confidence score
- Class probabilities
- Inference latency

---

# ✨ Features

- 🧠 Vision Transformer-based MRI classification
- ⚡ ONNX Runtime optimized inference
- 🚀 FastAPI REST API
- 🌐 Interactive Web Interface
- 📊 Confidence probability for each class
- 📈 Low latency CPU inference
- 💾 Quantized INT8 model
- ☁️ Cloud deployment ready
- 🔄 Automatic model download from GitHub Releases

---

# 🏗 Project Architecture

```
               MRI Image
                    │
                    ▼
          Image Preprocessing
      Resize → Normalize → Tensor
                    │
                    ▼
         Vision Transformer Model
            (ONNX Runtime CPU)
                    │
                    ▼
      Softmax Probability Prediction
                    │
                    ▼
          JSON Diagnostic Response
```

---

# 🚀 Local Installation

Clone the repository

```bash
git clone https://github.com/manav-g27/deit-oasis-web-app.git
```

Move into the project

```bash
cd deit-oasis-web-app
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

The pretrained model (~330 MB) is downloaded automatically on the first launch.

---

# 🌍 Live Deployment

### Frontend


```
https://huggingface.co/spaces/manav-g27/alzheimers-diagnostic-ui
```

### Backend API


```
https://alzheimers-api-backend.onrender.com/predict
```

---

# 📡 REST API

## POST `/predict`

Upload a brain MRI image.

### Request

```
multipart/form-data

image=<MRI Image>
```

### Response

```json
{
  "prediction": "Very Mild Dementia",
  "confidence": 0.8942,
  "probabilities": {
    "Non Demented": 0.0735,
    "Very Mild Dementia": 0.8942,
    "Mild Dementia": 0.0312,
    "Moderate Dementia": 0.0011
  },
  "inference_time_ms": 42.15
}
```

---

# ⚙️ Technology Stack

## Deep Learning

- Python
- PyTorch
- TorchVision
- timm
- ONNX Runtime

## Backend

- FastAPI
- Uvicorn
- NumPy
- Pillow

## Frontend

- Gradio
- HTML
- CSS

## Deployment

- Render
- GitHub Releases

---

# 📂 Project Structure

```
deit-oasis-web-app/
│
├── app.py
├── predictor.py
├── requirements.txt
├── assets/
│     └── screenshot2.png
│
├── models/
│
├── utils/
│
└── README.md
```

---

# ⚡ Performance Optimizations

- Exported PyTorch model to ONNX
- Dynamic INT8 Quantization
- Optimized CPU inference
- Reduced model size

| Before | After |
|---------|--------|
| 330 MB | 85 MB |
| PyTorch | ONNX Runtime |
| High RAM | Low RAM |
| Slower | Faster |

---

# 🔮 Future Improvements

- Explainable AI (Grad-CAM)
- Batch prediction support
- PostgreSQL integration
- User authentication
- Clinical evaluation using ADNI
- Docker Compose deployment
- GPU inference support

---

# ⚠️ Disclaimer

> **This software is intended solely for research and educational purposes.**

It is **NOT** a medical device and **must not** be used for clinical diagnosis, treatment planning, or patient care.

All predictions should be interpreted only by qualified healthcare professionals.

---

# 📄 License

This project is licensed under the **MIT License**.

---

# 👨‍💻 Author

## Manav Gupta

**M.Tech Computer Science**

Deep Learning • Computer Vision • Medical AI

⭐ If you found this project helpful, please consider giving it a **Star**.
