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
- 🧠 Quantised model for efficient CPU inference

---

# 🔗 Project Links

| Resource | Link |
|----------|------|
| 🌐 Live Web Application | https://huggingface.co/spaces/manav-g27/alzheimers-diagnostic-ui |
| 🚀 Backend API | https://alzheimers-api-backend.onrender.com |
| 🚀 Deployment Repository | https://github.com/manav-g27/alzheimers-api-backend |

---

# 📸 Local Web App Preview

<p align="center">
<img src="assests/screenshot2.png" width="95%" alt="Application Preview">
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
- ⚡ ONNX Runtime optimised inference
- 🚀 FastAPI REST API
- 🌐 Interactive web interface
- 📊 Confidence probability for each class
- 📈 Low-latency CPU inference
- 💾 Quantized INT8 model
- ☁️ Cloud deployment ready
- 🔄 Automatic model download from GitHub Releases

---

# 🏗 Project Architecture

```text
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

### Clone the repository

```bash
git clone https://github.com/manav-g27/deit-oasis-web-app.git
```

### Move into the project

```bash
cd deit-oasis-web-app
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
python app.py
```

The pretrained model (~330 MB) is downloaded automatically on the first launch.

---

# 🌍 Deployment

| Service | URL |
|---------|-----|
| 🌐 Live Web Application | https://huggingface.co/spaces/manav-g27/alzheimers-diagnostic-ui |
| 🚀 Backend API | https://alzheimers-api-backend.onrender.com |
| 🚀 Deployment Repository | https://github.com/manav-g27/alzheimers-api-backend |

---

# 📡 REST API

## POST `/predict`

Upload a brain MRI image.

### Request

```text
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
- Hugging Face Spaces
- GitHub Releases

---

# 📂 Project Structure

```text
deit-oasis-web-app/
│
├── app.py
├── predictor.py
├── requirements.txt
├── assets/
│   └── screenshot2.png
├── models/
├── utils/
└── README.md
```

---

# ⚡ Performance Optimizations

- Exported PyTorch model to ONNX
- Dynamic INT8 Quantization
- Optimized CPU inference
- Reduced model size and memory usage

| Before | After |
|---------|--------|
| 330 MB | 85 MB |
| PyTorch Runtime | ONNX Runtime |
| High RAM Usage | Low RAM Usage |
| Slower Inference | Faster Inference |

---

# 🔮 Future Improvements

- 🔍 Explainable AI (Grad-CAM / Attention Rollout)
- 📦 Batch prediction support
- 🗄 PostgreSQL integration
- 🔐 User authentication
- 🧪 Clinical validation using ADNI dataset
- 🐳 Docker Compose deployment
- ⚡ GPU inference support

---

# ⚠️ Disclaimer

> **This software is intended solely for research and educational purposes.**

It is **NOT** a medical device and **must not** be used for clinical diagnosis, treatment planning, or patient care.

All predictions generated by this application should be interpreted only by qualified healthcare professionals.

---

# 📄 License

This project is licensed under the **MIT License**.

---

# 👨‍💻 Author

## Manav Gupta

**M.Tech Computer Science**

Deep Learning • Computer Vision • Medical AI

📧 **Email:** mg2002.gupta@gmail.com

🌐 **Portfolio:** https://manavgupta27.netlify.app/

⭐ If you found this project helpful, please consider giving it a **Star**.
