# 🧠 Visual Question Answering API

This repository provides a Visual Question Answering (VQA) system built with FastAPI. It combines **YOLOv8** for object detection and **BLIP (BLIP-1)** for answering questions about the image. For complex reasoning tasks, it optionally uses **DeepSeek** as a fallback language model. Caching is handled using SQLite.

---

## 🔍 Features

- ✅ **Object Detection** with YOLOv8 (`yolov8m.pt`)
- 🧠 **Visual Question Answering** with BLIP (`Salesforce/blip-vqa-base`)
- 💬 **Reasoning Support** via DeepSeek API
- 🔁 **Smart Caching** with SQLite to reduce redundant computation
- ❓ Handles **yes/no object presence** questions efficiently

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Rohan8OnThePitch/Internship.git
cd Internship
conda create -n vqa-env python=3.10 -y
conda activate vqa-env
pip install fastapi uvicorn torch pillow transformers ultralytics requests
```

### 2. Model Setup

- **YOLOv8**: Automatically downloads `yolov8m.pt` on first run
- **BLIP**: Automatically downloads `Salesforce/blip-vqa-base` via Hugging Face

---

## 🚀 Running the API

```bash
uvicorn main:app --reload --port 8000
```

Visit: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for interactive Swagger UI.

---

## 📋 API Usage

### Endpoint: `POST /analyze/`

**Form Data:**
- `file`: Image file (e.g., `.jpg`, `.png`)
- `question`: Natural language question (e.g., "What is the person doing?")

### Sample Request (via curl)

```bash
curl -X POST "http://127.0.0.1:8000/analyze/" \
  -F "file=@sample.jpg" \
  -F "question=What is the person doing?"
```

### Sample Response

```json
{
  "answer": "The person is riding a bicycle.",
  "source": "blip",
  "objects": "person, bicycle, road"
}
```

---

## 🧠 How It Works

1. **Image Hashing** is used to cache previous answers
2. **YOLOv8** detects objects and summarizes them
3. **BLIP** answers visual questions
4. If BLIP's response is too generic or the question is reasoning-heavy, **DeepSeek** is invoked

---

## 🛠️ Technical Architecture

The system follows a multi-stage approach:

1. **Preprocessing**: Image is hashed for caching purposes
2. **Object Detection**: YOLOv8 identifies and localizes objects in the image
3. **Question Analysis**: The system determines the best approach for answering
4. **Answer Generation**: BLIP provides initial answers, with DeepSeek as fallback
5. **Caching**: Results are stored in SQLite for future queries

---

## 📝 Notes

- The system is optimized for both simple object presence queries and complex reasoning tasks
- Caching significantly improves response times for repeated queries
- The multi-model approach ensures robust performance across different question types
