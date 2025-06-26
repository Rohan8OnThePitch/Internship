import io
import sqlite3
import torch
import requests
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForQuestionAnswering

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
blip_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

# --- Model Setup ---
yolo_model = YOLO("yolov8m.pt")

# --- Caching ---
conn = sqlite3.connect("cache.db", check_same_thread=False)
c = conn.cursor()
c.execute('''
CREATE TABLE IF NOT EXISTS cache (
    image_hash TEXT,
    question TEXT,
    answer TEXT,
    PRIMARY KEY (image_hash, question)
)
''')
conn.commit()

def get_cache(image_hash, question):
    c.execute("SELECT answer FROM cache WHERE image_hash=? AND question=?", (image_hash, question))
    row = c.fetchone()
    return row[0] if row else None

def set_cache(image_hash, question, answer):
    c.execute("INSERT OR REPLACE INTO cache VALUES (?, ?, ?)", (image_hash, question, answer))
    conn.commit()

def hash_image(image: Image.Image):
    return str(hash(image.tobytes()))

def crop_image(image: Image.Image, bbox):
    return image.crop((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))

# --- Heuristics ---
def is_reasoning_question(question: str):
    reasoning_keywords = ["why", "how", "describe", "doing", "happening", "scene", "event"]
    return any(kw in question.lower() for kw in reasoning_keywords) or len(question.split()) > 12

def is_generic_blip_answer(answer: str, question: str):
    answer = answer.strip().lower()
    question = question.strip().lower()
    generic = ["object", "thing", "person", "people", "someone", "something", "none"]
    return (
        answer in generic or
        len(answer) < 4 or
        answer == question or
        answer in question
    )

# --- BLIP-2 VQA ---
def blip_vqa(image: Image.Image, question: str):
    inputs = blip_processor(image, question, return_tensors="pt").to(blip_model.device)
    with torch.no_grad():
        out = blip_model.generate(**inputs)
    return blip_processor.decode(out[0], skip_special_tokens=True).strip()

# --- DeepSeek Fallback ---
def call_deepseek(prompt):
    API_URL = "https://api.deepseek.com/v1/chat/completions"
    API_KEY = "your-deepseek-key-here"  # <-- replace with actual key
    headers = {"Authorization": f"Bearer {API_KEY}"}
    data = {
        "messages": [{"role": "user", "content": prompt}],
        "model": "deepseek-chat"
    }
    response = requests.post(API_URL, headers=headers, json=data)
    if response.ok:
        return response.json()["choices"][0]["message"]["content"]
    return "Sorry, couldn't get an answer from DeepSeek."

# --- FastAPI App ---
app = FastAPI()

@app.post("/analyze/")
async def analyze(
    file: UploadFile = File(...),
    question: str = Form("What is this image about?")
):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image_hash = hash_image(image)

    # Step 1: YOLO object detection
    results = yolo_model(np.array(image))
    boxes = results[0].boxes
    labels = []
    if boxes is not None and boxes.cls is not None:
        labels = [results[0].names[int(cls)] for cls in boxes.cls.cpu().numpy()]
    label_set = set(labels)
    detected_summary = ", ".join(label_set)
    
    # Step 2: Cache lookup
    cache_key = image_hash + question
    cached = get_cache(cache_key, question)
    if cached:
        return JSONResponse({"answer": cached, "source": "cache", "object": detected_summary})

    # Step 3: Check for yes/no presence questions
    lower_q = question.lower()
    if lower_q.startswith("is there") or lower_q.startswith("are there"):
        for label in label_set:
            if label in lower_q:
                answer = f"Yes, a {label} is present in the image."
                set_cache(cache_key, question, answer)
                return JSONResponse({"answer": answer, "source": "yolo", "object": detected_summary})
        answer = "No, that object is not present in the image."
        set_cache(cache_key, question, answer)
        return JSONResponse({"answer": answer, "source": "yolo", "object": detected_summary})

    # Step 4: Use BLIP-2 for visual question answering
    scene_summary = blip_vqa(image, "Describe this image briefly.")
    local_answer = blip_vqa(image, question)
    print(f"[DEBUG] BLIP answer: '{local_answer}'")

    # Step 5: Heuristics for DeepSeek fallback
    def is_reasoning_question(q: str):
        reasoning_keywords = ["why", "how", "doing", "happening", "event", "describe", "feeling", "emotion"]
        return any(kw in q.lower() for kw in reasoning_keywords) or len(q.split()) > 12

    def is_generic_blip_answer(answer: str):
        generic = ["object", "thing", "person", "people", "someone", "something"]
        return answer.strip().lower() in generic or len(answer.strip()) < 4

    if is_reasoning_question(question) or is_generic_blip_answer(local_answer):
        prompt = f"Scene contains: {detected_summary}\nBLIP Summary: {scene_summary}\nQuestion: {question}"
        final_answer = call_deepseek(prompt)
        source = "generated"
    else:
        final_answer = local_answer
        source = "blip"

    set_cache(cache_key, question, final_answer)
    return JSONResponse({
        "answer": final_answer,
        "source": source,
        "object": scene_summary
    })
