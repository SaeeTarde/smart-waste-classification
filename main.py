#main.py
from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
import io
import os
import sys


# 🔧 STEP 1: Fix project root path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = BASE_DIR
sys.path.append(PROJECT_ROOT)

# 🔧 STEP 2: Import recommendation engine
from garbage_classification.smart_waste_system.recommendation_engine import recommend

# 🔧 STEP 3: Load trained model ONCE
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "garbage_classification",
    "model",
    "garbage_model.keras"
)

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Model load failed: {e}")


import json

with open(os.path.join(
    PROJECT_ROOT,
    "garbage_classification",
    "model",
    "class_indices.json"
)) as f:
    class_indices = json.load(f)

# Convert index → label
labels = {v: k for k, v in class_indices.items()}


# 🚀 STEP 4: Create FastAPI app
app = FastAPI(title="Smart Waste Classification API")

# 🧠 Helper function: classify image
def classify_image(img: Image.Image):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    
    confidence = float(np.max(predictions[0]))
    class_index = np.argmax(predictions[0])
    predicted_class = labels[class_index]

    warning = None
    if confidence < 0.6:
        return "unknown", confidence, "Low confidence image. Try clearer lighting."

    return predicted_class, confidence, warning


# 📡 STEP 5: API endpoint
@app.post("/predict")
async def predict(file:UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    waste_type, confidence, warning = classify_image(img)
    recommendation = recommend(waste_type)


    return {
    "waste_type": waste_type,
    "confidence": round(confidence * 100, 2),
    "recommendation": recommendation,
    "warning": warning
}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
#open at browser after running uvicorn main:app --reload , http://127.0.0.1:8000/docs