Smart Waste Classification & Recommendation System ♻️
📌 Overview

The Smart Waste Classification & Recommendation System is an AI-based application that classifies waste images into categories such as Plastic, Paper, and Metal, and provides eco-friendly disposal recommendations using a rule-based sustainability engine.

This project combines deep learning, transfer learning, and FastAPI to create an end-to-end intelligent system for environmental awareness.

🎯 Features

Image-based waste classification using CNN

Transfer Learning with MobileNetV2

Confidence-based prediction handling

Sustainability recommendation engine

REST API using FastAPI

Ready for frontend and deployment

🛠️ Tech Stack

Language: Python

Deep Learning: TensorFlow, Keras

Model: MobileNetV2

Backend: FastAPI, Uvicorn

Libraries: NumPy, Pillow, Scikit-learn, Matplotlib

📂 Project Structure
AI_1/
├── dataset/
│   ├── metal/
│   ├── paper/
│   └── plastic/
├── model/
│   ├── garbage_model.keras
│   └── class_indices.json
├── src/
│   ├── train.py
│   ├── predict.py
│   └── recommendation_engine.py
├── main.py
├── requirements.txt
└── README.md

🧠 Model Training

Images resized to 224 × 224

Dataset split:

80% Training

20% Validation

Data Augmentation:

Rotation

Zoom

Horizontal Flip

Class imbalance handled using class weights

Training command:

python src/train.py

📈 Accuracy Optimization

Transfer Learning

Data Augmentation

Class Weight Balancing

Dropout (0.4)

Early Stopping

Final Accuracy:

Training: ~93%

Validation: ~88–91%

🧩 Label Mapping Fix

Keras automatically assigns class labels based on folder names.
To avoid mismatch during prediction:

json.dump(train_data.class_indices, open("model/class_indices.json","w"))


This ensures correct class-to-output mapping.

🔁 Recommendation Engine

Each predicted waste type is mapped to:

Disposal method

Eco score (0–100)

Environmental tip

This bridges AI output with sustainability logic.

🚀 Backend API

Run the server:

uvicorn main:app --reload


API Docs:

http://127.0.0.1:8000/docs

/predict Endpoint Returns:

Waste Type

Confidence Score

Disposal Recommendation

Warning (if confidence is low)

🧪 Example Output
{
  "waste_type": "plastic",
  "confidence": 0.91,
  "disposal_method": "Recycle",
  "eco_score": 70,
  "tip": "Rinse before recycling"
}

✅ Final Status

Fully working AI pipeline

Accurate predictions

API-ready system

Scalable and reusable

🌍 Conclusion

This project demonstrates how AI can be applied to real-world environmental problems by combining deep learning models, backend services, and rule-based intelligence into a practical solution.
