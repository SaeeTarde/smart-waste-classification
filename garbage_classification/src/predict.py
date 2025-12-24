import os
import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# 🔧 STEP 1: Tell Python where project root is
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(PROJECT_ROOT)

# 🔧 STEP 2: Now this import will work
from smart_waste_system.recommendation_engine import recommend

# Labels must match training order
labels = ["plastic", "paper", "metal"]

# 🔧 STEP 3: Load model (NO change to train.py)
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "garbage_model.keras")
model = tf.keras.models.load_model(MODEL_PATH)

def classify(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    return labels[class_index]

# 🔹 MAIN EXECUTION
if __name__ == "__main__":
    TEST_IMG = os.path.join(PROJECT_ROOT, "test_images", "test_paper1.avif")

   waste_type, confidence, warning = classify_image(img)
recommendation = recommend(waste_type)

    print("Waste Type:", waste_type, confidence, warning)
    print("Recommendation:", recommendation)
