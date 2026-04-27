import numpy as np
import json
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# 🔥 Fix Keras compatibility issue
tf.keras.utils.get_custom_objects().clear()

# 🔥 Get base directory (important for Render)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 🔥 Load model safely
model_path = os.path.join(BASE_DIR, "crop_classifier_model.h5")
model = load_model(model_path, compile=False)

# 🔥 Load class mapping safely
class_path = os.path.join(BASE_DIR, "crop_class_indices.json")
with open(class_path) as f:
    class_indices = json.load(f)

# Convert mapping to list
crop_classes = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]


def predict_crop(img_path):
    # Load image
    img = image.load_img(img_path, target_size=(128, 128))
    
    # Preprocess
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array)
    print("Crop raw prediction:", pred)

    index = np.argmax(pred)
    confidence = np.max(pred)

    return crop_classes[index], round(float(confidence * 100), 2)