import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# 🔥 Fix compatibility issue
tf.keras.utils.get_custom_objects().clear()

# 🔥 Base path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 🔥 Load model safely
model_path = os.path.join(BASE_DIR, "trained_disease_model.h5")
model = load_model(model_path, compile=False)

# Classes
disease_classes = [
    "Tomato___healthy", "Tomato___Late_blight",
    "Potato___healthy", "Potato___Early_blight",
    "Grape___healthy", "Grape___Black_rot",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy"
]

# Advice mapping
advisory = {
    "Late blight": "Avoid leaf wetness and use fungicide",
    "Early blight": "Apply fungicide weekly",
    "Black rot": "Remove infected leaves",
    "Healthy": "Crop is healthy"
}


def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    print("Disease raw prediction:", pred)

    index = np.argmax(pred)

    label = disease_classes[index]
    crop = label.split("___")[0]
    disease = label.split("___")[1].replace("_", " ").title()

    advice = advisory.get(disease, "Monitor crop")

    return crop, disease