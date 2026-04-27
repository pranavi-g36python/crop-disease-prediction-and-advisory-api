import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# Load model
model = load_model("trained_disease_model.h5")

disease_classes = [
    "Tomato___healthy", "Tomato___Late_blight",
    "Potato___healthy", "Potato___Early_blight",
    "Grape___healthy", "Grape___Black_rot",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy"
]

advisory = {
    "Late blight": "Avoid leaf wetness",
    "Early blight": "Apply fungicide weekly",
    "Black rot": "Remove infected leaves",
    "Healthy": "Crop is healthy"
}

def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(128,128))
    
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)   # only once
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    print("Disease raw prediction:", pred)

    index = np.argmax(pred)

    label = disease_classes[index]

    predicted_crop = label.split("___")[0]
    disease = label.split("___")[1].replace("_", " ").title()

    return predicted_crop, disease