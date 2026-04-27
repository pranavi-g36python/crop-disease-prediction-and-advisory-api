import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    
# Load model
model = load_model("crop_classifier_model.keras")

# Load class mapping
with open("crop_class_indices.json") as f:
    class_indices = json.load(f)

crop_classes = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]

def predict_crop(img_path):
    img = image.load_img(img_path, target_size=(128,128))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    print("Crop raw prediction:", pred)
    index = np.argmax(pred)
    confidence = np.max(pred)

    return crop_classes[index], round(confidence*100, 2)