import gradio as gr
import shutil

from crop_predict import predict_crop
from disease_predict import predict_disease

# Advice dictionary
advice_dict = {
    "late blight": "Avoid leaf wetness and use fungicide",
    "early blight": "Apply fungicide weekly",
    "black rot": "Remove infected leaves",
    "healthy": "Crop is healthy"
}

def predict(image):
    file_path = "temp.jpg"

    # Save uploaded image
    shutil.copy(image, file_path)

    # Predictions
    crop, crop_conf = predict_crop(file_path)
    pred_crop, disease = predict_disease(file_path)

    # Logic
    if crop.lower() != pred_crop.lower():
        disease = "Uncertain Disease"
        advice = "Mismatch between crop and disease"
    else:
        advice = advice_dict.get(disease.lower(), "Monitor crop")

    return f"""
🌱 Crop: {crop}
📊 Confidence: {crop_conf}%
🦠 Disease: {disease}
💡 Advice: {advice}
"""

# Gradio UI
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="filepath"),
    outputs="text",
    title="🌿 Crop Disease Prediction System",
    description="Upload a leaf image to detect crop type and disease"
)

iface.launch(server_name="0.0.0.0", server_port=7860)