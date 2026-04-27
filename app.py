from fastapi import FastAPI, File, UploadFile
import shutil
import os

from crop_predict import predict_crop
from disease_predict import predict_disease

app = FastAPI()

# Advice dictionary
advice_dict = {
    "late blight": "Avoid leaf wetness and use fungicide",
    "early blight": "Apply fungicide weekly",
    "black rot": "Remove infected leaves",
    "healthy": "Crop is healthy"
}

@app.get("/")
def home():
    return {"message": "Crop Disease Prediction API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_path = "temp.jpg"

    # Save uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Predictions
    crop, crop_conf = predict_crop(file_path)
    pred_crop, disease = predict_disease(file_path)

    # Logic
    if crop.lower() != pred_crop.lower():
        disease = "Uncertain Disease"
        advice = "Mismatch between crop and disease"
    else:
        advice = advice_dict.get(disease.lower(), "Monitor crop")

    return {
        "crop": crop,
        "confidence": float(crop_conf),
        "disease": disease,
        "advice": advice
    }