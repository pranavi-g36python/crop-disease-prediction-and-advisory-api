from crop_predict import predict_crop
from disease_predict import predict_disease
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# -------------------------------
# FILE PICKER
# -------------------------------
root = Tk()
root.withdraw()

img_path = askopenfilename(
    title="Select Leaf Image",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
)

root.destroy()

if img_path == "":
    print("No image selected")
    exit()

print("Using image:", img_path)

# -------------------------------
# PREDICTIONS
# -------------------------------
crop, crop_conf = predict_crop(img_path)
pred_crop, disease = predict_disease(img_path)

# -------------------------------
# ADVICE DICTIONARY (MOVE HERE)
# -------------------------------
advice_dict = {
    "late blight": "Avoid leaf wetness and use fungicide",
    "early blight": "Apply fungicide weekly",
    "black rot": "Remove infected leaves",
    "healthy": "Crop is healthy"
}

# -------------------------------
# MATCH LOGIC
# -------------------------------
if crop.lower() != pred_crop.lower():
    disease = "Uncertain Disease"
    advice = "Mismatch between crop and disease"
else:
    advice = advice_dict.get(disease.lower(), "Monitor crop")

# -------------------------------
# OUTPUT
# -------------------------------
print("\n🌱 FINAL RESULT")
print("----------------------------")
print(f"Crop      : {crop} ({crop_conf}%)")
print(f"Disease   : {disease}")
print(f"Advice    : {advice}")
