from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image

app = Flask(__name__)
CORS(app)

# 🔹 Load model once at startup
model = torch.load("model_checkpoint.pth", map_location="cpu")
model.eval()

# 🔹 Define preprocessing (resize + tensor + normalize)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # adjust if your training used different normalization
])

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(file).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        output = model(image_tensor)
        # assume binary classification with sigmoid
        prob = torch.sigmoid(output).item()
        prediction = "Real" if prob > 0.5 else "Fake"

    return jsonify({"prediction": prediction, "confidence": prob})
