from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image

app = Flask(__name__)
CORS(app)

# Load + quantize model once
model = torch.load("model_checkpoint.pth", map_location="cpu")
model.eval()
model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

# Faster preprocessing
transform = transforms.Compose([
    transforms.Resize((160, 160)),   # smaller size
    transforms.ToTensor()            # skip normalization if not essential
])

#prediction
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(file).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.sigmoid(output).item()
        prediction = "Real" if prob > 0.5 else "Fake"

    return jsonify({"prediction": prediction, "confidence": prob})

