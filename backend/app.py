from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)
CORS(app)

# 🔹 MobileNetV2 with 2 outputs (Real vs Fake)
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

# 🔹 Load weights
state_dict = torch.load("model_checkpoint.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# 🔹 Preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# 🔹 Warm up
dummy = torch.zeros(1, 3, 128, 128)
with torch.no_grad():
    model(dummy)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(file).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)[0]
        confidence, predicted_class = torch.max(probs, dim=0)

    prediction = "Real" if predicted_class.item() == 0 else "Fake"

    return jsonify({"prediction": prediction, "confidence": confidence.item()})

if __name__ == "__main__":
    app.run(debug=True)
