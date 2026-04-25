from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import time

app = Flask(__name__)
CORS(app)

# 🔹 Step 1: Recreate the model architecture
model = models.mobilenet_v2(weights=None)   # lightweight model
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

# 🔹 Step 2: Load state dict (weights only)
state_dict = torch.load("model_checkpoint.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# 🔹 Step 3: Dynamic quantization (safe for CPU)
model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# 🔹 Step 4: TorchScript conversion
scripted_model = torch.jit.script(model)
scripted_model.eval()

# 🔹 Step 5: Preprocessing (smaller input + normalization)
transform = transforms.Compose([
    transforms.Resize((96, 96)),   # smaller size speeds up inference
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 🔹 Warm up once
dummy = torch.zeros(1, 3, 96, 96)
with torch.no_grad():
    scripted_model(dummy)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(file).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    start = time.time()
    with torch.no_grad():
        output = scripted_model(image_tensor)
        probs = torch.softmax(output, dim=1)[0]
        confidence, predicted_class = torch.max(probs, dim=0)
    elapsed = time.time() - start

    prediction = "Real" if predicted_class.item() == 0 else "Fake"
    if confidence.item() < 0.6:
        prediction = "Uncertain"

    return jsonify({
        "prediction": prediction,
        "confidence": round(confidence.item(), 4),
        
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
