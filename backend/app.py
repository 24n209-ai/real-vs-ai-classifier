from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
from flask_cors import CORS   # 🔹 Added for CORS

app = Flask(__name__)
CORS(app)   # 🔹 Allow requests from frontend

# 🔹 Step 1: Recreate MobileNetV2 architecture
model = models.mobilenet_v2(weights=None)   # use weights=None instead of pretrained
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 2)  # 2 classes: Fake vs Real

# 🔹 Step 2: Load your checkpoint
state_dict = torch.load("model_checkpoint.pth", map_location="cpu")
model.load_state_dict(state_dict)

# 🔹 Step 3: Set to evaluation mode
model.eval()

# 🔹 Step 4: Define preprocessing (with normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        # 🔹 Corrected label mapping
        label = "Fake" if predicted.item() == 0 else "Real"

    return jsonify({"prediction": label})

if __name__ == "__main__":
    app.run(debug=True)
