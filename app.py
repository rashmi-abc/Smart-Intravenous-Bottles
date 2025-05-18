import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io

app = Flask(__name__)

# ✅ Step 1: Define the Model Class
class IVInfusionModel(nn.Module):
    def __init__(self):
        super(IVInfusionModel, self).__init__()
        self.model = models.resnet18(weights="DEFAULT")
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # 2 output classes

    def forward(self, x):
        return self.model(x)

# ✅ Step 2: Load Model
model = IVInfusionModel()
model.load_state_dict(torch.load("iv_infusion_model.pth", map_location=torch.device("cpu")), strict=False)
model.eval()

# ✅ Step 3: Define Image Preprocessing
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# ✅ Step 4: Define Flask Routes
@app.route("/")
def index():
    return render_template("index.html")  # Ensure index.html exists in 'templates' folder

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        image = transform_image(image)
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            label = "Normal Flow" if predicted.item() == 0 else "Reverse Blood Flow"
        return jsonify({"prediction": label})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Step 5: Run Flask Server
if __name__ == "__main__":
    app.run(debug=True)
