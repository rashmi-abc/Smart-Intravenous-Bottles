import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from flask import Flask, request, send_file, jsonify

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define model
class IVInfusionModel(nn.Module):
    def __init__(self):
        super(IVInfusionModel, self).__init__()
        self.model = models.resnet18(weights="DEFAULT")
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

    def forward(self, x):
        return self.model(x)

# Load trained model
model = IVInfusionModel()
model.load_state_dict(torch.load("iv_infusion_model.pth", map_location=torch.device("cpu")), strict=False)
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to predict image class
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return "Normal Flow" if predicted.item() == 0 else "Reverse Flow"

# Route to serve the HTML file from root folder
@app.route('/')
def index():
    return send_file("ee.html")

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        prediction = predict_image(filepath)
        return jsonify({"image_url": filepath, "prediction": prediction})

if __name__ == '__main__':
    app.run(debug=True)