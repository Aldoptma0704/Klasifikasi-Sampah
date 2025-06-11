from flask import Flask, render_template, request
from PIL import Image
import torch
import torchvision.transforms as transforms
from vit_model import load_model
import os

app = Flask(__name__)

# Load model
model = load_model()
model.load_state_dict(torch.load("model/vit_waste_classification_final.pth", map_location='cpu'))
model.eval()

# Label
class_names = ["Organik", "Daur Ulang"]

# Transformasi gambar
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        file = request.files["image"]
        if file:
            image = Image.open(file.stream).convert("RGB")
            img_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(img_tensor)
                _, predicted = torch.max(outputs, 1)
                prediction = class_names[predicted.item()]
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)