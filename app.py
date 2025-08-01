import os
import requests
from flask import Flask, render_template, request
from PIL import Image
import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from torchvision import transforms
import wikipedia

# Setup
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load pretrained ANN model (ViT from HuggingFace)
model_name = "google/vit-base-patch16-224"
extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)
model.eval()

def classify_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    label = model.config.id2label[predicted_class_idx]
    return label

def get_animal_info(animal_name):
    try:
        summary = wikipedia.summary(animal_name, sentences=3)
        return summary
    except Exception:
        return "No detailed info found on Wikipedia."

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        img = request.files['image']
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
        img.save(img_path)

        # Run classification
        predicted_label = classify_image(img_path)
        info = get_animal_info(predicted_label)

        return render_template('result.html',
                               image_path=img_path,
                               name=predicted_label,
                               description=info)

    return render_template('index.html')


