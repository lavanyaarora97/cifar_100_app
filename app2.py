from flask import Flask, request, render_template, jsonify
from torchvision import models, transforms
import torch
import torch.nn as nn
from PIL import Image
import io
from ultralytics import YOLO  # Ensure this is correctly installed

app = Flask(__name__)

# Define transformations for the classifier
#classifier_transform = transforms.Compose([
#    transforms.Resize((224, 224)),
#   transforms.ToTensor(),
#   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#])

inference_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

@app.route('/')
def home():
    return render_template("Home.html")

@app.route('/about')
def about():
    return render_template("About.html")

@app.route('/contact')
def contact():
    return render_template("Contact.html")

@app.route('/classify')
def classifier():
    return render_template("Classifier.html")

@app.route('/detect')
def detector():
    return render_template("Detector.html")

# Load the classifier model
#def load_classifier_model():
#    model = models.resnet50(pretrained=False)
#    num_ftrs = model.fc.in_features
#    model.fc = nn.Sequential(
#        nn.Linear(num_ftrs, 100)  # Assuming 100 classes
#    )
#    model.load_state_dict(torch.load('models/classifier.pth', map_location=torch.device('cpu')))
#    model.eval()
#    return model

#classifier_model = load_classifier_model()


def load_classifier_model():
    model = models.resnet50(pretrained=True)  # Start with a ResNet-50 model
    num_ftrs = model.fc.in_features
    # Update this definition to match the saved model's architecture
    model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 1024),
    nn.Dropout(0.2),
    nn.Linear(1024, 512),
    nn.Dropout(0.2),
    nn.Linear(512, 256),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.Dropout(0.2),
    nn.Linear(128, 100)
    )
    model.load_state_dict(torch.load('models/classifier.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

classifier_model = load_classifier_model()

# Load the detector model
# def load_detector_model():
#    model = YOLO('models/detector.pt', map_location='cpu')  # Adjust based on your actual saved model
#    model.eval()
#    return model

#detector_model = load_detector_model()

def load_detector_model():
    # Load the pre-trained YOLOv8s model from Ultralytics
    model = torch.hub.load('ultralytics/yolov8', 'yolov8s', pretrained=True)
    model.to('cpu')  # Move the model to CPU
    model.eval()  # Set the model to evaluation mode
    return model

detector_model = load_detector_model()

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        files = request.files.getlist('file')
        results = []
        for file in files:
            if file:
                image = Image.open(file.stream)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image_tensor = inference_transform(image).unsqueeze(0)  # Apply the transformation
                with torch.no_grad():
                    outputs = classifier_model(image_tensor)
                    _, predicted = torch.max(outputs, 1)
                    results.append(predicted.item())
        return render_template('Classifier.html', classification_results=results)
    return render_template('Classifier.html')

# @app.route('/detect', methods=['GET', 'POST'])
# def detect():
#    if request.method == 'POST':
#        files = request.files.getlist('file')
#        results = []
#        for file in files:
#            if file:
#                image = Image.open(file.stream)
#                if image.mode != 'RGB':
#                    image = image.convert('RGB')
#                results.append(detector_model(image).pandas().xyxy[0].to_dict(orient='records'))  # Using pandas for easier manipulation
#        return render_template('Detector.html', detection_results=results)
#    return render_template('Detector.html')


@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    image = Image.open(file.stream)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Prepare image for detection
    results = detector_model(image, size=640)  # Size can be adjusted based on model requirements
    detections = results.pandas().xyxy[0]  # Extracting results to DataFrame
    
    return jsonify(detections.to_dict(orient='records'))


if __name__ == '__main__':
    import os
    port = int(os.getenv("PORT", 10000))
    app.run(host='0.0.0.0', port=port)