from flask import Flask, request, render_template, jsonify

import os
import math
from PIL import Image
import io
from ultralytics import YOLO  # Ensure this is correctly installed

import base64
import cv2

import logging
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from flask_cors import CORS  # Import the CORS class

app = Flask(__name__)
#CORS(app, resources={r"*": {"origins": "*"}})
CORS.init_app(app)  # Allow requests from all origins

# Define transformations for the classifier
#classifier_transform = transforms.Compose([
#    transforms.Resize((224, 224)),
#   transforms.ToTensor(),
#   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#])

# inference_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

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

MODEL = tf.keras.models.load_model("./finetunedCifar100.h5")
CLASS_NAMES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'cra', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower',
    'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle',
    'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

model = YOLO('yolov8n.pt')
names = model.names
@app.route('/detection', methods=['GET', 'POST'])
def detect_objects():
    if 'file' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    image = request.files['file']

    if image.filename == '':
        return jsonify({'error': 'No selected image'}), 400

    # Save the uploaded image temporarily
    image.save("temp.jpg")

    # Use YOLO model to detect objects in the image
    results = model("temp.jpg")  # Assuming `model` is defined somewhere

    # Process detection results and prepare response
    detections = []
    img = cv2.imread("temp.jpg")

    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = math.ceil((box.conf[0] * 100)) / 100
            for c in box.cls:
                cls = names[int(c)]
                detections.append({"class": cls, "confidence": conf})
                # Draw bounding box and label on the image
                xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                cv2.putText(img, f"{cls} ", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                print(cls)

    # Encode annotated image to base64
    _, img_base64 = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(img_base64).decode('utf-8')

    # Clean up temporary image file
    os.remove("temp.jpg")

    # Prepare JSON response
    response_data = {
        "detections": detections,
        "annotated_image": img_base64
    }

    return jsonify(response_data)

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)).convert("RGB"))
    resized_image = np.array(Image.fromarray(image).resize((32, 32)))
    logging.info("Resized Image Shape: %s", resized_image.shape)
    return resized_image

@app.route("/upload", methods=["POST"])
def predict():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file part'}), 400

    image = read_file_as_image(file.read())

    img_batch = tf.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch / 255)

    top_predictions_indices = np.argsort(predictions[0])[::-1][:4]

    result = []
    for i, class_index in enumerate(top_predictions_indices):
        class_name = CLASS_NAMES[class_index]
        confidence = predictions[0][class_index]

        css_value = 'green' if i == 0 else 'red'

        result.append({
            'class': class_name,
            'confidence': float(confidence) * 100,
            'css': css_value
        })

    return jsonify(result)



if __name__ == '__main__':
    import os
    port = int(os.getenv("PORT", 10000))
    app.run(host='0.0.0.0', port=port)