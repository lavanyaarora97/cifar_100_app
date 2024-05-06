from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from ultralytics import YOLO
from fastapi.staticfiles import StaticFiles
import cv2
import math
import uvicorn
import base64

app = FastAPI()

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load YOLO model
model = YOLO('best1.pt')
names = model.names

def encode_image(image):
    _, img_encoded = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(img_encoded).decode()
    return img_base64

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(content=open("static/index.html", "r").read())

@app.get("/main", response_class=HTMLResponse)
async def home():
    return HTMLResponse(content=open("static/main.html", "r").read())

@app.post("/detect")
async def detect_objects(image: UploadFile = File(...)):
    # Save the uploaded image temporarily
    with open("temp.jpg", "wb") as temp:
        temp.write(await image.read())

    # Use YOLO model to detect objects in the image
    results = model("temp.jpg")

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
                cv2.putText(img, f"{cls} ", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    # Encode annotated image to base64
    img_base64 = encode_image(img)

    # Clean up temporary image file
    import os
    os.remove("temp.jpg")

    # Prepare JSON response
    response_data = {
        "detections": detections,
        "annotated_image": img_base64
    }

    return JSONResponse(content=response_data)

if __name__ == "__main__":
    # Run the FastAPI application using uvicorn
    uvicorn.run(app, host="localhost", port=8000)
