from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import math
import uvicorn

app = FastAPI()

# Load YOLO model
model = YOLO('best.pt')
names = model.names

@app.post("/detect")
async def detect_objects(image: UploadFile = File(...)):
    # Save the uploaded image temporarily
    with open("temp.jpg", "wb") as temp:
        temp.write(await image.read())

    # Use YOLO model to detect objects in the image
    results = model("temp.jpg")

    # Process detection results and prepare response
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = math.ceil((box.conf[0] * 100)) / 100
            for c in box.cls:
                cls = names[int(c)]
                detections.append({"class": cls, "confidence": conf})

    # Clean up temporary image file
    import os
    os.remove("temp.jpg")

    return {"detections": detections}

if __name__ == "__main__":
    # Run the FastAPI application using uvicorn
    uvicorn.run(app, host="localhost", port=8000)
