import io
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
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
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(img, f"{cls} {conf}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Encode annotated image to base64
    _, img_encoded = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(img_encoded).decode()

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
