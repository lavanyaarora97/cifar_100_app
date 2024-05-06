from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')
results = model("img.jpeg", show=True)
cv2.waitKey(1)