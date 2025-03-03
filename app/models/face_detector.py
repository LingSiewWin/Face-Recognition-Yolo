import cv2
import torch
from ultralytics import YOLO

model = YOLO("yolov11-face.pt")  # Load YOLOv11 model

def detect_faces(frame):
    results = model(frame)  # Run YOLOv11
    faces = []
    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = result
        if conf > 0.5:  # Confidence threshold
            face = frame[int(y1):int(y2), int(x1):int(x2)]
            faces.append(face)
    return faces
