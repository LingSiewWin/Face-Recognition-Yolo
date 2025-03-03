from flask import Flask, request
import cv2
import numpy as np
from models.face_detector import detect_faces
from models.face_recognition import get_face_embedding
from models.database import verify_face
from models.attendance import log_attendance
import sqlite3

app = Flask(__name__)
conn = sqlite3.connect("attendance.db", check_same_thread=False)
cursor = conn.cursor()

@app.route("/attendance", methods=["POST"])
def attendance():
    frame = request.files["image"].read()
    frame = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
    
    faces = detect_faces(frame)
    for face in faces:
        embedding = get_face_embedding(face)
        user_id = verify_face(embedding)
        if user_id:
            return log_attendance(cursor, conn, user_id)
    
    return {"message": "Face not recognized"}, 401

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
