import cv2
from scipy.spatial import distance as dist

# Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

def is_real_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return False

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi)
        if len(eyes) >= 2:
            return True  # At least 2 eyes detected
    return False