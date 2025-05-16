from deepface import DeepFace
import cv2

def get_emotion(frame):
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, model_name="Facenet512")
        emotion = result[0]['dominant_emotion']
        score = result[0]['emotion'][emotion]
        return emotion, score
    except Exception:
        return "No face", 0.0

def display_emotion(frame, emotion, score):
    text = f"{emotion} ({score:.1f}%)"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame