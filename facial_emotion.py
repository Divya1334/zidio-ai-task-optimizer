import cv2
import random

emotions = ['happy', 'sad', 'angry', 'surprise', 'neutral']

def predict_facial_emotion():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return "Failed to capture image"

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return "No face detected"

    # Just return a random emotion for demo
    predicted_emotion = random.choice(emotions)
    return predicted_emotion