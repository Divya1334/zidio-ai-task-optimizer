# text_emotion.py
from transformers import pipeline

# Load pre-trained emotion classification pipeline
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

def predict_text_emotion(text):
    result = emotion_classifier(text)
    return result[0][0]['label']  # Return top emotion