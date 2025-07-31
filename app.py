# app.py
import streamlit as st
from text_emotion import predict_text_emotion
from facial_emotion import predict_facial_emotion
from speech_emotion import predict_speech_emotion

st.title("Zidio AI-Powered Task Optimizer")

st.header("1. Text Emotion Detection")
text = st.text_input("Enter text:")
if st.button("Detect Text Emotion"):
    emotion = predict_text_emotion(text)
    st.success(f"Predicted Emotion: {emotion}")

st.header("2. Facial Emotion Detection")
if st.button("Detect Facial Emotion (via Webcam)"):
    emotion = predict_facial_emotion()
    st.success(f"Detected Facial Emotion: {emotion}")

st.header("3. Speech Emotion Detection")
audio_file = st.file_uploader("Upload Audio File (.wav)", type=['wav'])
if st.button("Detect Speech Emotion"):
    if audio_file is not None:
        with open("temp.wav", "wb") as f:
            f.write(audio_file.read())
        emotion = predict_speech_emotion("temp.wav")
        st.success(f"Detected Speech Emotion: {emotion}")
    else:
        st.warning("Please upload a .wav file.")