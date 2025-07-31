# speech_emotion.py
import librosa
import numpy as np

def extract_features(audio_file):
    y, sr = librosa.load(audio_file, duration=3, offset=0.5)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs

def predict_speech_emotion(audio_file):
    features = extract_features(audio_file)
    # Dummy model (replace with ML model if available)
    avg = np.mean(features)
    if avg < -100:
        return "sad"
    elif avg < 0:
        return "neutral"
    else:
        return "happy"