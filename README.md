
# Zidio AI Task Optimizer

An AI-powered project that detects employee emotions using multimodal inputs—text, facial expressions, and speech. The goal is to analyze real-time emotions and provide insights to boost productivity and well-being.

##Project Objective

To create a productivity-enhancing tool that:
- Analyzes written input (text) to detect emotional tone.
- Uses facial expression analysis via webcam to detect real-time emotions.
- Processes audio to identify emotional content in speech.
- Combines insights into a simple dashboard using Streamlit.

## 🗂 Project Structure

zidio-ai-task-optimizer/ ├── app.py                  # Streamlit dashboard ├── text_emotion.py         # Text emotion detection using Hugging Face transformers ├── facial_emotion.py       # Facial emotion recognition using DeepFace ├── speech_emotion.py       # Audio emotion recognition using librosa ├── requirements.txt        # Project dependencies

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Divya1334/zidio-ai-task-optimizer.git
cd zidio-ai-task-optimizer

2. Create and activate virtual environment

Windows:

python -m venv env
.\env\Scripts\activate

Mac/Linux:

python3 -m venv env
source env/bin/activate

3. Install dependencies

pip install -r requirements.txt

4. Run the Streamlit app

streamlit run app.py


---

🔍 Features

💬 Text Emotion Detection using transformer models

📷 Facial Emotion Recognition using DeepFace & OpenCV

🎤 Speech Emotion Analysis using audio features (MFCCs)

🧩 Easy-to-use Streamlit interface

📊 Multimodal emotion insight generation



---

🧰 Tech Stack

Python

Streamlit

Hugging Face Transformers

DeepFace

OpenCV

Librosa

TensorFlow / Keras



---

🙋‍♀ Author

K Divya

GitHub: Divya1334

LinkedIn: https://www.linkedin.com/in/divya-k-338a0a264?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app

Email: divyakothuru2895@gmail.com



---

📜 License

This project is intended for academic or educational use only.

📌 You can copy this and paste it as a file named README.md in your GitHub repo.  
Let me know if you'd like me to generate and upload it into your project folder for you.
