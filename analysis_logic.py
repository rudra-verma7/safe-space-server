# analysis_logic.py (UPGRADED)
import os
import requests
import json
import numpy as np
import joblib
import tensorflow as tf
import cv2
import librosa
from pathlib import Path

# --- Model & Scaler Loading ---
BASE_DIR = Path(__file__).parent.resolve()
MODELS_DIR = BASE_DIR / "saved_models"

try:
    facial_model = tf.keras.models.load_model(MODELS_DIR / "facial_model.h5")
    audio_model = tf.keras.models.load_model(MODELS_DIR / "audio_model.h5")
    dass21_model = tf.keras.models.load_model(MODELS_DIR / "dass211_model.h5")
    physio_model = tf.keras.models.load_model(MODELS_DIR / "physio_model.h5")
    dass21_scaler = joblib.load(MODELS_DIR / "dass211_scaler.pkl")
    physio_scaler = joblib.load(MODELS_DIR / 'physio_scaler.pkl')
    print("All models and scalers loaded successfully.")
except Exception as e:
    print(f"Error loading models/scalers: {e}")
    # Handle missing models gracefully
    facial_model = audio_model = dass21_model = physio_model = None
    dass21_scaler = physio_scaler = None

# --- API-based Functions (Unchanged) ---
def transcribe_audio_with_deepgram(audio_file_path):
    # ... (code from before, no changes needed)
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key: return "[ERROR: Deepgram API key not set]"
    with open(audio_file_path, 'rb') as audio:
        headers = {'Authorization': f'Token {api_key}', 'Content-Type': 'audio/wav'}
        url = "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true"
        response = requests.post(url, headers=headers, data=audio)
    if response.status_code == 200:
        return response.json()['results']['channels'][0]['alternatives'][0]['transcript']
    return "[ERROR: Transcription failed]"

def get_llm_suggestion_with_groq(stress_score_percent, user_paragraph):
    # ... (code from before, no changes needed)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key: return "AI Coach is unavailable: API key not configured."
    prompt = f"""[INSTRUCTION] You are a compassionate AI mental health coach. The user has a stress score of {stress_score_percent}% and wrote this about their feelings: "{user_paragraph}". Respond in the required format. [/INSTRUCTION]"""
    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": "llama3-8b-8192", "messages": [{"role": "user", "content": prompt}], "temperature": 0.7})
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Could not reach the AI coach. Error: {e}"

# --- Prediction Functions (Re-added Facial/Audio) ---
def predict_facial(photo_path):
    if not facial_model: return "Not Stressed", 0.5
    try:
        img = cv2.imread(str(photo_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))
        input_img = (resized / 255.0).reshape(1, 48, 48, 1)
        prediction = facial_model.predict(input_img, verbose=0)
        class_id = np.argmax(prediction)
        label = "Stressed" if class_id == 1 else "Not Stressed"
        return label, float(prediction[0][class_id])
    except: return "Error", 0.5

def predict_audio(audio_file):
    if not audio_model: return "Not Stressed", 0.5
    try:
        y, sr = librosa.load(audio_file, sr=22050)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=38)
        target_length = 98
        if mfccs.shape[1] < target_length:
            mfccs = np.pad(mfccs, ((0, 0), (0, target_length - mfccs.shape[1])))
        else:
            mfccs = mfccs[:, :target_length]
        mfccs = np.expand_dims(mfccs, axis=(0, -1))
        prediction = audio_model.predict(mfccs, verbose=0)[0][0]
        label = "Stressed" if prediction >= 0.5 else "Not Stressed"
        confidence = prediction if prediction >= 0.5 else 1 - prediction
        return label, confidence
    except: return "Error", 0.5

def predict_dass21(q_responses):
    # ... (code from before, no changes needed)
    if not all([dass21_model, dass21_scaler]): return "Not Stressed", 0.5
    try:
        X = np.array([float(r) for r in q_responses]).reshape(1, -1)
        X_scaled = dass21_scaler.transform(X)
        pred_prob = dass21_model.predict(X_scaled, verbose=0)[0][0]
        label = "Stressed" if pred_prob >= 0.5 else "Not Stressed"
        return label, float(pred_prob)
    except: return "Error", 0.5

def predict_physio_from_line(line):
    # ... (code from before, no changes needed)
    if not all([physio_model, physio_scaler]): return "Not Stressed", 0.5
    try:
        data = json.loads(line)
        input_data = np.array([[float(data.get(k, 0)) for k in ["eda_uS", "hrv_rmssd", "temp_c", "acc_x_g", "acc_y_g", "acc_z_g"]]])
        input_scaled = physio_scaler.transform(input_data)
        prediction = physio_model.predict(input_scaled, verbose=0)[0][0]
        label = "Stressed" if prediction >= 0.5 else "Not Stressed"
        confidence = prediction if prediction >= 0.5 else 1 - prediction
        return label, float(confidence)
    except: return "Error", 0.5

# --- Fusion Logic (Re-added Agreement Fusion) ---
def get_stress_confidence(label, confidence):
    if "error" in str(label).lower(): return 0.5
    return float(confidence) if str(label).lower() == 'stressed' else 1.0 - float(confidence)

def agreement_fusion(confidences):
    valid_confidences = [c for c in confidences if c != 0.5]
    if not valid_confidences: return 0.5
    if len(valid_confidences) == 1: return valid_confidences[0]
    M = len(valid_confidences)
    agree_scores = [sum(1 - abs(valid_confidences[i] - valid_confidences[j]) for j in range(M) if i != j) / (M - 1) for i in range(M)]
    sum_agree = sum(agree_scores)
    if sum_agree < 1e-9: return np.mean(valid_confidences)
    return float(np.sum(np.array(agree_scores) * np.array(valid_confidences)) / sum_agree)
