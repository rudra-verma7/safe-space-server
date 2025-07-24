# analysis_logic.py

import os
import requests
import json
import numpy as np
import joblib # Using joblib for the scaler
import tensorflow as tf

# --- Model & Scaler Loading ---
# We still need the lightweight models and scalers.
# Ensure these files are in your deployment folder.
try:
    # IMPORTANT: You must upload these specific files with your code.
    dass21_model = tf.keras.models.load_model("dass211_model.h5")
    physio_model = tf.keras.models.load_model("physio_model.h5")
    dass21_scaler = joblib.load("dass211_scaler.pkl")
    physio_scaler = joblib.load("physio_scaler.pkl")
except Exception as e:
    print(f"Error loading a local model or scaler: {e}")
    # In a real app, you'd handle this more gracefully.
    dass21_model = physio_model = dass21_scaler = physio_scaler = None

# --- API-based Functions ---

def transcribe_audio_with_deepgram(audio_file_path):
    """Transcribes audio using Deepgram's API."""
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        return "[ERROR: Deepgram API key not set]"

    with open(audio_file_path, 'rb') as audio:
        headers = {'Authorization': f'Token {api_key}', 'Content-Type': 'audio/wav'}
        url = "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true"
        response = requests.post(url, headers=headers, data=audio)
        if response.status_code == 200:
            return response.json()['results']['channels'][0]['alternatives'][0]['transcript']
    return "[ERROR: Transcription failed]"


def get_llm_suggestion_with_groq(stress_score_percent, user_paragraph):
    """Gets coaching suggestions from Groq's LLaMA3 model."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "AI Coach is unavailable: API key not configured."

    prompt = f"""
[INSTRUCTION]
You are a compassionate AI mental health coach. The user has a stress score of {stress_score_percent}% and wrote this about their feelings: "{user_paragraph}".
Your job is to analyze the score and the text.

Respond in the following format ONLY:
1. **Likely causes of stress:** Based on the user's paragraph.
2. **Emotional tone:** 1-sentence mood interpretation.
3. **Personalized suggestions (3 tips):** These must fit the user's exact situation.
4. **Motivational closing line:** Encourage the user with warmth and empathy.

Keep the total response under 250 words and maintain an empathetic, non-judgmental tone.
[/INSTRUCTION]
"""
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "llama3-8b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
            },
            timeout=15
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        return f"Could not reach the AI coach. Error: {e}"


# --- Original Prediction Functions (Facial/Audio are removed) ---

def predict_dass21(q_responses):
    if not all([dass21_model, dass21_scaler]):
        return "Not Stressed", 0.5
    try:
        X = np.array([float(r) for r in q_responses]).reshape(1, -1)
        X_scaled = dass21_scaler.transform(X)
        pred_prob = dass21_model.predict(X_scaled, verbose=0)[0][0]
        label = "Stressed" if pred_prob >= 0.5 else "Not Stressed"
        return label, float(pred_prob)
    except Exception:
        return "Error", 0.5


def predict_physio_from_line(line):
    if not all([physio_model, physio_scaler]):
        return "Not Stressed", 0.5
    try:
        # IMPORTANT: This must match the JSON from your ESP32
        data = json.loads(line)
        input_data = np.array([[float(data.get(k, 0)) for k in ["eda_uS", "hrv_rmssd", "temp_c", "acc_x_g", "acc_y_g", "acc_z_g"]]])
        input_scaled = physio_scaler.transform(input_data)
        prediction = physio_model.predict(input_scaled, verbose=0)[0][0]
        label = "Stressed" if prediction >= 0.5 else "Not Stressed"
        confidence = prediction if prediction >= 0.5 else 1 - prediction
        return label, float(confidence)
    except Exception:
        return "Error", 0.5

# --- Fusion Logic (Simplified) ---

def get_stress_confidence(label, confidence):
    if "error" in str(label).lower(): return 0.5
    return float(confidence) if str(label).lower() == 'stressed' else 1.0 - float(confidence)

def simple_fusion(confidences):
    """A simpler fusion for a 2-modality system."""
    valid_confidences = [c for c in confidences if c != 0.5]
    if not valid_confidences: return 0.5
    return float(np.mean(valid_confidences))
