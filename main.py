# main.py (UPGRADED)
from fastapi import FastAPI, File, UploadFile, Form
import analysis_logic
import json
import os

app = FastAPI()

@app.post("/analyze")
async def analyze_stress(
    dass_data: str = Form(...),
    physio_data: str = Form(...),
    image_file: UploadFile = File(...),
    audio_file: UploadFile = File(...)
):
    # --- 1. Save uploaded files ---
    image_path = f"temp_{image_file.filename}"
    with open(image_path, "wb") as f:
        f.write(await image_file.read())

    audio_path = f"temp_{audio_file.filename}"
    with open(audio_path, "wb") as f:
        f.write(await audio_file.read())

    # --- 2. Transcribe Audio ---
    transcribed_text = analysis_logic.transcribe_audio_with_deepgram(audio_path)
    
    # --- 3. Get all four individual predictions ---
    facial_label, facial_conf = analysis_logic.predict_facial(image_path)
    audio_label, audio_conf = analysis_logic.predict_audio(audio_path)
    survey_label, survey_conf = analysis_logic.predict_dass21(json.loads(dass_data)['responses'])
    physio_label, physio_conf = analysis_logic.predict_physio_from_line(physio_data)

    # --- 4. Fuse the results ---
    confidences = [
        analysis_logic.get_stress_confidence(facial_label, facial_conf),
        analysis_logic.get_stress_confidence(audio_label, audio_conf),
        analysis_logic.get_stress_confidence(survey_label, survey_conf),
        analysis_logic.get_stress_confidence(physio_label, physio_conf)
    ]
    fused_score = analysis_logic.agreement_fusion(confidences)
    stress_score_percent = round(fused_score * 100)
    overall_label = "Stressed" if fused_score >= 0.5 else "Not Stressed"

    # --- 5. Get LLM suggestion ---
    llm_suggestion = analysis_logic.get_llm_suggestion_with_groq(stress_score_percent, transcribed_text)
    
    # --- 6. Clean up temp files ---
    os.remove(image_path)
    os.remove(audio_path)

    # --- 7. Return the final result ---
    return {
        "stress_level": overall_label,
        "stress_score_percent": stress_score_percent,
        "transcribed_text": transcribed_text,
        "llm_suggestion": llm_suggestion
    }

@app.get("/")
def read_root():
    return {"status": "Safe Space Full multimodal server is running."}
