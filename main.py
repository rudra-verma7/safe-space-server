# main.py

from fastapi import FastAPI, File, UploadFile, Form
import analysis_logic
import json
import os

app = FastAPI()

@app.post("/analyze")
async def analyze_stress(
    dass_data: str = Form(...),
    physio_data: str = Form(...),
    audio_file: UploadFile = File(...)
):
    # We no longer need the user's photo since we removed the heavy facial model
    audio_path = f"temp_{audio_file.filename}"
    with open(audio_path, "wb") as f:
        f.write(await audio_file.read())

    # --- Analysis ---
    transcribed_text = analysis_logic.transcribe_audio_with_deepgram(audio_path)
    
    # Get individual predictions
    survey_label, survey_conf = analysis_logic.predict_dass21(json.loads(dass_data)['responses'])
    physio_label, physio_conf = analysis_logic.predict_physio_from_line(physio_data)

    # Fuse them
    confidences = [
        analysis_logic.get_stress_confidence(survey_label, survey_conf),
        analysis_logic.get_stress_confidence(physio_label, physio_conf)
    ]
    fused_score = analysis_logic.simple_fusion(confidences)
    stress_score_percent = round(fused_score * 100)
    overall_label = "Stressed" if fused_score >= 0.5 else "Not Stressed"

    # Get LLM suggestion
    llm_suggestion = analysis_logic.get_llm_suggestion_with_groq(stress_score_percent, transcribed_text)
    
    # Clean up temp file
    os.remove(audio_path)

    # Return the final result
    return {
        "stress_level": overall_label,
        "stress_score_percent": stress_score_percent,
        "transcribed_text": transcribed_text,
        "llm_suggestion": llm_suggestion
    }

@app.get("/")
def read_root():
    return {"status": "Safe Space server is running."}
