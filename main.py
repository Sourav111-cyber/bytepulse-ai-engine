from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from transformers import pipeline

app = FastAPI(title="BytePulse Multimodal Engine API")

# Enable CORS so the frontend team can connect without browser blocks
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all frontend origins during development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading Behavioral Anomaly Model...")
behavior_model = joblib.load('./models/behavioral_anomaly_model.pkl')

print("Loading NLP Model...")
nlp_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

class VictimData(BaseModel):
    text_input: str
    screen_time_hours: float
    missed_checkins: int
    late_night_hours: float

@app.post("/analyze-distress")
def analyze_distress(data: VictimData):
    total_risk_score = 0
    
    # --- BEHAVIOR ANALYSIS ---
    user_metrics = np.array([[data.screen_time_hours, data.missed_checkins, data.late_night_hours]])
    behavior_prediction = behavior_model.predict(user_metrics)[0]
    behavior_status = "Anomalous" if behavior_prediction == -1 else "Normal"
    if behavior_status == "Anomalous":
        total_risk_score += 30
        
    # --- TEXT ANALYSIS ---
    raw_results = nlp_model(data.text_input)
    if isinstance(raw_results[0], list):
        emotion_list = raw_results[0]
    else:
        emotion_list = raw_results
        
    distress_scores = {}
    for emotion in emotion_list:
        if emotion['label'] in ['fear', 'sadness', 'anger']:
            distress_scores[emotion['label']] = round(emotion['score'], 4)
            
    nlp_risk = sum(distress_scores.values()) * 40
    total_risk_score += nlp_risk
    
    # --- FINAL SCORING ---
    if total_risk_score >= 60:
        category = "High Risk"
    elif total_risk_score >= 35:
        category = "Moderate Risk"
    else:
        category = "Low Risk"
        
    return {
        "risk_category": category,
        "dynamic_score": round(total_risk_score, 2),
        "breakdown": {
            "behavior": behavior_status,
            "text_emotions": distress_scores
        }
    }
