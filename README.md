# 🚀 BytePulse AI Engine (SIH 26094)
> **AI-Powered Dynamic Mental Health Monitoring and Distress Prediction System for Victims of Atrocities**

---

## 🛠️ Project Architecture
This repository contains the backend and machine learning pipeline for **BytePulse**, combining:
1. **Behavioral Anomaly Detection:** An unsupervised **Isolation Forest** model tracking patterns like screen time, missed check-ins, and late-night activity.
2. **Text Emotion Analysis (NLP):** A pre-trained Hugging Face Transformer (`j-hartmann/emotion-english-distilroberta-base`) extracting precise distress weights (fear, sadness, anger).
3. **Dynamic Risk Scoring Engine:** Fuses behavioral and NLP features into a real-time risk score ($0-100$) and categorizes risk levels via a **FastAPI** backend.

---

## ⚙️ Quick Start for Teammates

### 1. Clone the Repository
\`\`\`bash
git clone https://github.com/Sourav111-cyber/bytepulse-ai-engine.git
cd bytepulse-ai-engine
\`\`\`

### 2. Install Dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 3. Run the FastAPI Server
\`\`\`bash
python -m uvicorn main:app --reload
\`\`\`

### 4. Test the API
Open your browser and navigate to:
* **Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
\`\`\`

---

