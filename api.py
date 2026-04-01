from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
from datetime import datetime
from face_module.predict import predict_emotion
import os
from fastapi.middleware.cors import CORSMiddleware
import csv

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@app.get("/")
def home():
    return {"message": "Mental Health API Running"}


@app.get("/analysis")
def get_analysis():
    try:
        import pandas as pd

        csv_path = os.path.join(BASE_DIR, "shared_outputs", "emotion_output.csv")

        print("\n--- STEP 1: Reading CSV ---")
        print("CSV PATH:", csv_path)

        if not os.path.exists(csv_path):
            print("❌ CSV NOT FOUND")
            return {"error": "CSV file not found"}

        df = pd.read_csv(csv_path)
        
        from datetime import datetime

        today = datetime.now().strftime("%Y-%m-%d")
        df = df[df["date"] == today]

        print("STEP 2: RAW CSV SHAPE:", df.shape)
        print(df.head())

        # 🔥 TAKE LAST 100
        df = df.tail(100)

        print("STEP 3: AFTER TAIL:", df.shape)

        
        temp_path = os.path.join(BASE_DIR, "shared_outputs", "temp.csv")
        df.to_csv(temp_path, index=False)

        from behavior_module.behavior_analysis import analyze_emotions
        result = analyze_emotions(temp_path)

        print("STEP 4: RESULT FROM ANALYSIS:", result)

        return result

    except Exception as e:
        print("❌ ERROR:", str(e))
        return {"error": str(e)}

# ✅ REAL-TIME PREDICT ROUTE
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        emotion, confidence = predict_emotion(image)
        if emotion == "no_face":
            return {
                "emotion": emotion,
                "confidence": confidence,
            }

        # 📁 CSV path
        csv_path = os.path.join(BASE_DIR, "shared_outputs", "emotion_output.csv")

        file_exists = os.path.exists(csv_path)

        with open(csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow(["date", "timestamp", "emotion", "confidence"])
                
            session_id = datetime.now().strftime("%Y%m%d_%H%M")
            now = datetime.now()

            writer.writerow([
                now.strftime("%Y-%m-%d"),   # DATE
                now.strftime("%H:%M:%S"),   # TIME
                emotion,
                confidence
            ])

        return {
            "emotion": emotion,
            "confidence": confidence
        }

    except Exception as e:
        return {"error": str(e)}