from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
from face_module.predict import predict_emotion
import os
from fastapi.middleware.cors import CORSMiddleware

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
    csv_path = os.path.join(BASE_DIR, "shared_outputs", "emotion_output.csv")

    if not os.path.exists(csv_path):
        return {"error": "CSV file not found"}

    from behavior_module.behavior_analysis import analyze_emotions
    result = analyze_emotions(csv_path)
    return result


# ✅ REAL-TIME PREDICT ROUTE
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        emotion, confidence = predict_emotion(image)

        return {
            "emotion": emotion,
            "confidence": confidence
        }

    except Exception as e:
        return {"error": str(e)}