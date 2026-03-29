import pandas as pd
import os

def analyze_emotions(csv_path: str):
    data = pd.read_csv(csv_path)

    total_seconds = len(data)

    # Emotion ratios
    neutral_ratio = (data["emotion"] == "neutral").sum() / total_seconds
    sad_ratio = (data["emotion"] == "sad").sum() / total_seconds
    happy_ratio = (data["emotion"] == "happy").sum() / total_seconds
    angry_ratio = (data["emotion"] == "angry").sum() / total_seconds
    fear_ratio = (data["emotion"] == "fear").sum() / total_seconds

    stress_ratio = angry_ratio + fear_ratio

    # Emotion variability
    emotion_changes = (data["emotion"] != data["emotion"].shift()).sum() - 1
    emotion_variability = emotion_changes / total_seconds

    # Confidence
    avg_confidence = data["confidence"].mean()

    # Sad trend
    sad_series = (data["emotion"] == "sad").astype(int)
    sad_trend = sad_series.diff().sum()

    # Score calculation
    score = 0

    if sad_ratio > 0.4:
        score += 2

    if happy_ratio < 0.2:
        score += 2

    if stress_ratio > 0.4:
        score += 2

    if neutral_ratio > 0.75 and emotion_variability < 0.15:
        score += 2

    if sad_trend > 0:
        score += 1

    # Risk level
    if score <= 3:
        risk = "Normal"
    elif score <= 6:
        risk = "Moderate"
    else:
        risk = "High"

    return {
        "neutral_ratio": round(neutral_ratio, 2),
        "sad_ratio": round(sad_ratio, 2),
        "happy_ratio": round(happy_ratio, 2),
        "angry_ratio": round(angry_ratio, 2),
        "fear_ratio": round(fear_ratio, 2),
        "stress_ratio": round(stress_ratio, 2),
        "emotion_variability": round(emotion_variability, 2),
        "confidence": round(avg_confidence, 2),
        "risk_level": risk,
        "score": score
    }