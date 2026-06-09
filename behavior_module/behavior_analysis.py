import pandas as pd
from collections import Counter


def analyze_emotions(csv_path: str):
    try:
        data = pd.read_csv(csv_path)

        # ================= REQUIRED COLUMNS =================
        required_cols = ["emotion", "confidence"]

        for col in required_cols:
            if col not in data.columns:
                return {"error": f"Missing column: {col}"}

        # ================= CLEAN DATA =================
        data = data.dropna(subset=["emotion"])

        # TAKE LAST N RECORDS
        N = 100
        data = data.tail(N).reset_index(drop=True)

        if len(data) == 0:
            return {
                "emotion_list": [],
                "timestamps": [],
                "emotion_counts": {},
                "face_counts": {},
                "voice_counts": {},
                "confidence": 0,
                "dominant_emotion": "none",
                "insight": "No data available"
            }

        # ================= BASE ARRAYS =================
        emotions = data["emotion"].astype(str).tolist()

        if "timestamp" in data.columns:
            timestamps = data["timestamp"].astype(str).tolist()
        else:
            timestamps = list(range(len(emotions)))

        # ================= MODALITY BREAKDOWN =================
        if "modality" in data.columns:
            face_data = data[data["modality"] == "face"]
            voice_data = data[data["modality"] == "voice"]

            face_counts = Counter(face_data["emotion"].astype(str).tolist())
            voice_counts = Counter(voice_data["emotion"].astype(str).tolist())
        else:
            face_counts = {}
            voice_counts = {}

        # ================= RATIOS =================
        total = len(emotions)

        def ratio(e):
            return emotions.count(e) / total

        neutral_ratio = ratio("neutral")
        sad_ratio = ratio("sad")
        happy_ratio = ratio("happy")
        angry_ratio = ratio("angry")
        fear_ratio = ratio("fear")

        stress_ratio = angry_ratio + fear_ratio

        # ================= VARIABILITY =================
        changes = sum(
            1 for i in range(1, total)
            if emotions[i] != emotions[i - 1]
        )

        emotion_variability = changes / total

        # ================= CONFIDENCE =================
        avg_confidence = data["confidence"].mean()

        # ================= DOMINANT EMOTION =================
        counter = Counter(emotions)
        dominant_emotion = counter.most_common(1)[0][0]

        # ================= RISK SCORE =================
        score = 0

        if sad_ratio > 0.4:
            score += 2

        if happy_ratio < 0.2:
            score += 2

        if stress_ratio > 0.4:
            score += 2

        if neutral_ratio > 0.75 and emotion_variability < 0.15:
            score += 2

        # ================= RISK LEVEL =================
        if score <= 3:
            risk = "Normal"
        elif score <= 6:
            risk = "Moderate"
        else:
            risk = "High"

        # ================= INSIGHT =================
        if risk == "High":
            insight = "High stress detected. Consider rest."
        elif risk == "Moderate":
            insight = "Some emotional fluctuations detected."
        else:
            insight = "Your emotional state appears stable."

        # ================= RETURN =================
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
            "score": score,

            "dominant_emotion": dominant_emotion,

            "emotion_counts": dict(counter),
            "face_counts": dict(face_counts),
            "voice_counts": dict(voice_counts),

            "timestamps": timestamps,
            "emotion_list": emotions,

            "insight": insight
            
        }

    except Exception as e:
        return {
            "emotion_list": [],
            "timestamps": [],
            "emotion_counts": {},
            "face_counts": {},
            "voice_counts": {},
            "error": str(e)
        }