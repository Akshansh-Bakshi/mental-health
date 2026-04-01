import pandas as pd
from collections import Counter

def analyze_emotions(csv_path: str):
    import pandas as pd
    from collections import Counter

    try:
        data = pd.read_csv(csv_path)

        # ✅ Ensure required columns exist
        required_cols = ["emotion", "confidence"]
        for col in required_cols:
            if col not in data.columns:
                return {"error": f"Missing column: {col}"}

        # ✅ Drop bad rows
        data = data.dropna(subset=["emotion"])

        # 🔥 TAKE LAST N RECORDS
        N = 100
        data = data.tail(N).reset_index(drop=True)

        if len(data) == 0:
            return {
                "emotion_list": [],
                "timestamps": [],
                "emotion_counts": {},
                "confidence": 0,
                "dominant_emotion": "none",
                "insight": "No data available"
            }

        emotions = data["emotion"].astype(str).tolist()

        # ✅ Safe timestamps
        if "timestamp" in data.columns:
            timestamps = data["timestamp"].astype(str).tolist()
        else:
            timestamps = list(range(len(emotions)))

        # 🔹 Ratios
        total = len(emotions)

        def ratio(e):
            return emotions.count(e) / total

        neutral_ratio = ratio("neutral")
        sad_ratio = ratio("sad")
        happy_ratio = ratio("happy")
        angry_ratio = ratio("angry")
        fear_ratio = ratio("fear")

        stress_ratio = angry_ratio + fear_ratio

        # 🔹 Variability
        changes = sum(
            1 for i in range(1, total) if emotions[i] != emotions[i-1]
        )
        emotion_variability = changes / total

        # 🔹 Confidence
        avg_confidence = data["confidence"].mean()

        # 🔹 Dominant emotion
        counter = Counter(emotions)
        dominant_emotion = counter.most_common(1)[0][0]

        # 🔹 Score
        score = 0
        if sad_ratio > 0.4: score += 2
        if happy_ratio < 0.2: score += 2
        if stress_ratio > 0.4: score += 2
        if neutral_ratio > 0.75 and emotion_variability < 0.15: score += 2

        # 🔹 Risk
        if score <= 3:
            risk = "Normal"
        elif score <= 6:
            risk = "Moderate"
        else:
            risk = "High"

        # 🔹 Insight
        if risk == "High":
            insight = "High stress detected. Consider rest."
        elif risk == "Moderate":
            insight = "Some emotional fluctuations detected."
        else:
            insight = "Your emotional state appears stable."

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

            "timestamps": timestamps,
            "emotion_list": emotions,

            "insight": insight
        }

    except Exception as e:
        return {
            "emotion_list": [],
            "timestamps": [],
            "emotion_counts": {},
            "error": str(e)
        }