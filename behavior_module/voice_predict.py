from pathlib import Path
import joblib
import librosa
import numpy as np


# Absolute path relative to this file
MODEL_PATH = Path(__file__).resolve().parent / "ser_mlp_pipeline.joblib"

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
scaler = bundle["scaler"]
label_encoder = bundle["label_encoder"]
target_sr = int(bundle["target_sr"])
target_duration = float(bundle["target_duration"])


def pad_or_trim(y, target_length):
    if len(y) > target_length:
        return y[:target_length]
    if len(y) < target_length:
        return np.pad(y, (0, target_length - len(y)))
    return y


def preprocess_audio_array(y_raw, sr_raw):
    if sr_raw != target_sr:
        y_raw = librosa.resample(
            y_raw,
            orig_sr=sr_raw,
            target_sr=target_sr
        )

    target_samples = int(target_sr * target_duration)
    y = pad_or_trim(y_raw, target_samples)

    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y / peak

    return y.astype(np.float32)


def extract_feature_vector(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(chroma, axis=1),
        np.std(chroma, axis=1),
        np.mean(mel_db, axis=1),
        np.std(mel_db, axis=1),
    ])

    return np.nan_to_num(features).reshape(1, -1)


def predict_voice_emotion(audio_path):
    y_raw, sr_raw = librosa.load(audio_path, sr=None)

    y_proc = preprocess_audio_array(y_raw, sr_raw)

    features = extract_feature_vector(y_proc, target_sr)
    features_scaled = scaler.transform(features)

    probs = model.predict_proba(features_scaled)[0]

    pred_idx = int(np.argmax(probs))
    emotion = label_encoder.inverse_transform([pred_idx])[0]
    confidence = float(np.max(probs))

    return emotion, confidence