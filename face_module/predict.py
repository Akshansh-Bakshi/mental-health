import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model", "emotion_model.h5")

model = load_model(model_path)

emotion_labels = ["angry", "fear", "happy", "neutral", "sad"]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def predict_emotion(image):
    
    print("Model input shape:", model.input_shape)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=7,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        return "no_face", 0.0

    # pick largest face
    faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
    x, y, w, h = faces[0]

    # padding
    pad = int(0.3 * w)

    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(gray.shape[1], x + w + pad)
    y2 = min(gray.shape[0], y + h + pad)

    face = image[y1:y2, x1:x2]

    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.GaussianBlur(face, (3, 3), 0)
    face = cv2.equalizeHist(face)
    face = cv2.resize(face, (48, 48))
    face = face.astype("float32") / 255.0
    face = face.reshape(1, 48, 48, 1)

    preds = model.predict(face)

    emotion = emotion_labels[np.argmax(preds)]
    confidence = float(np.max(preds))

    return emotion, confidence