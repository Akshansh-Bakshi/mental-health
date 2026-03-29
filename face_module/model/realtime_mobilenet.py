import csv
import os
import time
from collections import deque

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# 🔥 Emotion smoothing buffer
emotion_history = deque(maxlen=15)

# 🔥 Load trained model
model = load_model("emotion_mobilenet_finetuned.h5")

emotion_labels = ["angry", "fear", "happy", "neutral", "sad"]

# 🔥 Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
current_file_dir = os.path.dirname(os.path.abspath(__file__))

# Project root = 2 level upar
project_root = os.path.abspath(os.path.join(current_file_dir, "../../"))

# Shared folder path
shared_folder = os.path.join(project_root, "shared_outputs")

# Agar folder exist nahi karta to bana do
os.makedirs(shared_folder, exist_ok=True)

# Final CSV path
csv_file = os.path.join(shared_folder, "emotion_output.csv")

print("CSV Saving To:", csv_file)  # 🔥 Debug print

file_exists = os.path.isfile(csv_file)
file = open(csv_file, mode="a", newline="")
writer = csv.writer(file)

if not file_exists:
    writer.writerow(["timestamp", "emotion", "confidence"])
    file.flush()

# 🔥 Control logging every 1 second
last_logged_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
 
        face = frame[y:y+h, x:x+w]

        face = cv2.resize(face, (224,224))
        face = np.array(face, dtype=np.float32)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        prediction = model.predict(face, verbose=0)

        # 🔥 Smoothing
        emotion_history.append(prediction[0])
        avg_prediction = np.mean(emotion_history, axis=0)

        emotion_index = np.argmax(avg_prediction)
        confidence = float(np.max(avg_prediction))

        label = f"{emotion_labels[emotion_index]} ({confidence*100:.1f}%)"

        # 🔥 Log once per second (if confident)
        current_time = time.time()

        if current_time - last_logged_time >= 1:
            writer.writerow([
                current_time,
                emotion_labels[emotion_index],
                confidence
            ])
            file.flush()
            last_logged_time = current_time

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, label,
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0), 2)

    cv2.imshow("Real-Time Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
file.close()
cv2.destroyAllWindows()


import os
import subprocess

# Get project root
current_file = os.path.abspath(__file__)
face_folder = os.path.dirname(current_file)
project_root = os.path.dirname(face_folder)

behavior_script = os.path.join(project_root, "behavior_module", "behavior_analysis.py")

python_exe = os.path.join(project_root, ".venv", "Scripts", "python.exe")

print("\nRunning automatic behavior analysis...\n")

subprocess.run([python_exe, behavior_script])