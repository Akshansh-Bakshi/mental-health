import os
import pandas as pd
import numpy as np
import cv2

# Load CSV
data = pd.read_csv("fer2013.csv")

emotion_map = {
    0: "angry",
    1: "disgust",   # ignore later
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",  # ignore later
    6: "neutral"
}

# Create folders if not exist
for folder in ["train", "test"]:
    for emotion in ["angry","fear","happy","neutral","sad"]:
        os.makedirs(f"dataset/{folder}/{emotion}", exist_ok=True)

for index, row in data.iterrows():
    emotion = emotion_map[row["emotion"]]

    # ignore disgust and surprise
    if emotion not in ["angry","fear","happy","neutral","sad"]:
        continue

    pixels = np.array(row["pixels"].split(), dtype="uint8")
    image = pixels.reshape(48,48)

    usage = row["Usage"]

    if usage == "Training":
        folder = "train"
    else:
        folder = "test"

    filename = f"dataset/{folder}/{emotion}/{index}.jpg"
    cv2.imwrite(filename, image)

print("Conversion Done ✅")
