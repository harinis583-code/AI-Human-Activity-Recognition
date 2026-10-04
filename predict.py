import cv2
import numpy as np
from tensorflow.keras.models import load_model

from config import *
from preprocessing_module import preprocess

model = load_model(MODEL_PATH)

video_path = input("Enter video path: ")

cap = cv2.VideoCapture(video_path)

frames = []

while len(frames) < SEQ_LEN:
    ret, frame = cap.read()

    if not ret:
        break

    frame = preprocess(frame)
    frames.append(frame)

cap.release()

if len(frames) == SEQ_LEN:
    x = np.array(frames)
    x = np.expand_dims(x, axis=0)

    prediction = model.predict(x)
    class_id = np.argmax(prediction)

    print("Predicted Activity:", CLASSES[class_id])
else:
    print("Not enough frames in video.")