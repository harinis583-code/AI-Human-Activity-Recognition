import streamlit as st
import tempfile
import cv2
import numpy as np
from tensorflow.keras.models import load_model

from config import MODEL_PATH, CLASSES, IMG_SIZE, SEQ_LEN
from preprocessing_module import preprocess


st.title("AI Human Activity Recognition")

st.write("Upload a CCTV video to predict human activity.")


video = st.file_uploader(
    "Choose a video",
    type=["mp4"]
)


if video is not None:

    st.video(video)

    if st.button("Predict Activity"):

        # Save uploaded video temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as temp_file:

            temp_file.write(video.read())
            video_path = temp_file.name

        # Open video
        cap = cv2.VideoCapture(video_path)

        frames = []

        while len(frames) < SEQ_LEN:

            ret, frame = cap.read()

            if not ret:
                break

            frame = preprocess(frame)
            frames.append(frame)

        cap.release()

        # Check frames
        if len(frames) == SEQ_LEN:

            x = np.array(frames)
            x = np.expand_dims(x, axis=0)

            # Load trained model
            model = load_model(MODEL_PATH)

            # Prediction
            prediction = model.predict(
                x,
                verbose=0
            )

            class_id = np.argmax(prediction)

            activity = CLASSES[class_id]

            st.success(
                f"Predicted Activity: {activity}"
            )

        else:

            st.error(
                "Not enough frames in the video."
            )
            
