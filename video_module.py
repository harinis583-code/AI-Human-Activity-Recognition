import os
import cv2

from config import FRAME_STEP


def open_video(source):
    try:
        if isinstance(source, str) and not os.path.exists(source):
            raise FileNotFoundError("Video file not found: " + source)

        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise IOError("Cannot open the video source.")

        return cap

    except Exception as err:
        print("[ERROR]", err)
        return None


def extract_frames(cap):
    frame_no = 0

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        if frame is None or frame.size == 0:
            continue

        if frame_no % FRAME_STEP == 0:
            yield frame_no, frame

        frame_no += 1

    cap.release()


def save_output_video(path, width, height, fps=25):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    return cv2.VideoWriter(
        path,
        fourcc,
        fps,
        (width, height)
    )