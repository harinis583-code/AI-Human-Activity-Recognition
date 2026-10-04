import cv2
import numpy as np
from collections import deque
from config import IMG_SIZE, SEQ_LEN


def resize_frame(frame):
    return cv2.resize(frame, (IMG_SIZE, IMG_SIZE))


def remove_noise(frame):
    return cv2.GaussianBlur(frame, (3, 3), 0)


def normalize_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame.astype("float32") / 255.0


def preprocess(frame):
    return normalize_frame(
        remove_noise(
            resize_frame(frame)
        )
    )


def create_sequences(frames, seq_len=SEQ_LEN):
    sequences = []

    for start in range(0, len(frames) - seq_len + 1, seq_len):
        sequences.append(
            np.array(frames[start:start + seq_len])
        )

    return np.array(sequences)


class SequenceBuffer:

    def __init__(self):
        self.buffers = {}

    def add(self, person_id, crop):

        if person_id not in self.buffers:
            self.buffers[person_id] = deque(maxlen=SEQ_LEN)

        self.buffers[person_id].append(
            preprocess(crop)
        )

    def ready(self, person_id):
        return (
            person_id in self.buffers
            and len(self.buffers[person_id]) == SEQ_LEN
        )

    def get(self, person_id):
        return np.expand_dims(
            np.array(self.buffers[person_id]),
            axis=0
        )