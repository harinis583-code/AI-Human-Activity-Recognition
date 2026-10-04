import glob
import os
import numpy as np
import cv2
from sklearn.utils.class_weight import compute_class_weight

from config import *
from preprocessing_module import preprocess


def augment_frame(frame):
    if np.random.rand() < 0.5:
        frame = cv2.flip(frame, 1)

    alpha = np.random.uniform(0.7, 1.3)

    frame = cv2.convertScaleAbs(
        frame,
        alpha=alpha,
        beta=0
    )

    angle = np.random.uniform(-10, 10)

    m = cv2.getRotationMatrix2D(
        (frame.shape[1] // 2, frame.shape[0] // 2),
        angle,
        1.0
    )

    return cv2.warpAffine(
        frame,
        m,
        (frame.shape[1], frame.shape[0])
    )


def load_clip(path, augment=False):
    cap = cv2.VideoCapture(path)

    total = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    indexes = np.linspace(
        0,
        max(total - 1, 0),
        SEQ_LEN
    ).astype(int)

    frames = []

    for idx in indexes:
        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(idx)
        )

        ok, frame = cap.read()

        if not ok:
            frame = np.zeros(
                (IMG_SIZE, IMG_SIZE, 3),
                dtype="uint8"
            )

        frame = cv2.resize(
            frame,
            (IMG_SIZE, IMG_SIZE)
        )

        if augment:
            frame = augment_frame(frame)

        frames.append(preprocess(frame))

    cap.release()

    return np.array(frames)


def load_dataset(folder, augment=False):
    x_data, y_data = [], []

    for class_id, name in enumerate(CLASSES):
        for path in glob.glob(
            os.path.join(folder, name, "*.mp4")
        ):
            x_data.append(
                load_clip(path, augment)
            )

            y_data.append(class_id)

    x_data = np.array(
        x_data,
        dtype="float32"
    )

    y_data = np.array(y_data)

    return (
    x_data,
    y_data,
    y_data
)


def get_class_weights(y_labels):
    weights = compute_class_weight(
        "balanced",
        classes=np.unique(y_labels),
        y=y_labels
    )

    return dict(enumerate(weights))
