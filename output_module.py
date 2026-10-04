import csv
import cv2
from datetime import datetime

from config import LOG_PATH


def draw_results(
    frame,
    box,
    person_id,
    activity,
    confidence,
    alert
):
    x1, y1, x2, y2 = box

    colour = (
        (0, 0, 255)
        if alert
        else (0, 200, 0)
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        colour,
        2
    )

    text = "ID %d: %s (%.2f)" % (
        person_id,
        activity,
        confidence
    )

    cv2.putText(
        frame,
        text,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        colour,
        2
    )

    if alert:
        cv2.putText(
            frame,
            "ALERT: FALL DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3
        )

    return frame


def generate_alert(person_id, frame_no):
    print(
        "[ALERT] Fall detected - person %d, frame %d"
        % (person_id, frame_no)
    )


def init_log():
    with open(
        LOG_PATH,
        "w",
        newline=""
    ) as f:
        csv.writer(f).writerow(
            [
                "log_id",
                "timestamp",
                "frame_no",
                "person_id",
                "activity",
                "confidence",
                "alert"
            ]
        )


def write_log(
    log_id,
    frame_no,
    person_id,
    activity,
    confidence,
    alert
):
    now = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    with open(
        LOG_PATH,
        "a",
        newline=""
    ) as f:
        csv.writer(f).writerow(
            [
                log_id,
                now,
                frame_no,
                person_id,
                activity,
                confidence,
                "Yes" if alert else "No"
            ]
        )