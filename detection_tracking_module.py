from ultralytics import YOLO
from config import YOLO_WEIGHTS, YOLO_CONF


def load_yolo_model():
    return YOLO(YOLO_WEIGHTS)


def detect_humans(model, frame):
    boxes = []

    results = model(frame, verbose=False)[0]

    for box in results.boxes:
        class_id = int(box.cls[0])
        conf = float(box.conf[0])

        if class_id == 0 and conf >= YOLO_CONF:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append(
                [x1, y1, x2, y2, conf]
            )

    return boxes


def iou(a, b):
    xa = max(a[0], b[0])
    ya = max(a[1], b[1])
    xb = min(a[2], b[2])
    yb = min(a[3], b[3])

    inter = max(0, xb - xa) * max(0, yb - ya)

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])

    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


class PersonTracker:

    def __init__(
        self,
        iou_threshold=0.3,
        max_missed=10
    ):
        self.tracks = {}
        self.next_id = 1
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed

    def update(self, detections):
        assigned = {}
        used_ids = set()

        for det in detections:
            best_id = None
            best_iou = self.iou_threshold

            for pid, (old_box, _) in self.tracks.items():

                if pid in used_ids:
                    continue

                score = iou(det, old_box)

                if score > best_iou:
                    best_id = pid
                    best_iou = score

            if best_id is None:
                best_id = self.next_id
                self.next_id += 1

            self.tracks[best_id] = [
                det[:4],
                0
            ]

            used_ids.add(best_id)
            assigned[best_id] = det[:4]

        for pid in list(self.tracks):

            if pid not in used_ids:
                self.tracks[pid][1] += 1

                if self.tracks[pid][1] > self.max_missed:
                    del self.tracks[pid]

        return assigned


def track_humans(tracker, detections):
    return tracker.update(detections)


def crop_person(frame, box):
    x1, y1, x2, y2 = box

    h, w = frame.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    return frame[y1:y2, x1:x2]