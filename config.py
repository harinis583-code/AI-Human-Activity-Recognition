import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "validation")
TEST_DIR = os.path.join(DATASET_DIR, "test")

MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

YOLO_WEIGHTS = os.path.join(MODEL_DIR, "yolo_weights.pt")
MODEL_PATH = os.path.join(MODEL_DIR, "activity_model.h5")
LOG_PATH = os.path.join(OUTPUT_DIR, "activity_log.csv")

CLASSES = [
    "walking",
    "running",
    "sitting",
    "standing",
    "falling"
]

NUM_CLASSES = len(CLASSES)

UNUSUAL = ["falling"]

IMG_SIZE = 224
SEQ_LEN = 16
FRAME_STEP = 3
BATCH_SIZE = 32
EPOCHS = 30
LEARN_RATE = 0.001
DROPOUT = 0.5

YOLO_CONF = 0.5
ALERT_CONF = 0.60

for folder in (MODEL_DIR, OUTPUT_DIR):
    os.makedirs(folder, exist_ok=True)