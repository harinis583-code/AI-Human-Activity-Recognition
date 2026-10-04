import os
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from config import CLASSES, OUTPUT_DIR


def calculate_metrics(y_true, y_pred):
    print(
        "Accuracy :",
        round(
            accuracy_score(y_true, y_pred),
            4
        )
    )

    print(
        "Precision :",
        round(
            precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            ),
            4
        )
    )

    print(
        "Recall :",
        round(
            recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            ),
            4
        )
    )

    print(
        "F1-score :",
        round(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            ),
            4
        )
    )

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=CLASSES,
            zero_division=0
        )
    )


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")

    plt.xticks(
        range(len(CLASSES)),
        CLASSES,
        rotation=45
    )

    plt.yticks(
        range(len(CLASSES)),
        CLASSES
    )

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.xlabel("Predicted activity")
    plt.ylabel("Actual activity")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "confusion_matrix.png"
        )
    )

    plt.close()


def plot_training_graph(history):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(
        history.history["accuracy"],
        label="Training"
    )
    plt.plot(
        history.history["val_accuracy"],
        label="Validation"
    )

    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(
        history.history["loss"],
        label="Training"
    )
    plt.plot(
        history.history["val_loss"],
        label="Validation"
    )

    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "training_graph.png"
        )
    )

    plt.close()