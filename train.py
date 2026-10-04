import numpy as np

from config import *
from dataset_module import (
    load_dataset,
    get_class_weights
)
from model_module import (
    build_cnn_lstm_model,
    train_model
)
from evaluation_module import (
    calculate_metrics,
    plot_confusion_matrix,
    plot_training_graph
)


if __name__ == "__main__":
    x_train, y_train, y_train_id = load_dataset(
        TRAIN_DIR,
        augment=True
    )

    x_val, y_val, _ = load_dataset(
        VAL_DIR
    )

    x_test, y_test, y_test_id = load_dataset(
        TEST_DIR
    )

    model = build_cnn_lstm_model()

    model.summary()

    history = train_model(
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        class_weight=get_class_weights(
            y_train_id
        )
    )

    y_pred = np.argmax(
        model.predict(x_test),
        axis=1
    )

    calculate_metrics(
        y_test_id,
        y_pred
    )

    plot_confusion_matrix(
        y_test_id,
        y_pred
    )

    plot_training_graph(
        history
    )

    print(
        "Training finished. Best model saved to",
        MODEL_PATH
    )