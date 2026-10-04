import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from config import *


def build_cnn_lstm_model():
    model = models.Sequential(name="CNN_LSTM_Activity")

    model.add(
        layers.Input(
            shape=(SEQ_LEN, IMG_SIZE, IMG_SIZE, 3)
        )
    )

    for filters in (32, 64, 128):
        model.add(
            layers.TimeDistributed(
                layers.Conv2D(
                    filters, (3, 3),
                    activation="relu",
                    padding="same"
                )
            )
        )

        model.add(
            layers.TimeDistributed(
                layers.MaxPooling2D((2, 2))
            )
        )

    model.add(
        layers.TimeDistributed(
            layers.GlobalAveragePooling2D()
        )
    )

    model.add(layers.LSTM(128))
    model.add(layers.Dropout(DROPOUT))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

def train_model(
    model,
    x_train,
    y_train,
    x_val,
    y_val,
    class_weight=None
):
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight
    )

    model.save(MODEL_PATH)

    return history