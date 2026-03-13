"""
train_model.py
--------------
Trains the Acoustic Shield CNN on Mel spectrogram images.

Optimized for low RAM systems (8GB) + logging.

Logs training progress to:
models/training_log.txt
"""

import os
import sys
import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

# Allow imports from scripts/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from model import build_cnn_model


# --------------------------------------------------
# Paths
# --------------------------------------------------

SPECTROGRAM_DIR = os.path.join(PROJECT_ROOT, "spectrograms")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "sound_classifier.h5")
PLOT_PATH = os.path.join(MODEL_DIR, "training_history.png")
LOG_PATH = os.path.join(MODEL_DIR, "training_log.txt")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 4
VALIDATION_SPLIT = 0.2
EPOCHS = 10
EARLY_STOP_PATIENCE = 3
SEED = 42


# --------------------------------------------------
# Logging
# --------------------------------------------------

def log(message):

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"

    print(line)

    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# --------------------------------------------------
# Dataset Loading
# --------------------------------------------------

def load_datasets():

    log("Loading training dataset")

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        SPECTROGRAM_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=True
    )

    log("Loading validation dataset")

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        SPECTROGRAM_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False
    )

    class_names = train_ds.class_names

    log(f"Detected {len(class_names)} classes: {class_names}")

    return train_ds, val_ds, class_names


# --------------------------------------------------
# Dataset Pipeline
# --------------------------------------------------

def build_pipeline(dataset):

    normalization_layer = tf.keras.layers.Rescaling(1./255)

    dataset = dataset.map(
        lambda x, y: (normalization_layer(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.prefetch(buffer_size=1)

    return dataset


# --------------------------------------------------
# Callbacks
# --------------------------------------------------

class EpochLogger(tf.keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs=None):
        log(f"Starting epoch {epoch+1}")

    def on_epoch_end(self, epoch, logs=None):

        acc = logs.get("accuracy")
        val_acc = logs.get("val_accuracy")

        loss = logs.get("loss")
        val_loss = logs.get("val_loss")

        log(
            f"Epoch {epoch+1} finished | "
            f"acc={acc:.4f} val_acc={val_acc:.4f} "
            f"loss={loss:.4f} val_loss={val_loss:.4f}"
        )


def get_callbacks():

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOP_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    epoch_logger = EpochLogger()

    return [early_stop, checkpoint, epoch_logger]


# --------------------------------------------------
# Plot Training History
# --------------------------------------------------

def plot_history(history):

    epochs = range(1, len(history.history["accuracy"]) + 1)

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(epochs, history.history["accuracy"], label="Train")
    plt.plot(epochs, history.history["val_accuracy"], label="Validation")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, history.history["loss"], label="Train")
    plt.plot(epochs, history.history["val_loss"], label="Validation")
    plt.title("Loss")
    plt.legend()

    plt.savefig(PLOT_PATH)

    log(f"Training graph saved to {PLOT_PATH}")


# --------------------------------------------------
# Training
# --------------------------------------------------

def train():

    log("Training script started")

    if not os.path.exists(SPECTROGRAM_DIR):

        log("ERROR: spectrogram folder not found. Run build_dataset.py first.")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)

    train_ds, val_ds, class_names = load_datasets()

    train_ds = build_pipeline(train_ds)
    val_ds = build_pipeline(val_ds)

    num_classes = len(class_names)

    log(f"Building CNN model for {num_classes} classes")

    model = build_cnn_model(num_classes=num_classes)

    callbacks = get_callbacks()

    log("Training started")

    try:

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=callbacks
        )

        log("Training completed successfully")

        plot_history(history)

        labels_path = os.path.join(MODEL_DIR, "class_names.txt")

        with open(labels_path, "w") as f:
            for name in class_names:
                f.write(name + "\n")

        log(f"Class names saved to {labels_path}")

    except Exception as e:

        log(f"Training crashed: {str(e)}")


# --------------------------------------------------

if __name__ == "__main__":
    train()