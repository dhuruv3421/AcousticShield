import os
import sys
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

# Allow imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from model import build_cnn_model

# ---------------------------
# Paths
# ---------------------------
SPECTROGRAM_DIR = os.path.join(PROJECT_ROOT, "clean_spectrogram")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "sound_classifier.h5")
PLOT_PATH = os.path.join(MODEL_DIR, "training_history.png")
LOG_PATH = os.path.join(MODEL_DIR, "training_log.txt")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.2
EPOCHS = 10
EARLY_STOP_PATIENCE = 3
SEED = 42

# ---------------------------
# Logging
# ---------------------------
def log(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)

    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")

# ---------------------------
# Dataset
# ---------------------------
def load_datasets():

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        SPECTROGRAM_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        SPECTROGRAM_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )

    return train_ds, val_ds, train_ds.class_names

# ---------------------------
# Pipeline
# ---------------------------
def build_pipeline(dataset):
    normalization_layer = tf.keras.layers.Rescaling(1./255)

    dataset = dataset.map(
        lambda x, y: (normalization_layer(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return dataset.prefetch(tf.data.AUTOTUNE)

# ---------------------------
# Callbacks
# ---------------------------
class EpochLogger(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        log(
            f"Epoch {epoch+1} | "
            f"acc={logs['accuracy']:.4f} "
            f"val_acc={logs['val_accuracy']:.4f}"
        )

def get_callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            save_best_only=True
        ),
        EpochLogger()
    ]

# ---------------------------
# Class Weights
# ---------------------------
def compute_weights(train_ds):

    labels = []

    for _, y in train_ds.unbatch():
        labels.append(tf.argmax(y).numpy())

    labels = np.array(labels)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )

    return dict(enumerate(weights))

# ---------------------------
# Plot
# ---------------------------
def plot_history(history):

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Accuracy")

    plt.subplot(1,2,2)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Loss")

    plt.savefig(PLOT_PATH)

# ---------------------------
# Train
# ---------------------------
def train():

    log("Training started")

    os.makedirs(MODEL_DIR, exist_ok=True)

    train_ds, val_ds, class_names = load_datasets()

    train_ds = build_pipeline(train_ds)
    val_ds = build_pipeline(val_ds)

    num_classes = len(class_names)
    log(f"Classes: {num_classes}")

    model = build_cnn_model(num_classes)

    class_weights = compute_weights(train_ds)
    log("Class weights computed")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=get_callbacks(),
        class_weight=class_weights
    )

    plot_history(history)

    with open(os.path.join(MODEL_DIR, "class_names.txt"), "w") as f:
        for c in class_names:
            f.write(c + "\n")

    log("Training complete")

# ---------------------------
if __name__ == "__main__":
    train()