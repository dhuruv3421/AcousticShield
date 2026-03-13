"""
model.py
--------
Defines and compiles a CNN model for Mel spectrogram classification in the
Acoustic Shield pipeline.

Architecture overview:
    Input (224, 224, 3)
    → Conv2D(32)  + MaxPool
    → Conv2D(64)  + MaxPool
    → Conv2D(128) + MaxPool
    → Flatten
    → Dense(256) + Dropout(0.5)
    → Dense(num_classes, softmax)

Usage:
    from model import build_cnn_model
    model = build_cnn_model(num_classes=8)
    model.summary()
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


# ─── Constants ────────────────────────────────────────────────────────────────
INPUT_SHAPE  = (224, 224, 3)   # Height × Width × Channels
DROPOUT_RATE = 0.5             # Fraction of neurons dropped during training
DENSE_UNITS  = 256             # Neurons in the fully-connected hidden layer
LEARNING_RATE = 1e-3           # Adam default; easy to override via build_cnn_model
# ──────────────────────────────────────────────────────────────────────────────


def build_cnn_model(
    num_classes: int,
    input_shape: tuple = INPUT_SHAPE,
    learning_rate: float = LEARNING_RATE,
) -> tf.keras.Model:
    """
    Build and compile a CNN model for spectrogram-based sound classification.

    Architecture:
        Block 1 : Conv2D(32,  3×3, relu) → MaxPool(2×2)
        Block 2 : Conv2D(64,  3×3, relu) → MaxPool(2×2)
        Block 3 : Conv2D(128, 3×3, relu) → MaxPool(2×2)
        Head    : Flatten → Dense(256, relu) → Dropout(0.5)
                  → Dense(num_classes, softmax)

    Compiled with:
        Optimizer : Adam (lr=learning_rate)
        Loss      : categorical_crossentropy
        Metrics   : accuracy

    Args:
        num_classes   (int):   Number of output classes (e.g. 8 for 8 sound types).
        input_shape   (tuple): Input tensor shape (H, W, C). Default: (224, 224, 3).
        learning_rate (float): Initial learning rate for Adam. Default: 1e-3.

    Returns:
        tf.keras.Model: Compiled Keras model ready for training.
    """
    model = models.Sequential(name="AcousticShield_CNN")

    # ── Input layer ────────────────────────────────────────────────────────────
    model.add(layers.Input(shape=input_shape, name="input"))

    # ── Convolutional Block 1 ──────────────────────────────────────────────────
    # 32 filters learn low-level features (edges, textures in the spectrogram)
    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv1"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool1"))

    # ── Convolutional Block 2 ──────────────────────────────────────────────────
    # 64 filters learn mid-level patterns (frequency bands, transients)
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool2"))

    # ── Convolutional Block 3 ──────────────────────────────────────────────────
    # 128 filters learn high-level audio signatures (bark, gunshot, etc.)
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv3"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool3"))

    # ── Flatten: 3-D feature maps → 1-D feature vector ───────────────────────
    model.add(layers.Flatten(name="flatten"))

    # ── Fully-Connected Head ───────────────────────────────────────────────────
    model.add(layers.Dense(DENSE_UNITS, activation="relu", name="dense1"))

    # Dropout prevents over-fitting by randomly zeroing 50% of activations
    model.add(layers.Dropout(DROPOUT_RATE, name="dropout"))

    # ── Output layer: one probability per class ───────────────────────────────
    model.add(layers.Dense(num_classes, activation="softmax", name="output"))

    # ── Compile ───────────────────────────────────────────────────────────────
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",   # Use with one-hot encoded labels
        metrics=["accuracy"],
    )

    return model


def get_class_names(spectrogram_dir: str) -> list[str]:
    """
    Discover class names by listing leaf folders inside `spectrogram_dir`.

    Walks the spectrogram directory and returns a sorted list of
    unique sub-sub-folder names (the actual class labels).

    Example:
        spectrograms/natural/bird  →  "bird"
        spectrograms/threat/gunshot →  "gunshot"

    Args:
        spectrogram_dir (str): Root folder containing generated spectrograms.

    Returns:
        list[str]: Sorted list of class name strings.
    """
    import os
    class_names = set()
    for root, dirs, files in os.walk(spectrogram_dir):
        # A leaf folder has .png files and no sub-directories
        if files and not dirs:
            class_names.add(os.path.basename(root))
    return sorted(class_names)


# ─── Example usage ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    import sys

    # ── Auto-detect number of classes from spectrograms/ folder ───────────────
    SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT    = os.path.dirname(SCRIPT_DIR)
    SPECTROGRAM_DIR = os.path.join(PROJECT_ROOT, "spectrograms")

    if os.path.isdir(SPECTROGRAM_DIR):
        classes    = get_class_names(SPECTROGRAM_DIR)
        num_classes = len(classes)
        print(f"[model] Detected {num_classes} classes: {classes}")
    else:
        # Fallback: use a sensible default for demonstration
        num_classes = 8
        print(f"[model] spectrograms/ not found – using num_classes={num_classes} as default.")

    if num_classes == 0:
        print("[model] ERROR – No classes detected. Run build_dataset.py first.")
        sys.exit(1)

    # ── Build & inspect the model ─────────────────────────────────────────────
    model = build_cnn_model(num_classes=num_classes)
    model.summary()

    # ── Quick shape sanity-check (no real data needed) ────────────────────────
    import numpy as np
    dummy_batch = np.zeros((1, 224, 224, 3), dtype=np.float32)   # Fake image
    preds       = model.predict(dummy_batch, verbose=0)
    print(f"\n[model] Output shape for a single image: {preds.shape}")
    print(f"[model] Probabilities sum to: {preds.sum():.4f}  (should be ≈ 1.0)")
    print("[model] Model is ready for training.")
