"""
predict.py
----------
Runs prediction on ALL audio files inside test_audio/ folder.

Usage:
    python scripts/predict.py
"""

import os
import sys
import numpy as np
import tensorflow as tf

# Allow imports from scripts/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from audio_loader import audio_to_image


MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "sound_classifier.h5")
CLASS_NAMES_PATH = os.path.join(PROJECT_ROOT, "models", "class_names.txt")
TEST_AUDIO_DIR = os.path.join(PROJECT_ROOT, "test_audio")

IMAGE_SIZE = (224, 224)


# --------------------------------------------------
# Load class names
# --------------------------------------------------

def load_class_names():

    if not os.path.exists(CLASS_NAMES_PATH):
        print("ERROR: class_names.txt not found.")
        sys.exit(1)

    with open(CLASS_NAMES_PATH) as f:
        class_names = [line.strip() for line in f]

    print(f"Loaded {len(class_names)} classes")

    return class_names


# --------------------------------------------------
# Load model
# --------------------------------------------------

def load_model():

    if not os.path.exists(MODEL_PATH):
        print("ERROR: Model not found.")
        sys.exit(1)

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    return model


# --------------------------------------------------
# Preprocess audio
# --------------------------------------------------

def preprocess_audio(file_path):

    image = audio_to_image(file_path)

    image = image.astype(np.float32) / 255.0

    image = np.expand_dims(image, axis=0)

    return image


# --------------------------------------------------
# Predict one file
# --------------------------------------------------

def predict_file(model, class_names, file_path):

    input_tensor = preprocess_audio(file_path)

    predictions = model.predict(input_tensor, verbose=0)[0]

    index = np.argmax(predictions)

    predicted_class = class_names[index]

    confidence = predictions[index]

    return predicted_class, confidence


# --------------------------------------------------
# Predict all files
# --------------------------------------------------

def predict_all():

    model = load_model()

    class_names = load_class_names()

    if not os.path.exists(TEST_AUDIO_DIR):

        print("ERROR: test_audio folder not found")
        return

    files = [f for f in os.listdir(TEST_AUDIO_DIR) if f.endswith(".wav")]

    if len(files) == 0:

        print("No audio files found in test_audio/")
        return

    print(f"\nFound {len(files)} audio files\n")

    for file in files:

        file_path = os.path.join(TEST_AUDIO_DIR, file)

        print(f"Processing: {file}")

        predicted_class, confidence = predict_file(model, class_names, file_path)

        print(f"Prediction: {predicted_class}")
        print(f"Confidence: {confidence:.3f}\n")


# --------------------------------------------------

if __name__ == "__main__":

    predict_all()