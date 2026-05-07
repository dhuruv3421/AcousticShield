import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

# -----------------------
# CONFIG
# -----------------------
MODEL_PATH = "models\sound_classifier.h5"
TEST_FOLDER = "test_audio"
IMG_SIZE = (224, 224)

# -----------------------
# Load model
# -----------------------
model = tf.keras.models.load_model(MODEL_PATH)

# Load class names
with open("models/class_names.txt") as f:
    class_names = [line.strip() for line in f.readlines()]

print("Loaded classes:", len(class_names))


# -----------------------
# Convert audio → spectrogram
# -----------------------
def audio_to_spectrogram(file_path):
    y, sr = librosa.load(file_path, duration=5)

    # Mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Convert to image
    plt.figure(figsize=(2,2))
    librosa.display.specshow(S_dB, sr=sr)
    plt.axis('off')

    # Save temp image
    temp_path = "temp.png"
    plt.savefig(temp_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Read image
    img = cv2.imread(temp_path)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0

    return np.expand_dims(img, axis=0)


# -----------------------
# Prediction loop
# -----------------------
THREAT_CLASSES = [
    "gunshot",
    "chainsaw",
    "drilling",
    "explosion",
    "siren"
]

print("\n🔍 Predictions:\n")

for file in os.listdir(TEST_FOLDER):

    if not file.endswith(".wav"):
        continue

    file_path = os.path.join(TEST_FOLDER, file)

    img = audio_to_spectrogram(file_path)

    pred = model.predict(img, verbose=0)
    pred_class = class_names[np.argmax(pred)]
    confidence = np.max(pred)

    # Threat detection
    if pred_class in THREAT_CLASSES:
        status = "🚨 THREAT"
    else:
        status = "🌿 SAFE"

    print(f"{file}")
    print(f" → Predicted: {pred_class}")
    print(f" → Confidence: {confidence:.2f}")
    print(f" → Status: {status}")
    print("-" * 40)