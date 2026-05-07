import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa

# -----------------------
# Load YAMNet
# -----------------------
print("Loading YAMNet model...")
model = hub.load("https://tfhub.dev/google/yamnet/1")

class_map_path = model.class_map_path().numpy().decode('utf-8')
class_names = [line.strip().split(',')[2] for line in open(class_map_path)]

print("Model loaded successfully!\n")


# -----------------------
# Load audio
# -----------------------
def load_audio(file_path):
    waveform, sr = librosa.load(file_path, sr=16000)
    return waveform


# -----------------------
# Predict (Top-3)
# -----------------------
def predict(file_path):

    waveform = load_audio(file_path)

    scores, _, _ = model(waveform)
    scores = scores.numpy()

    mean_scores = np.mean(scores, axis=0)

    # Top 3 predictions
    top_indices = np.argsort(mean_scores)[-3:][::-1]
    results = [(class_names[i], mean_scores[i]) for i in top_indices]

    return results


# -----------------------
# CATEGORY MAPPING (KEY FIX)
# -----------------------
CATEGORY_MAP = {
    "chainsaw": ["chainsaw", "lawn mower", "engine", "power tool"],
    "gunshot": ["gunshot", "gunfire", "explosion", "drill", "firecracker"],
    "frog": ["frog", "insect", "bee"],
    "crow": ["bird", "crow"],
    "forest": ["bird", "insect", "wind"]
}


# -----------------------
# Smart correctness logic
# -----------------------
def is_correct(filename, predictions):

    filename = filename.lower()
    labels = " ".join([label.lower() for label, _ in predictions])

    for key, values in CATEGORY_MAP.items():
        if key in filename:
            return any(v in labels for v in values)

    return False


# -----------------------
# Threat detection logic (IMPROVED)
# -----------------------
THREAT_KEYWORDS = [
    "gunshot",
    "gunfire",
    "explosion",
    "chainsaw",
    "power tool",
    "drill",
    "engine",
    "firecracker",
    "siren",
    "alarm"
]

def is_threat(predictions):
    labels = " ".join([label.lower() for label, _ in predictions])
    return any(k in labels for k in THREAT_KEYWORDS)


# -----------------------
# Evaluate all files
# -----------------------
TEST_FOLDER = "test_audio"

correct = 0
total = 0
threat_count = 0

print("\n🔍 Evaluating all audio files:\n")

for file in os.listdir(TEST_FOLDER):

    if not file.endswith(".wav"):
        continue

    file_path = os.path.join(TEST_FOLDER, file)

    predictions = predict(file_path)

    # Accuracy
    result = is_correct(file, predictions)

    if result:
        status = "✅ CORRECT"
        correct += 1
    else:
        status = "❌ WRONG"

    # Threat detection
    if is_threat(predictions):
        threat_status = "🚨 THREAT"
        threat_count += 1
    else:
        threat_status = "🌿 SAFE"

    total += 1

    print(f"{file}")
    print(" → Top Predictions:")
    for label, conf in predictions:
        print(f"    - {label} ({conf:.2f})")

    print(f" → Accuracy: {status}")
    print(f" → Threat: {threat_status}")
    print("-" * 50)


# -----------------------
# Final Results
# -----------------------
print("\n📊 FINAL RESULTS:")
print(f"Accuracy: {correct}/{total} = {(correct/total)*100:.2f}%")
print(f"Threats detected: {threat_count}/{total}")