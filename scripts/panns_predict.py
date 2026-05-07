import os
import torch
from panns_inference import AudioTagging
import librosa

# -----------------------
# Load model
# -----------------------
model = AudioTagging(
    checkpoint_path=r"D:\Acoustic_Shield\AcousticShield\models\Cnn14_mAP=0.431.pth",
    device='cpu'
)

# -----------------------
# Threat keywords
# -----------------------
THREAT_KEYWORDS = [
    "gunshot", "gunfire", "explosion", "blast", "artillery", "bomb",
    "firecracker", "fireworks",
    "chainsaw", "power tool", "drill", "cutting", "grinding",
    "sawing", "tools", "wood", "filing", "rasp",
    "engine", "motor", "vehicle", "truck", "motorcycle",
    "helicopter", "heavy engine",
    "glass", "breaking", "shatter", "impact", "metal",
    "hammer", "knock", "crash", "thud",
    "siren", "alarm",
    "scream", "shout", "yell"
]

# -----------------------
# Threat detection
# -----------------------
def is_threat(results):
    for label, score in results:
        if score < 0.2:
            continue
        if any(k in label.lower() for k in THREAT_KEYWORDS):
            return True
    return False

# -----------------------
# Prediction (Top 3)
# -----------------------
def predict(file_path):

    waveform, sr = librosa.load(file_path, sr=32000)
    waveform = waveform[None, :]

    clipwise_output, _ = model.inference(waveform)

    top_indices = clipwise_output[0].argsort()[-3:][::-1]

    labels = [model.labels[i] for i in top_indices]
    scores = clipwise_output[0][top_indices]

    return list(zip(labels, scores))

# -----------------------
# Accuracy logic
# -----------------------
def is_correct(filename, predictions):

    filename = filename.lower()
    labels = " ".join([l.lower() for l, _ in predictions])

    if "chainsaw" in filename:
        return any(x in labels for x in ["chainsaw", "tool", "saw"])

    if "gunshot" in filename:
        return any(x in labels for x in ["gunshot", "fire", "explosion"])

    if "crow" in filename:
        return "crow" in labels or "bird" in labels

    if "frog" in filename:
        return "frog" in labels

    if "forest" in filename:
        return any(x in labels for x in ["insect", "bird", "wind"])

    return False

# -----------------------
# Evaluate folder
# -----------------------
TEST_FOLDER = "test_audio"

correct = 0
total = 0

print("\nEvaluating with PANNs:\n")

for file in os.listdir(TEST_FOLDER):

    if not file.endswith(".wav"):
        continue

    path = os.path.join(TEST_FOLDER, file)

    results = predict(path)

    print(file)
    for label, score in results:
        print(f"   - {label} ({score*100:.2f}%)")

    if is_threat(results):
        print("   -> THREAT")
    else:
        print("   -> SAFE")

    # Accuracy check
    if is_correct(file, results):
        correct += 1

    total += 1

    print("-" * 50)

# -----------------------
# Final accuracy
# -----------------------
if total > 0:
    accuracy = (correct / total) * 100
    print(f"\nFinal Accuracy: {correct}/{total} = {accuracy:.2f}%")
else:
    print("No audio files found.")