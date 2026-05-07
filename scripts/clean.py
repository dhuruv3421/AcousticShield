import os
import shutil
import json
from google import genai

# 🔑 ✅ PUT YOUR API KEY HERE
client = genai.Client(api_key="AIzaSyA2OgMb3qbYTDiCnkm-Gr4Sby2_ALNq518")

SOURCE_DIR = "spectrograms"
TARGET_DIR = "clean_spectrogram"
MAPPING_FILE = "class_mapping.json"

os.makedirs(TARGET_DIR, exist_ok=True)

# Step 1: Get folder names
folders = [
    f for f in os.listdir(SOURCE_DIR)
    if os.path.isdir(os.path.join(SOURCE_DIR, f))
]

print(f"✅ Found {len(folders)} folders")

# Step 2: Prompt
prompt = f"""
Group similar sound classes and merge duplicates.

Rules:
- Merge similar names (dog, dog_bark → dog)
- Keep names simple
- Every class must appear exactly once
- Return ONLY JSON

Format:
{{"clean_class": ["original1", "original2"]}}

Classes:
{folders}
"""

# Step 3: Gemini call
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=prompt
)

# Step 4: Extract + parse
try:
    text_output = response.text
    mapping = json.loads(text_output)
except Exception:
    print("❌ Invalid JSON from Gemini:")
    print(response.text)
    raise

# Step 5: Save mapping
with open(MAPPING_FILE, "w") as f:
    json.dump(mapping, f, indent=4)

print("✅ Mapping saved")

# Step 6: Copy files
def copy_files(src, dest):
    os.makedirs(dest, exist_ok=True)

    for file in os.listdir(src):
        src_file = os.path.join(src, file)
        dest_file = os.path.join(dest, file)

        if os.path.exists(dest_file):
            base, ext = os.path.splitext(file)
            dest_file = os.path.join(dest, base + "_dup" + ext)

        shutil.copy(src_file, dest_file)

# Step 7: Apply mapping
processed = set()

for clean_class, raw_classes in mapping.items():
    print(f"📂 {clean_class}")

    target_path = os.path.join(TARGET_DIR, clean_class)

    for raw in raw_classes:
        src_path = os.path.join(SOURCE_DIR, raw)

        if os.path.exists(src_path):
            copy_files(src_path, target_path)
            processed.add(raw)

# Step 8: Safety fallback
missing = set(folders) - processed

for m in missing:
    copy_files(
        os.path.join(SOURCE_DIR, m),
        os.path.join(TARGET_DIR, m)
    )

print("🎯 DONE")