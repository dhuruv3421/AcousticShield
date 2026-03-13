"""
build_dataset.py
----------------
Converts the entire Acoustic Shield raw audio dataset into Mel spectrogram
PNG images, mirroring the source folder structure under `spectrograms/`.

Source layout:
    dataset_final/
        natural/bird/audio.wav
        threat/gunshot/audio.wav
        ...

Output layout (auto-created):
    spectrograms/
        natural/bird/audio.png
        threat/gunshot/audio.png
        ...

Usage (from project root):
    python scripts/build_dataset.py
"""

import os
import sys
import cv2
from tqdm import tqdm

# ── Allow imports from the same `scripts/` directory ────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from audio_loader import audio_to_image   # noqa: E402  (local import after path fix)

# ─── Path configuration ───────────────────────────────────────────────────────
# Resolve paths relative to the project root (one level up from scripts/)
PROJECT_ROOT   = os.path.dirname(SCRIPT_DIR)
DATASET_DIR    = os.path.join(PROJECT_ROOT, "dataset_final")
SPECTROGRAM_DIR = os.path.join(PROJECT_ROOT, "spectrograms")
# ──────────────────────────────────────────────────────────────────────────────


def collect_wav_files(dataset_dir: str) -> list[tuple[str, str]]:
    """
    Walk the dataset directory and collect all .wav files.

    Args:
        dataset_dir (str): Root folder of the raw audio dataset.

    Returns:
        list of (wav_path, relative_path) tuples, where `relative_path`
        is the path *relative to dataset_dir* (used to mirror the structure).
    """
    wav_files = []
    for root, _dirs, files in os.walk(dataset_dir):
        for fname in files:
            if fname.lower().endswith(".wav"):
                full_path = os.path.join(root, fname)
                # Relative path keeps the folder hierarchy (e.g. natural/bird/audio.wav)
                rel_path  = os.path.relpath(full_path, dataset_dir)
                wav_files.append((full_path, rel_path))
    return wav_files


def build_output_path(rel_wav_path: str, output_root: str) -> str:
    """
    Convert a relative .wav path to the corresponding .png output path.

    Example:
        rel_wav_path = "natural/bird/118466.wav"
        → output     = "<output_root>/natural/bird/118466.png"

    Args:
        rel_wav_path (str): Relative path to the .wav file inside DATASET_DIR.
        output_root  (str): Root folder where spectrograms are saved.

    Returns:
        str: Full path to the output .png file.
    """
    # Replace .wav extension with .png
    rel_png_path = os.path.splitext(rel_wav_path)[0] + ".png"
    return os.path.join(output_root, rel_png_path)


def process_dataset(dataset_dir: str, output_dir: str) -> None:
    """
    Main processing loop: iterate over all .wav files, convert each to a
    spectrogram image, and save under `output_dir`.

    Args:
        dataset_dir (str): Root of the raw audio dataset.
        output_dir  (str): Root folder for generated spectrogram images.
    """
    # ── Collect all .wav files ────────────────────────────────────────────────
    print(f"[build_dataset] Scanning dataset: {dataset_dir}")
    wav_files = collect_wav_files(dataset_dir)

    if not wav_files:
        print("[build_dataset] No .wav files found. Exiting.")
        return

    total   = len(wav_files)
    success = 0
    failed  = 0

    print(f"[build_dataset] Found {total} .wav files. Starting conversion...\n")

    # ── Iterate with a tqdm progress bar ─────────────────────────────────────
    last_class = None  # Tracks the previously seen class for header printing

    for idx, (wav_path, rel_path) in enumerate(
        tqdm(wav_files, desc="Converting", unit="file", ncols=80), start=1
    ):
        # Normalise to forward slashes so the split works on Windows too
        rel_parts  = rel_path.replace("\\", "/").split("/")
        class_name = rel_parts[1] if len(rel_parts) >= 2 else "unknown"

        # Print per-class header whenever the class label changes
        if class_name != last_class:
            tqdm.write(f"\n  Processing class: {class_name}")
            last_class = class_name

        # ── Build output path and create directories ──────────────────────────
        out_path = build_output_path(rel_path, output_dir)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # Skip files that have already been converted (incremental runs)
        if os.path.exists(out_path):
            success += 1
            continue

        # ── Convert audio to spectrogram image ────────────────────────────────
        try:
            image = audio_to_image(wav_path)            # Full pipeline call
            cv2.imwrite(out_path, image)                # Save as PNG
            success += 1

        except Exception as e:
            # Log the error but continue processing remaining files
            tqdm.write(f"  [SKIP] {rel_path} — {type(e).__name__}: {e}")
            failed += 1

        # Periodic console update (every 50 files)
        if idx % 50 == 0:
            tqdm.write(f"  Processed: {idx}/{total}")

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n[build_dataset] ─── Summary ───────────────────────────────────")
    print(f"[build_dataset]   Total files  : {total}")
    print(f"[build_dataset]   Succeeded    : {success}")
    print(f"[build_dataset]   Failed/Skipped: {failed}")
    print(f"[build_dataset]   Output folder: {output_dir}")
    print(f"[build_dataset] ────────────────────────────────────────────────")


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Validate that the source dataset directory exists before doing any work
    if not os.path.isdir(DATASET_DIR):
        print(f"[build_dataset] ERROR – Dataset directory not found: {DATASET_DIR}")
        print("[build_dataset]          Make sure you run this from the project root,")
        print("[build_dataset]          or adjust DATASET_DIR at the top of this file.")
        sys.exit(1)

    process_dataset(DATASET_DIR, SPECTROGRAM_DIR)
