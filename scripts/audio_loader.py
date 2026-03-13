"""
audio_loader.py
---------------
Reusable audio preprocessing utilities for the Acoustic Shield CNN pipeline.

Functions:
    - load_audio(file_path)          : Load and preprocess a .wav file
    - create_mel_spectrogram(audio)  : Convert audio to Mel spectrogram
    - spectrogram_to_image(spec)     : Convert spectrogram matrix to 224x224 image
    - audio_to_image(file_path)      : Full pipeline: audio file → numpy image
"""

import numpy as np
import librosa
import cv2

# ─── Constants ────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 22050       # Standard audio sampling rate (Hz)
DURATION      = 5           # Maximum clip length in seconds
N_MELS        = 128         # Number of Mel filter banks
HOP_LENGTH    = 512         # Hop length for STFT
IMAGE_SIZE    = (224, 224)  # Output image dimensions (width, height)
# ──────────────────────────────────────────────────────────────────────────────


def load_audio(file_path: str) -> np.ndarray:
    """
    Load a .wav audio file, resample it, and enforce a fixed duration.

    Steps:
        1. Load the file using librosa (auto-resamples to SAMPLE_RATE).
        2. Trim to DURATION seconds if longer.
        3. Zero-pad (right) if shorter than DURATION seconds.

    Args:
        file_path (str): Absolute or relative path to the .wav file.

    Returns:
        np.ndarray: 1-D float32 audio signal of length SAMPLE_RATE * DURATION.

    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: If librosa cannot decode the audio.
    """
    # Load and resample in one step; librosa returns (signal, sr)
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    target_length = SAMPLE_RATE * DURATION  # Expected number of samples

    if len(audio) > target_length:
        # Truncate to exactly target_length samples
        audio = audio[:target_length]
    elif len(audio) < target_length:
        # Pad with zeros on the right to reach target_length
        padding = target_length - len(audio)
        audio = np.pad(audio, (0, padding), mode="constant")

    return audio


def create_mel_spectrogram(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Convert a raw audio signal into a Mel spectrogram in decibel (dB) scale.

    Steps:
        1. Compute the Mel spectrogram using librosa.
        2. Convert power to dB scale (log compression).

    Args:
        audio (np.ndarray): 1-D audio signal.
        sr    (int):        Sampling rate (default: SAMPLE_RATE).

    Returns:
        np.ndarray: 2-D float32 Mel spectrogram of shape (N_MELS, time_frames).
    """
    # Compute Mel spectrogram (power spectrum)
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH
    )

    # Convert power spectrogram to decibel units for better dynamic range
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec_db


def spectrogram_to_image(spec: np.ndarray) -> np.ndarray:
    """
    Convert a 2-D Mel spectrogram matrix into a 3-channel 224×224 NumPy image.

    Uses a fast NumPy + OpenCV path (no matplotlib figure rendering) so that
    the full dataset conversion stays well under 1 s/file on typical hardware.

    Steps:
        1. Normalise values to [0, 255] uint8.
        2. Apply OpenCV's COLORMAP_MAGMA for a perceptually uniform colour map.
        3. Flip vertically so low frequencies appear at the bottom (standard).
        4. Resize to IMAGE_SIZE (224×224).

    Args:
        spec (np.ndarray): 2-D Mel spectrogram (float), shape (n_mels, frames).

    Returns:
        np.ndarray: uint8 BGR image of shape (224, 224, 3).
    """
    # ── Normalise to 0-255 ────────────────────────────────────────────────────
    spec_min, spec_max = spec.min(), spec.max()
    spec_uint8 = ((spec - spec_min) / (spec_max - spec_min + 1e-8) * 255).astype(np.uint8)

    # ── Apply magma colour map (single OpenCV call, very fast) ────────────────
    # COLORMAP_MAGMA is available in OpenCV 4.x; fall back to INFERNO if needed
    try:
        coloured = cv2.applyColorMap(spec_uint8, cv2.COLORMAP_MAGMA)
    except AttributeError:
        coloured = cv2.applyColorMap(spec_uint8, cv2.COLORMAP_INFERNO)

    # ── Flip so frequency axis goes low → high (bottom → top) ────────────────
    coloured = cv2.flip(coloured, 0)

    # ── Resize to required spatial dimensions ─────────────────────────────────
    image = cv2.resize(coloured, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)

    return image  # shape: (224, 224, 3), dtype: uint8


def audio_to_image(file_path: str) -> np.ndarray:
    """
    End-to-end pipeline: .wav file path → 224×224×3 spectrogram image.

    Pipeline:
        load_audio → create_mel_spectrogram → spectrogram_to_image

    Args:
        file_path (str): Path to the .wav audio file.

    Returns:
        np.ndarray: uint8 image of shape (224, 224, 3).

    Raises:
        Exception: Propagated from any stage of the pipeline.
    """
    audio  = load_audio(file_path)              # Step 1 – load & pad/trim
    spec   = create_mel_spectrogram(audio)      # Step 2 – Mel spectrogram
    image  = spectrogram_to_image(spec)         # Step 3 – convert to image
    return image


# ─── Example usage ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    # Quick smoke-test: pass a .wav path as a command-line argument
    # Usage: python audio_loader.py path/to/sample.wav
    if len(sys.argv) < 2:
        print("Usage: python audio_loader.py <path_to_wav_file>")
        sys.exit(1)

    wav_path = sys.argv[1]
    print(f"[audio_loader] Processing: {wav_path}")

    try:
        img = audio_to_image(wav_path)
        print(f"[audio_loader] Output image shape : {img.shape}")
        print(f"[audio_loader] Output image dtype : {img.dtype}")
        print(f"[audio_loader] Pixel value range  : [{img.min()}, {img.max()}]")
    except Exception as e:
        print(f"[audio_loader] ERROR – {e}")
        sys.exit(1)
