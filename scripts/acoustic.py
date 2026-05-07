import os
import sys
import time
import queue
import threading
import argparse
import numpy as np
import torch
import librosa
import sounddevice as sd
import soundfile as sf
from datetime import datetime
from panns_inference import AudioTagging

# ─────────────────────────────────────────────
#  ANSI colour helpers
# ─────────────────────────────────────────────
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def banner():
    print(f"""
{CYAN}{BOLD}
  ╔═══════════════════════════════════════╗
  ║        🛡  ACOUSTIC SHIELD  🛡         ║
  ║   Real-time Threat Detection System   ║
  ╚═══════════════════════════════════════╝
{RESET}""")

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────
SAMPLE_RATE    = 32000   # PANNs native rate
INTERVAL_SEC   = 2       # analyse every N seconds
TOP_K          = 3       # top predictions to show
THREAT_THRESHOLD = 0.20  # min confidence to flag

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
    "scream", "shout", "yell",
]

DEFAULT_MODEL_PATH = r"D:\Acoustic_Shield\AcousticShield\models\Cnn14_mAP=0.431.pth"

# ─────────────────────────────────────────────
#  Model loader
# ─────────────────────────────────────────────
def load_model(checkpoint_path: str, device: str = "cpu"):
    print(f"{DIM}Loading model from {checkpoint_path} …{RESET}")
    model = AudioTagging(checkpoint_path=checkpoint_path, device=device)
    print(f"{GREEN}✔ Model loaded.{RESET}\n")
    return model

# ─────────────────────────────────────────────
#  Inference helpers
# ─────────────────────────────────────────────
def predict_waveform(model, waveform: np.ndarray):
    """Run PANNs inference on a 1-D float32 waveform @ 32 kHz."""
    x = waveform[None, :]                            # (1, T)
    clipwise_output, _ = model.inference(x)
    top_idx = clipwise_output[0].argsort()[-TOP_K:][::-1]
    labels  = [model.labels[i] for i in top_idx]
    scores  = clipwise_output[0][top_idx]
    return list(zip(labels, scores))

def predict_file(model, file_path: str):
    waveform, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    return predict_waveform(model, waveform)

def is_threat(results):
    for label, score in results:
        if score < THREAT_THRESHOLD:
            continue
        if any(k in label.lower() for k in THREAT_KEYWORDS):
            return True, label, float(score)
    return False, None, 0.0

# ─────────────────────────────────────────────
#  Pretty-print one analysis result
# ─────────────────────────────────────────────
def print_result(results, source: str = ""):
    ts = datetime.now().strftime("%H:%M:%S")
    threat, tlabel, tscore = is_threat(results)

    status_str = (
        f"{RED}{BOLD}⚠  THREAT DETECTED{RESET}"
        if threat else
        f"{GREEN}✔  SAFE{RESET}"
    )

    print(f"\n{DIM}[{ts}]{RESET} {CYAN}{source}{RESET}")
    for label, score in results:
        bar_len = int(score * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        kw_hit = any(k in label.lower() for k in THREAT_KEYWORDS)
        colour = RED if (kw_hit and score >= THREAT_THRESHOLD) else RESET
        print(f"   {colour}{label:<35}{RESET} [{bar}] {score*100:5.1f}%")

    print(f"   → {status_str}", end="")
    if threat:
        print(f"  ({tlabel}, {tscore*100:.1f}%)")
    else:
        print()
    print("─" * 60)

# ─────────────────────────────────────────────
#  FILE MODE
# ─────────────────────────────────────────────
def run_file_mode(model, folder: str):
    correct = total = 0
    print(f"\n{BOLD}Evaluating folder: {folder}{RESET}\n")

    for file in sorted(os.listdir(folder)):
        if not file.endswith(".wav"):
            continue
        path = os.path.join(folder, file)
        results = predict_file(model, path)
        print_result(results, source=file)
        total += 1

    if total == 0:
        print(f"{YELLOW}No .wav files found in '{folder}'.{RESET}")

# ─────────────────────────────────────────────
#  MIC MODE
# ─────────────────────────────────────────────
class MicMonitor:
    """
    Continuously records from the default microphone.
    Every INTERVAL_SEC seconds the accumulated audio is
    analysed in a background thread.
    """

    def __init__(self, model, interval: float = INTERVAL_SEC,
                 save_clips: bool = False, clips_dir: str = "clips"):
        self.model       = model
        self.interval    = interval
        self.save_clips  = save_clips
        self.clips_dir   = clips_dir
        self._audio_buf  = []          # accumulates raw frames
        self._lock       = threading.Lock()
        self._stop_event = threading.Event()
        self._q          = queue.Queue()  # analysis jobs

        if save_clips:
            os.makedirs(clips_dir, exist_ok=True)

    # sounddevice callback (called from audio thread)
    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"{YELLOW}[audio] {status}{RESET}", file=sys.stderr)
        with self._lock:
            self._audio_buf.append(indata[:, 0].copy())   # mono

    # timer fires every self.interval seconds
    def _snapshot_and_enqueue(self):
        while not self._stop_event.is_set():
            time.sleep(self.interval)
            with self._lock:
                if not self._audio_buf:
                    continue
                chunk = np.concatenate(self._audio_buf)
                self._audio_buf.clear()
            self._q.put(chunk)

    # analysis worker
    def _analysis_worker(self):
        clip_idx = 0
        while not self._stop_event.is_set() or not self._q.empty():
            try:
                chunk = self._q.get(timeout=0.5)
            except queue.Empty:
                continue

            # optionally save the raw clip
            if self.save_clips:
                fname = os.path.join(
                    self.clips_dir,
                    f"clip_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{clip_idx:04d}.wav"
                )
                sf.write(fname, chunk, SAMPLE_RATE)
                clip_idx += 1

            # PANNs needs float32 in [-1, 1]
            waveform = chunk.astype(np.float32)
            if waveform.max() > 0:
                waveform /= (np.abs(waveform).max() + 1e-9)

            results = predict_waveform(self.model, waveform)
            print_result(results, source=f"mic  Δt={self.interval}s")

    def run(self):
        print(f"{CYAN}{BOLD}🎙  Mic monitoring started  "
              f"(interval={self.interval}s, Ctrl+C to stop){RESET}\n")
        print("─" * 60)

        snapshot_thread  = threading.Thread(target=self._snapshot_and_enqueue, daemon=True)
        analysis_thread  = threading.Thread(target=self._analysis_worker,      daemon=True)
        snapshot_thread.start()
        analysis_thread.start()

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
        ):
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print(f"\n{YELLOW}Stopping…{RESET}")
                self._stop_event.set()

        snapshot_thread.join()
        analysis_thread.join()
        print(f"{GREEN}Monitoring stopped.{RESET}")

# ─────────────────────────────────────────────
#  CLI entry-point
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Acoustic Shield – PANNs-based threat detector"
    )
    p.add_argument(
        "--mode", choices=["mic", "file"], default="mic",
        help="'mic'  = real-time microphone monitoring (default)\n"
             "'file' = evaluate a folder of .wav files",
    )
    p.add_argument(
        "--model", default=DEFAULT_MODEL_PATH,
        help="Path to PANNs checkpoint (.pth)"
    )
    p.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"],
        help="Inference device"
    )
    # mic-mode options
    p.add_argument(
        "--interval", type=float, default=INTERVAL_SEC,
        help="Analysis interval in seconds (mic mode, default 2)"
    )
    p.add_argument(
        "--save-clips", action="store_true",
        help="Save each microphone clip as a .wav file"
    )
    p.add_argument(
        "--clips-dir", default="clips",
        help="Directory for saved mic clips (default: ./clips)"
    )
    # file-mode options
    p.add_argument(
        "--folder", default="test_audio",
        help="Folder containing .wav files (file mode, default: test_audio)"
    )
    return p.parse_args()


def main():
    banner()
    args = parse_args()

    model = load_model(args.model, args.device)

    if args.mode == "mic":
        monitor = MicMonitor(
            model,
            interval=args.interval,
            save_clips=args.save_clips,
            clips_dir=args.clips_dir,
        )
        monitor.run()

    elif args.mode == "file":
        run_file_mode(model, args.folder)


if __name__ == "__main__":
    main()