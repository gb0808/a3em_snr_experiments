#!/usr/bin/env python3
"""
Apply median filtering in the time–frequency domain to all mixture files.
"""

import os
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import medfilt2d
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
MIXES_DIR = Path("data/mixes/train")
FILTERED_DIR = Path("results/median-filter")
SAMPLE_RATE = 48000
KERNEL_SIZE = (3, 3)  # (time, freq) median filter window
NUM_WORKERS = 8

print_lock = Lock()

# ------------------------------------------------------------
# Processing logic
# ------------------------------------------------------------
def apply_median_filter(in_path, out_path):
    try:
        y, sr = librosa.load(in_path, sr=SAMPLE_RATE, mono=True)

        # Compute STFT
        S = librosa.stft(y, n_fft=2048, hop_length=512)
        magnitude, phase = np.abs(S), np.angle(S)

        # Apply median filter in time–frequency domain
        filtered_mag = medfilt2d(magnitude, kernel_size=KERNEL_SIZE)

        # Reconstruct complex spectrogram and invert
        S_filtered = filtered_mag * np.exp(1j * phase)
        y_filtered = librosa.istft(S_filtered, hop_length=512)

        # Normalize to safe peak
        y_filtered = y_filtered / np.max(np.abs(y_filtered)) * 0.99

        # Ensure output directory exists
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Save filtered audio
        sf.write(out_path, y_filtered, SAMPLE_RATE)

        with print_lock:
            print(f"✅ Filtered: {in_path.relative_to(MIXES_DIR)} → {out_path.relative_to(FILTERED_DIR.parent)}")

    except Exception as e:
        with print_lock:
            print(f"❌ Error processing {in_path}: {e}")

# ------------------------------------------------------------
# Main routine
# ------------------------------------------------------------
def main():
    all_mix_files = list(MIXES_DIR.rglob("*.wav"))
    if not all_mix_files:
        print("⚠️ No mixture files found.")
        return

    print(f"🎧 Found {len(all_mix_files)} mixture files.")
    print(f"Applying median filtering (kernel={KERNEL_SIZE}) using {NUM_WORKERS} threads...\n")

    futures = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for in_path in all_mix_files:
            rel_path = in_path.relative_to(MIXES_DIR)
            out_path = FILTERED_DIR / rel_path
            futures.append(executor.submit(apply_median_filter, in_path, out_path))

        for _ in as_completed(futures):
            pass

    print("\n🎯 All mixtures filtered successfully!")
    print(f"Results saved in: {FILTERED_DIR}")

# ------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    main()