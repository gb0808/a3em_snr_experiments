"""
Apply wavelet denoising to all mixture files.
"""

import os
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import pywt
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
MIXES_DIR = Path("data/mixes/train")

SAMPLE_RATE = 48000
WAVELET = ['db8', 'sym5', 'coif5']       # Daubechies wavelet
LEVEL = None          # Decomposition level (None = max level)
THRESHOLD_METHOD = ['soft', 'hard']  # 'soft' or 'hard'
NUM_WORKERS = 8

print_lock = Lock()

# ------------------------------------------------------------
# Wavelet denoising function
# ------------------------------------------------------------
def wavelet_denoise(signal, wavelet='db8', level=None, threshold_method='soft'):
    # Decompose signal
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)

    # Estimate noise sigma using the detail coefficients at the first level
    detail_coeffs = coeffs[-1]
    sigma = np.median(np.abs(detail_coeffs)) / 0.6745

    # Universal threshold
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))

    # Apply thresholding to all detail coefficients
    coeffs_thresh = [coeffs[0]]  # Approximation remains untouched
    for c in coeffs[1:]:
        coeffs_thresh.append(pywt.threshold(c, threshold, mode=threshold_method))

    # Reconstruct signal
    return pywt.waverec(coeffs_thresh, wavelet=wavelet)

# ------------------------------------------------------------
# Processing logic
# ------------------------------------------------------------
def apply_wavelet_denoise(in_path, out_path, wavelet, threshold):
    try:
        y, sr = librosa.load(in_path, sr=SAMPLE_RATE, mono=True)

        # Apply wavelet denoising
        y_denoised = wavelet_denoise(y, wavelet, level=LEVEL, threshold_method=threshold)

        # Normalize to safe peak
        y_denoised = y_denoised / np.max(np.abs(y_denoised)) * 0.99

        # Ensure output directory exists
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Save denoised audio
        sf.write(out_path, y_denoised, SAMPLE_RATE)

        with print_lock:
            print(f"✅ Denoised: {in_path} → {out_path}")

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
    print(f"Applying wavelet denoising (wavelet={WAVELET}) using {NUM_WORKERS} threads...\n")

    futures = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for in_path in all_mix_files:
            for wavelet in WAVELET:
                for mode in THRESHOLD_METHOD:
                    FILTERED_DIR = Path(f"results/wavelet-denoise-{wavelet}-{mode}")
                    rel_path = in_path.relative_to(MIXES_DIR)
                    out_path = FILTERED_DIR / rel_path
                    futures.append(executor.submit(apply_wavelet_denoise, in_path, out_path, wavelet, mode))

        for _ in as_completed(futures):
            pass

    print("\n🎯 All mixtures denoised successfully!")
    print(f"Results saved in: {FILTERED_DIR}")

# ------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    main()