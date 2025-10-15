import os
import argparse
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
TARGET_SAMPLE_RATE = 48000
PEAK_TARGET = 0.99  # target peak amplitude after normalization
RMS_TARGET_DBFS = None  # e.g., -20 for RMS normalization (optional)

print_lock = Lock()  # for thread-safe console printing

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def normalize_peak(y, peak_target=0.99):
    peak = np.max(np.abs(y))
    if peak == 0:
        return y
    return y * (peak_target / peak)

def normalize_rms(y, target_dbfs=-20.0):
    """Normalize to target RMS in dBFS."""
    rms = np.sqrt(np.mean(y**2))
    if rms == 0:
        return y
    current_dbfs = 20 * np.log10(rms)
    gain = 10 ** ((target_dbfs - current_dbfs) / 20)
    return y * gain

def process_file(in_path, out_path):
    """Load, normalize, and save a single file."""
    try:
        y, sr = librosa.load(in_path, sr=TARGET_SAMPLE_RATE, mono=True)

        # Apply normalization
        if RMS_TARGET_DBFS is not None:
            y = normalize_rms(y, RMS_TARGET_DBFS)
        else:
            y = normalize_peak(y, PEAK_TARGET)

        # Ensure output directory exists
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to processed folder
        sf.write(out_path, y, TARGET_SAMPLE_RATE)

        with print_lock:
            print(f"✅ Normalized: {in_path.name} → {out_path.relative_to(PROCESSED_DIR.parent)}")

    except Exception as e:
        with print_lock:
            print(f"❌ Error processing {in_path.name}: {e}")

# ------------------------------------------------------------
# Main routine
# ------------------------------------------------------------
def main(num_workers=8):
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw data folder not found: {RAW_DIR}")

    files = list(RAW_DIR.rglob("*.wav"))
    if not files:
        print("⚠️ No WAV files found in data/raw/")
        return

    print(f"🎧 Found {len(files)} files to process using {num_workers} threads...\n")

    futures = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for in_path in files:
            relative_path = in_path.relative_to(RAW_DIR)
            out_path = PROCESSED_DIR / relative_path
            futures.append(executor.submit(process_file, in_path, out_path))

        # Wait for all to complete
        for _ in as_completed(futures):
            pass

    print("\n🎯 All files normalized successfully!")

# ------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize audio files in parallel.")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of threads to use (default: 8).")
    args = parser.parse_args()

    main(num_workers=args.num_workers)