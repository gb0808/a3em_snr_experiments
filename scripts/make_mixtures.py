import os
import json
import random
import csv
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
MIXES_DIR = DATA_DIR / "mixes"
ANNOTATIONS_DIR = DATA_DIR / "annotations"

SPLITS = ["train", "val", "test"]
SPLIT_RATIOS = [0.8, 0.1, 0.1]
TARGET_SNRS_DB = [-10, -5, 0, 5]
MIX_DURATION_S = 5.0
SAMPLE_RATE = 48000
SEED = 42
NUM_WORKERS = 8  # threads
random.seed(SEED)

print_lock = Lock()
csv_lock = Lock()

# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------
def rms(signal):
    return np.sqrt(np.mean(signal**2))

def set_snr(target, noise, snr_db):
    rms_t = rms(target)
    rms_n = rms(noise)
    if rms_t == 0 or rms_n == 0:
        return target
    target_rms_new = rms_n * 10 ** (snr_db / 20)
    scale = target_rms_new / rms_t
    return target * scale

def pad_or_trim(y, target_len):
    if len(y) > target_len:
        return y[:target_len]
    elif len(y) < target_len:
        pad_width = target_len - len(y)
        return np.pad(y, (0, pad_width), mode='constant')
    return y

# ------------------------------------------------------------
# Mixture generation for a single combination
# ------------------------------------------------------------
def create_mixture(split, mix_id, tr_path, bg_path, snr_db, writer):
    try:
        # Output paths
        mix_name = f"mix_{split}_{mix_id:06d}_snr_{snr_db:+d}.wav"
        out_path = MIXES_DIR / split / mix_name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Load audio
        tr_y, _ = librosa.load(tr_path, sr=SAMPLE_RATE, mono=True)
        bg_y, _ = librosa.load(bg_path, sr=SAMPLE_RATE, mono=True)

        # Pad or trim both to same length
        target_len = int(MIX_DURATION_S * SAMPLE_RATE)
        tr_y = pad_or_trim(tr_y, target_len)
        bg_y = pad_or_trim(bg_y, target_len)

        # Mix at target SNR
        tr_scaled = set_snr(tr_y, bg_y, snr_db)
        mix = bg_y + tr_scaled

        # Normalize and write
        mix = mix / np.max(np.abs(mix)) * 0.99
        sf.write(out_path, mix, SAMPLE_RATE)

        # Metadata
        meta = {
            "mix_id": f"mix_{split}_{mix_id:06d}",
            "split": split,
            "background_path": str(bg_path),
            "transient_path": str(tr_path),
            "target_snr_db": snr_db,
            "duration_s": MIX_DURATION_S,
            "sample_rate": SAMPLE_RATE,
            "seed": SEED,
        }

        json_path = ANNOTATIONS_DIR / f"{meta['mix_id']}.json"
        with open(json_path, "w") as jf:
            json.dump(meta, jf, indent=2)

        # Write CSV row safely
        with csv_lock:
            writer.writerow({
                "mix_id": meta["mix_id"],
                "split": split,
                "mix_path": str(out_path),
                "background_id": bg_path.stem,
                "transient_id": tr_path.stem,
                "target_snr_db": snr_db,
                "duration_s": MIX_DURATION_S,
                "sample_rate": SAMPLE_RATE,
                "seed": SEED
            })

        with print_lock:
            print(f"✅ {mix_name} created")

    except Exception as e:
        with print_lock:
            print(f"❌ Error in {mix_name}: {e}")

# ------------------------------------------------------------
# Main mixing logic
# ------------------------------------------------------------
def generate_mixtures():
    background = list((PROCESSED_DIR / "background").rglob("*.wav"))
    foreground = list((PROCESSED_DIR / "foreground").rglob("*.wav"))
    assert background and foreground, "No background or foreground files found."

    print(f"Found {len(background)} backgrounds and {len(foreground)} foregrounds.")
    print(f"Generating mixtures at SNRs: {TARGET_SNRS_DB} dB using {NUM_WORKERS} threads")
    print("Onset mode: fixed overlay (no offset)")

    for split in SPLITS:
        (MIXES_DIR / split).mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Assign foreground files to splits
    num_foreground = len(foreground)
    indices = list(range(num_foreground))
    random.shuffle(indices)
    split_sizes = [int(r * num_foreground) for r in SPLIT_RATIOS]
    split_bounds = np.cumsum(split_sizes).tolist()
    split_assignments = {
        "train": indices[:split_bounds[0]],
        "val": indices[split_bounds[0]:split_bounds[1]],
        "test": indices[split_bounds[1]:],
    }

    manifest_path = ANNOTATIONS_DIR / "mix_manifest.csv"
    with open(manifest_path, "w", newline="") as csvfile:
        fieldnames = [
            "mix_id", "split", "mix_path", "background_id", "transient_id",
            "target_snr_db", "duration_s", "sample_rate", "seed"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Submit mixture jobs
        futures = []
        mix_id = 0
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            for split, idxs in split_assignments.items():
                for i in idxs:
                    tr_path = foreground[i]
                    bg_path = random.choice(background)
                    for snr_db in TARGET_SNRS_DB:
                        mix_id += 1
                        futures.append(executor.submit(
                            create_mixture, split, mix_id, tr_path, bg_path, snr_db, writer
                        ))

            # Wait for all to finish
            for _ in as_completed(futures):
                pass

    print("\n🎯 All mixtures generated successfully!")
    print(f"Manifest saved to: {manifest_path}")

# ------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    generate_mixtures()
