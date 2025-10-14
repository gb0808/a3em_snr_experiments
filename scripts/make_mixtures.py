"""
make_mixtures_fixed.py
----------------------
Generate 5-second synthetic mixtures of transients + backgrounds
at controlled SNRs. Supports optional random onset timing.

Usage:
    python make_mixtures_fixed.py [--random-onset]

Inputs:
    data/processed/background/*.wav
    data/processed/foreground/*.wav

Outputs:
    data/mixes/{split}/mix_<split>_<id>_snr_<dB>.wav
    data/annotations/*.json
    data/annotations/mix_manifest.csv
"""

import os
import json
import random
import csv
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa

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
random.seed(SEED)

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
# Main mixing logic
# ------------------------------------------------------------
def generate_mixtures(random_onset=False):
    background = list((PROCESSED_DIR / "background").rglob("*.wav"))
    foreground = list((PROCESSED_DIR / "foreground").rglob("*.wav"))
    assert background and foreground, "No background or foreground files found."

    total_combos = len(foreground) * len(TARGET_SNRS_DB)
    print(f"Found {len(background)} backgrounds and {len(foreground)} foregrounds.")
    print(f"Generating mixtures at SNRs: {TARGET_SNRS_DB} dB")
    print(f"Onset mode: {'randomized' if random_onset else 'fixed (2.0 s)'}")

    # Create output directories
    for split in SPLITS:
        (MIXES_DIR / split).mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Split data indices
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
            "target_snr_db", "onset_s", "duration_s", "sample_rate", "seed"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        mix_id = 0
        for split, idxs in split_assignments.items():
            for i in idxs:
                tr_path = foreground[i]
                bg_path = random.choice(background)
                for snr_db in TARGET_SNRS_DB:
                    mix_id += 1
                    mix_name = f"mix_{split}_{mix_id:06d}_snr_{snr_db:+d}.wav"
                    out_path = MIXES_DIR / split / mix_name

                    # Load audio
                    tr_y, _ = librosa.load(tr_path, sr=SAMPLE_RATE, mono=True)
                    bg_y, _ = librosa.load(bg_path, sr=SAMPLE_RATE, mono=True)

                    # Ensure both 5 s long
                    target_len = int(MIX_DURATION_S * SAMPLE_RATE)
                    tr_y = pad_or_trim(tr_y, target_len)
                    bg_y = pad_or_trim(bg_y, target_len)

                    # Choose onset
                    if random_onset:
                        onset_s = random.uniform(0.5, 4.0)
                    else:
                        onset_s = 2.0
                    onset_idx = int(onset_s * SAMPLE_RATE)

                    # Overlay transient
                    mix = bg_y.copy()
                    end_idx = min(onset_idx + len(tr_y), len(bg_y))
                    tr_seg = tr_y[: end_idx - onset_idx]
                    bg_seg = bg_y[onset_idx:end_idx]
                    tr_scaled = set_snr(tr_seg, bg_seg, snr_db)
                    mix[onset_idx:end_idx] += tr_scaled

                    # Normalize output
                    mix = mix / np.max(np.abs(mix)) * 0.99
                    sf.write(out_path, mix, SAMPLE_RATE)

                    # Write metadata JSON
                    meta = {
                        "mix_id": f"mix_{split}_{mix_id:06d}",
                        "split": split,
                        "background_path": str(bg_path),
                        "transient_path": str(tr_path),
                        "target_snr_db": snr_db,
                        "onset_s": onset_s,
                        "duration_s": MIX_DURATION_S,
                        "sample_rate": SAMPLE_RATE,
                        "seed": SEED,
                    }
                    json_path = ANNOTATIONS_DIR / f"{meta['mix_id']}.json"
                    with open(json_path, "w") as jf:
                        json.dump(meta, jf, indent=2)

                    writer.writerow({
                        "mix_id": meta["mix_id"],
                        "split": split,
                        "mix_path": str(out_path),
                        "background_id": bg_path.stem,
                        "transient_id": tr_path.stem,
                        "target_snr_db": snr_db,
                        "onset_s": onset_s,
                        "duration_s": MIX_DURATION_S,
                        "sample_rate": SAMPLE_RATE,
                        "seed": SEED
                    })
                    print(f"✅ {mix_name} created")

    print(f"\n🎯 All mixtures generated successfully!")
    print(f"Manifest saved to: {manifest_path}")

# ------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic mixtures.")
    parser.add_argument("--random-onset", action="store_true",
                        help="Randomize transient onset times (default: fixed 2.0 s).")
    args = parser.parse_args()

    generate_mixtures(random_onset=args.random_onset)

