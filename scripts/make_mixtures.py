#!/usr/bin/env python3
"""
make_mixtures.py
----------------
Generate synthetic audio mixtures at controlled SNR levels.

Inputs:
    data/processed/background/*.wav
    data/processed/foreground/*.wav

Outputs:
    data/mixes/{split}/mix_<split>_<id>_snr_<dB>.wav
    data/annotations/mix_<id>.json
    data/annotations/mix_manifest.csv

Each mix = 1 transient + 1 background at a target SNR.
"""

import os
import json
import random
import csv
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
SPLIT_RATIOS = [0.8, 0.1, 0.1]  # proportion of generated mixes
TARGET_SNRS_DB = [-10, -5, 0, 5]
MIX_DURATION_S = 10.0
SAMPLE_RATE = 48000
SEED = 42
random.seed(SEED)

# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------
def rms(signal):
    """Compute RMS of an audio signal."""
    return np.sqrt(np.mean(signal**2))

def set_snr(target, noise, snr_db):
    """
    Scale target signal to achieve desired SNR (in dB) relative to noise.
    Returns scaled target.
    """
    rms_t = rms(target)
    rms_n = rms(noise)
    if rms_t == 0 or rms_n == 0:
        return target
    target_rms_new = rms_n * 10 ** (snr_db / 20)
    scale = target_rms_new / rms_t
    return target * scale

def pad_or_trim(y, target_len):
    """Pad or trim signal to a fixed length."""
    if len(y) > target_len:
        return y[:target_len]
    elif len(y) < target_len:
        pad_width = target_len - len(y)
        return np.pad(y, (0, pad_width), mode='constant')
    return y

def load_random_segment(y, sr, dur_s):
    """Randomly select a segment of duration dur_s from signal y."""
    total_len = len(y)
    seg_len = int(dur_s * sr)
    if total_len <= seg_len:
        return pad_or_trim(y, seg_len)
    start = random.randint(0, total_len - seg_len)
    return y[start:start + seg_len]

# ------------------------------------------------------------
# Main mixing logic
# ------------------------------------------------------------
def generate_mixtures():
    background = list((PROCESSED_DIR / "background").rglob("*.wav"))
    foreground = list((PROCESSED_DIR / "foreground").rglob("*.wav"))
    assert background and foreground, "No background or foreground files found."

    total_combos = len(background) * len(TARGET_SNRS_DB)
    print(f"Found {len(background)} background and {len(foreground)} foreground.")
    print(f"Generating mixtures at SNRs: {TARGET_SNRS_DB} dB")

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

                    bg_seg = load_random_segment(bg_y, SAMPLE_RATE, MIX_DURATION_S)
                    tr_y = pad_or_trim(tr_y, int(MIX_DURATION_S * SAMPLE_RATE))

                    # Choose random onset for transient (avoid edges)
                    onset_s = random.uniform(1.0, MIX_DURATION_S - 1.0)
                    onset_idx = int(onset_s * SAMPLE_RATE)
                    mix = bg_seg.copy()

                    # Mix transient into background
                    mix_segment = np.zeros_like(bg_seg)
                    end_idx = min(onset_idx + len(tr_y), len(bg_seg))
                    mix_segment[onset_idx:end_idx] += tr_y[:end_idx - onset_idx]
                    mix_segment = set_snr(mix_segment, bg_seg, snr_db)
                    mix = bg_seg + mix_segment
                    mix = mix / np.max(np.abs(mix)) * 0.99  # normalize output

                    # Write WAV
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
# Run script
# ------------------------------------------------------------
if __name__ == "__main__":
    generate_mixtures()
