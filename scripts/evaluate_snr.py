import os
import json
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm

ANNOTATIONS_DIR = Path("data/annotations")
MIXES_DIR = Path("data/mixes/train")
RESULT_DIRS = { 
    "spectral-gates": Path("results/spectral-gates"), 
    "median-filter": Path("results/median-filter")
}


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------
def classical_snr(clean, estimate):
    """Compute classical SNR (dB) given clean and estimate signals."""
    eps = 1e-10
    clean = clean[: len(estimate)]
    estimate = estimate[: len(clean)]
    noise = estimate - clean
    p_signal = np.sum(clean ** 2)
    p_noise = np.sum(noise ** 2)
    return 10 * np.log10((p_signal + eps) / (p_noise + eps))


def compute_snr_improvement(clean_path, noisy_path, denoised_path):
    """Load files and compute SNR improvement."""
    clean, sr = librosa.load(clean_path, sr=None)
    noisy, _ = librosa.load(noisy_path, sr=sr)
    denoised, _ = librosa.load(denoised_path, sr=sr)

    # Align lengths to shortest
    L = min(len(clean), len(noisy), len(denoised))
    clean, noisy, denoised = clean[:L], noisy[:L], denoised[:L]

    snr_before = classical_snr(clean, noisy)
    snr_after = classical_snr(clean, denoised)
    improvement = snr_after - snr_before

    return snr_before, snr_after, improvement


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    json_files = sorted(ANNOTATIONS_DIR.glob("mix_train_*.json"))
    if not json_files:
        raise FileNotFoundError("No test annotation JSON files found in data/annotations/")

    print(f"Evaluating {len(json_files)} test mixtures...\n")

    for name, dir in RESULT_DIRS.items():
        dir.mkdir(parents=True, exist_ok=True)
        CSV_PATH = Path(f"results/{name}-snr-improvement.csv")
        results = []
        for json_path in tqdm(json_files):
            with open(json_path) as f:
                meta = json.load(f)

            mix_id = meta["mix_id"]
            snr_db = meta["target_snr_db"]
            clean_path = Path(meta["transient_path"])
            noisy_path = MIXES_DIR / f"{mix_id}_snr_{snr_db:+d}.wav"
            denoised_path = dir / f"{mix_id}_snr_{snr_db:+d}.wav"

            if not (clean_path.exists() and noisy_path.exists() and denoised_path.exists()):
                continue

            snr_before, snr_after, improvement = compute_snr_improvement(
                clean_path, noisy_path, denoised_path
            )

            results.append({
                "mix_id": mix_id,
                "target_snr_db": snr_db,
                "snr_before_db": snr_before,
                "snr_after_db": snr_after,
                "snr_improvement_db": improvement,
                "clean_path": str(clean_path),
                "noisy_path": str(noisy_path),
                "denoised_path": str(denoised_path),
            })

        df = pd.DataFrame(results)
        df.to_csv(CSV_PATH, index=False)
        print(f"\n✅ SNR evaluation complete. Results saved to: {CSV_PATH}")
        print(df.head())


if __name__ == "__main__":
    main()