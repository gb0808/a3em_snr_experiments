import os
import numpy as np
import librosa
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ============================================================
# 🔧 CONFIG
# ============================================================
TEST_DIR = "data/mixes/train"
RESULT_DIR = "results/spectral-gates"

os.makedirs(RESULT_DIR, exist_ok=True)

# ============================================================
# 🧠 Spectral Gate Function
# ============================================================

def spectral_gate(y, sr, noise_reduce_db=15, threshold=0.5):
    """
    Simple spectral gating denoiser.

    Args:
        y (np.ndarray): input signal
        sr (int): sample rate
        noise_reduce_db (float): unused placeholder (for extension)
        threshold (float): gating threshold factor

    Returns:
        np.ndarray: denoised signal
    """
    S = librosa.stft(y)
    magnitude, phase = librosa.magphase(S)

    # Estimate noise profile from 25th percentile of magnitudes
    noise_profile = np.percentile(magnitude, 25, axis=1, keepdims=True)

    # Create mask
    mask = magnitude > (noise_profile * threshold)

    # Apply mask
    magnitude_denoised = magnitude * mask

    # Reconstruct signal, preserving original length
    y_denoised = librosa.istft(magnitude_denoised * phase, length=len(y))

    return y_denoised


# ============================================================
# 🧩 File Processing Function
# ============================================================

def process_file(fname):
    """Apply spectral gating and save result."""
    try:
        in_path = os.path.join(TEST_DIR, fname)
        out_path = os.path.join(RESULT_DIR, fname)

        y, sr = librosa.load(in_path, sr=None)
        y_denoised = spectral_gate(y, sr)

        sf.write(out_path, y_denoised, sr)
        return f"✅ {fname}"
    except Exception as e:
        return f"❌ {fname}: {e}"


# ============================================================
# 🚀 Main Execution
# ============================================================

if __name__ == "__main__":
    files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(".wav")]

    print(f"Applying spectral gating to {len(files)} files...")
    os.makedirs(RESULT_DIR, exist_ok=True)

    with ProcessPoolExecutor() as ex:
        for msg in tqdm(ex.map(process_file, files), total=len(files)):
            print(msg)

    print(f"\n🎯 All denoised files saved to: {RESULT_DIR}")
