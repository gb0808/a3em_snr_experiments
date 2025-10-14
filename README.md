# SNR Enhancement Project
**Author:** Gabriel H. Barnard  

---

## Overview
This project explores methods for **increasing the signal-to-noise ratio (SNR)** of **transient sounds** when mixed with **pervasive or invariant background noise** — such as rainfall, running water, crickets, breeze through trees, and frogs.

The ultimate goal is to design and evaluate algorithms capable of isolating or enhancing short-duration transients under natural, persistent environmental conditions.

---

## Project Phases

1. **Transient Detection**  
   Identify the presence and timing of transient events in noisy recordings.

2. **Transient Enhancement / Extraction**  
   Suppress background noise and recover a cleaner transient signal.

---

## Dataset
All audio data is sourced from the [**ESC-50 dataset**](https://github.com/karolpiczak/ESC-50), which contains environmental sound recordings from 50 categories.  
Synthetic mixtures will be generated at controlled SNR levels to enable quantitative evaluation.

---

## Planned Tasks

- [ ] **Preprocess raw audio**  
  Normalize and resample clips for consistent analysis.

- [ ] **Generate synthetic mixes**  
  Create mixtures at SNR levels: **−10 dB, −5 dB, 0 dB, +5 dB**.

- [ ] **Implement baseline enhancement methods**
  - STFT spectral gating  
  - Median filtering in the time–frequency domain  
  - Wavelet denoising

---

## Repository Structure (planned)
```
snr-enhancement/
├─ data/
│  ├─ raw/             # Original ESC-50 audio
│  ├─ processed/       # Normalized/resampled clips
│  └─ mixes/           # Synthetic mixtures at target SNRs
├─ code/
│  ├─ preprocessing/
│  ├─ enhancement/
│  └─ evaluation/
├─ results/
│  ├─ metrics/
│  └─ visualizations/
└─ README.md
```


---

## Future Work
- Evaluate enhancement quality using SNR and SDR metrics.  
- Compare classical and deep learning approaches (e.g., CRNN or U-Net models).  
- Extend dataset with additional environmental recordings.

---

## Status
*Project initialization phase — data preprocessing in progress.*

