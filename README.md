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

- [x] **Preprocess raw audio**  
  Normalize and resample clips for consistent analysis.

- [x] **Generate synthetic mixes**  
  Create mixtures at SNR levels: **−10 dB, −5 dB, 0 dB, +5 dB**.

- [ ] **Implement baseline enhancement methods**
  - STFT spectral gating  
  - Median filtering in the time–frequency domain  
  - Wavelet denoising

---

## Future Work
- Evaluate enhancement quality using SNR and SDR metrics.  
- Compare classical and deep learning approaches (e.g., CRNN or U-Net models).  
- Extend dataset with additional environmental recordings.

---

## Status
*Enhancement implementation in progress.*

## Notes
### Spectral Gates
Let $X(f,t)$ be the STFT of the noisy signal, $N(f)$ be the estimaed noise spectrum, and $T(f) = \alpha N(f)$ be the threshold.

A simple spectral gate is:
$$\hat{X}(f,t)=\begin{cases}X(f,t), \;\;\;\; |{X(f,t)}|>T(f)\\\beta X(f,t), \;\; |{X(f,t)}|\leq T(f)\end{cases}$$
where $ \alpha$ is a multiplier for sensitivity and $\beta$ is the attenuation factor.

