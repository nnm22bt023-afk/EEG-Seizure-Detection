# 🧠 EEG Seizure Detection System

## Overview
This project detects seizure activity from EEG signals using signal processing techniques.

## Features
- EEG (.edf) file upload
- Signal visualization
- Frequency analysis (FFT)
- Band power calculation
- Feature extraction
- Seizure detection
- Clinical interpretation module

## Methodology
- Bandpass filtering (0.5–40 Hz)
- Sliding window analysis
- Feature extraction (mean, std, energy, peak)
- Threshold-based classification

## Output
- Seizure / Normal detection
- Confidence score
- Clinical recommendation

## Technologies
- Python
- Streamlit
- NumPy
- SciPy
- MNE
- Matplotlib

## Note
This is a research/demo system and not for clinical diagnosis.
