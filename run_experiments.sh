source venv/bin/activate
python scripts/normalize_audio.py
python scripts/make_mixtures.py
python scripts/spectral_gate.py
python scripts/median_filter.py
python scripts/evaluate_snr.py
deactivate