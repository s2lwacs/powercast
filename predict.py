
import numpy as np

def predict_next_24h(last_168_hours):
    history = np.asarray(last_168_hours, dtype=float).reshape(-1)

    if len(history) != 168:
        raise ValueError("Expected 168 hourly consumption values.")

    if not np.isfinite(history).all():
        raise ValueError("Input contains missing or invalid values.")

    # Weekly Naive: same 24 hours one week earlier (hours t-168 ... t-145)
    return history[:24].copy()
