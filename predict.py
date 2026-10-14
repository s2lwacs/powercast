"""PowerCast: forecast the next 24 hours with the final model.

X (past):   last 168 hours of Global_active_power (kW) + calendar features of those hours
X (future): calendar features of the next 24 hours (known in advance -> no leakage)
Y:          next 24 hours of Global_active_power (kW)
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings('ignore', message='X does not have valid feature names')

_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_DIR, 'models', 'powercast_lstm_final.keras')
_SCALER_PATH = os.path.join(_DIR, 'models', 'powercast_scaler_final.pkl')
_CONFIG_PATH = os.path.join(_DIR, 'models', 'powercast_final_config.json')
_cache = {}


def calendar_features(timestamps):
    """Time features for any list of hours (same formulas as in training)."""
    ts = pd.DatetimeIndex(timestamps)
    return pd.DataFrame({
        'hour': ts.hour,
        'day_of_year': ts.dayofyear,
        'is_weekend': (ts.dayofweek >= 5).astype(int),
        'hour_sin': np.sin(2 * np.pi * ts.hour / 24),
        'hour_cos': np.cos(2 * np.pi * ts.hour / 24),
        'doy_sin': np.sin(2 * np.pi * ts.dayofyear / 365.25),
        'doy_cos': np.cos(2 * np.pi * ts.dayofyear / 365.25),
    }, index=ts)


def make_future_features(forecast_start, horizon=24):
    """Steps 1-3: one row per future hour with calendar features only (no consumption)."""
    hours = pd.date_range(pd.Timestamp(forecast_start), periods=horizon, freq='h')
    return calendar_features(hours)


def _load():
    if not _cache:
        from tensorflow.keras.models import load_model
        _cache['model'] = load_model(_MODEL_PATH, compile=False)
        _cache['scaler'] = joblib.load(_SCALER_PATH)
        with open(_CONFIG_PATH) as f:
            _cache['config'] = json.load(f)
    return _cache['model'], _cache['scaler'], _cache['config']


def predict_next_24h(last_168_hours, forecast_start=None):
    """Forecast the next 24 hours (kW).

    last_168_hours: the 168 hourly values before the forecast (a pandas Series with a
                    datetime index is best: forecast_start is then found automatically).
    forecast_start: timestamp of the first forecast hour (needed for a plain array).
    """
    model, scaler, cfg = _load()
    history, horizon = cfg['history'], cfg['horizon']

    values = np.asarray(last_168_hours, dtype=float).ravel()
    if values.size != history or not np.all(np.isfinite(values)):
        raise ValueError(f'Need exactly {history} finite hourly values, got {values.size}.')
    if forecast_start is None:
        if not isinstance(last_168_hours, pd.Series):
            raise ValueError('Pass a pandas Series with a datetime index, or give forecast_start.')
        forecast_start = last_168_hours.index[-1] + pd.Timedelta(hours=1)
    forecast_start = pd.Timestamp(forecast_start)

    # Past: scaled power + calendar of the 168 past hours
    past_hours = pd.date_range(end=forecast_start - pd.Timedelta(hours=1), periods=history, freq='h')
    power_s = scaler.transform(values.reshape(-1, 1)).ravel()
    past_cal = calendar_features(past_hours)[cfg['past_cols']].to_numpy()
    X = np.column_stack([power_s, past_cal]).astype('float32')[None, ...]

    # Future: calendar of the next 24 hours (if the model uses it)
    if cfg['fut_cols']:
        fut = make_future_features(forecast_start, horizon)[cfg['fut_cols']].to_numpy('float32')[None, ...]
        X = [X, fut]

    pred = model.predict(X, verbose=0).reshape(-1, 1)
    return scaler.inverse_transform(pred).ravel()
