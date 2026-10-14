# ⚡ PowerCast

**Forecasting a household's electricity consumption for the next 24 hours, hour by hour, and flagging the expected peak hour.**

Le Wagon × Saudi Digital Academy · Data Science Bootcamp #2344 · Project Weeks, October 2026

---

## Results

Our final model, **LSTM V4**, reads the last 7 days of consumption and the calendar of the next 24 hours.

| | Validation (Jan–Jun 2010) | Test (Jul–Nov 2010, unseen) |
|---|---|---|
| **LSTM V4 MAE** | **0.464 kW** | **0.409 kW** |
| Best simple baseline MAE | 0.580 kW (same hour last week) | 0.479 kW (same hour yesterday) |
| Improvement | **−20%** | **−15%** |
| R² | +0.39 | +0.35 (all baselines below 0) |

## Data

[UCI Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption): one home in France, 2,075,259 minute readings (Dec 2006 → Nov 2010).

- Resampled to **34,589 hourly values**; 421 missing hours (1.2%) filled using past data only.
- Split **by date** (no shuffle): train → Dec 2009, validation Jan–Jun 2010, test Jul–Nov 2010.
- Target: `Global_active_power` (kW). `Global_intensity` was removed (leakage).

## Approach

- **Input:** last 168 hours (power + `is_weekend`, `hour_sin`, `hour_cos`) and the calendar of the next 24 hours (known in advance, so no leakage).
- **Output:** the next 24 hourly values in kW, plus the expected peak hour.
- **Baselines:** last value, same hour yesterday, same hour last week.
- **Experiments:** 7 LSTM versions, one change at a time. The final model was chosen on validation only.

| Model | What changed | Val MAE (kW) |
|---|---|---|
| V1 | Past power only, 1,104 daily windows | 0.720 |
| V2 | + past calendar | 0.744 |
| V3 | A training window at every hour (~26,000) | 0.474 |
| **V4 ✓** | **+ calendar of the next 24 hours** | **0.464** |
| V5 | + day of year (sin/cos) | 0.469 |
| V6 | log1p target | 0.470 |
| V7 | + season (one-hot) | 0.474 |

**Architecture (V4):** LSTM(32) on the past 168 h × 4 features → concatenated with the flattened next-24h calendar (24 × 3) → Dense(24). Adam, MSE loss, batch 128, early stopping (patience 3), seed 42 with deterministic ops.

## Repository

| File | Content |
|---|---|
| `00_final_demo.ipynb` | End-to-end final notebook: data → EDA → baselines → LSTM V1–V7 → test → next-24h forecast |
| `01_data_cleaning_rana.ipynb` | Cleaning, hourly resampling, gap filling, date split |
| `02_EDA_salwa.ipynb` | Exploratory data analysis |
| `03_feature_engineering_modeling_renad.ipynb` | Feature engineering, baselines and LSTM experiments |
| `PowerCast_demo_raneem.ipynb` | Demo notebook for the dashboard |
| `predict.py` | `predict_next_24h()`: forecast the next 24 hours with the final model |
| `app.py` | Streamlit dashboard |
| `models/` | Final model, scaler and feature config |
| `data/hourly_clean.csv` | Cleaned hourly data |

## Run the dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app needs `models/` (`powercast_lstm_final.keras`, `powercast_scaler_final.pkl`, `powercast_final_config.json`) and `data/hourly_clean.csv` next to `app.py`.

Two modes:
- **A test day:** forecast vs actual for any day the model never saw, with the predicted and actual peak hour.
- **Next 24 hours:** forecast for the day after the data ends.

## Limitations and next steps

- One household only; no weather or holiday data.
- The target is right-skewed (skewness 1.31), so peak hours are predicted too low.
- **Next:** add weather forecasts and holidays, and give more weight to peak hours.

## Team

| Name | Role |
|---|---|
| Rana Aljumayi (team lead) | Data cleaning & preprocessing, final notebook |
| Salwa Alharbi | Exploratory data analysis |
| Renad Alotaibi | Feature engineering & modeling |
| Raneem Alghamdi | Streamlit dashboard |
