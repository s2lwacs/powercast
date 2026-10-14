import streamlit as st
import pandas as pd
from predict import predict_next_24h, make_future_features

# ---------- Page setup ----------
st.set_page_config(page_title="PowerCast", page_icon="⚡", layout="wide")


# ---------- Load data (once) ----------
@st.cache_data
def load_data():
    df = pd.read_csv("data/hourly_clean.csv", parse_dates=["datetime"])
    return df


df = load_data()
series = df.set_index("datetime")["Global_active_power"]
test = df[df["split"] == "test"]

first_day = test["datetime"].min().date()
last_day = test["datetime"].max().date() - pd.Timedelta(days=1)

TEST_DAY = "📅 A test day (forecast vs actual)"
NEXT_24H = "🔮 Next 24 hours (after our data ends)"


# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.radio("What do you want to see?", [TEST_DAY, NEXT_24H])
    if mode == TEST_DAY:
        chosen = st.date_input(
            "Pick a day", value=pd.Timestamp("2010-11-19").date(),
            min_value=first_day, max_value=last_day
        )
        st.caption("Only test days are available (days the model never saw).")
    st.divider()
    st.markdown("**Model:** LSTM V4")
    st.markdown("**Input:** last 168 hours (7 days) + calendar of the next 24 hours")
    st.markdown("**Output:** next 24 hours")


# ---------- Header ----------
st.title("⚡ PowerCast")
st.markdown("#### Forecasting a household's electricity use for the next 24 hours")
st.divider()


# ======================= Mode 1: a test day =======================
if mode == TEST_DAY:
    start = pd.Timestamp(chosen)
    history = series[start - pd.Timedelta(hours=168): start - pd.Timedelta(hours=1)]
    actual = series[start: start + pd.Timedelta(hours=23)]
    forecast = predict_next_24h(history)   # pandas Series with dates -> calendar is built automatically

    result = pd.DataFrame(
        {"Actual": actual.values, "Forecast": forecast}, index=actual.index.hour
    )
    result.index.name = "Hour"

    peak = result["Forecast"].idxmax()
    actual_peak = result["Actual"].idxmax()
    mae = (result["Actual"] - result["Forecast"]).abs().mean()
    total = result["Forecast"].sum()

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("⚡ Predicted peak", f"{peak:02d}:00")
    c2.metric("✅ Actual peak", f"{actual_peak:02d}:00")
    c3.metric("📏 Avg error (MAE)", f"{mae:.2f} kW")
    c4.metric("🔋 Predicted total", f"{total:.1f} kWh")

    # Chart
    st.subheader(f"📈 Actual vs Forecast — {chosen.strftime('%A, %d %B %Y')}")
    st.line_chart(
        result,
        color=["#F5A623", "#4A90E2"],
        x_label="Hour of day",
        y_label="Power (kW)",
        height=420,
    )

    if peak == actual_peak:
        st.success(f"🎯 The model caught the peak hour correctly: {peak:02d}:00")
    elif abs(peak - actual_peak) <= 1:
        st.info(f"👍 Close: predicted peak {peak:02d}:00, real peak {actual_peak:02d}:00 (within 1 hour)")
    else:
        st.warning(
            f"Predicted peak at {peak:02d}:00, but the real peak was at {actual_peak:02d}:00"
        )

    with st.expander("📋 See hourly numbers"):
        st.dataframe(result.round(2), width="stretch")


# =================== Mode 2: the next 24 hours ===================
else:
    last_hour = series.index[-1]
    start = last_hour + pd.Timedelta(hours=1)
    history = series.iloc[-168:]                      # the last 7 days we really know
    future = make_future_features(start)              # 24 rows: calendar only, no consumption
    future["Forecast (kW)"] = predict_next_24h(history, start)

    peak_time = future["Forecast (kW)"].idxmax()

    st.info(
        f"Our data ends on **{last_hour:%A %d %B %Y, %H:%M}**. "
        f"This is a real forecast for hours the model has never seen: "
        f"**{start:%a %d %b %H:%M} → {start + pd.Timedelta(hours=23):%a %d %b %H:%M}**."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("⚡ Expected peak", f"{peak_time:%a %H:00}")
    c2.metric("📈 Peak power", f"{future['Forecast (kW)'].max():.2f} kW")
    c3.metric("🔋 Expected total", f"{future['Forecast (kW)'].sum():.1f} kWh")

    st.subheader("🔮 Forecast for the next 24 hours")
    chart = pd.concat([
        series.iloc[-48:].rename("Actual (last 2 days)"),
        future["Forecast (kW)"].rename("Forecast"),
    ], axis=1)
    st.line_chart(
        chart,
        color=["#F5A623", "#4A90E2"],
        x_label="Time",
        y_label="Power (kW)",
        height=420,
    )

    with st.expander("📋 What the model receives for the next 24 hours (no consumption values)"):
        st.dataframe(future.round(3), width="stretch")


# ---------- How it works ----------
with st.expander("❓ How does it work?"):
    st.markdown(
        """
1. We take the **168 hours (7 days)** before the forecast.
2. We add the **calendar of the next 24 hours**: weekend or not, and the hour of the day (as sine/cosine). The calendar is known in advance, so there is no leakage.
3. The **LSTM V4** model predicts the next **24 hours** and we find the **peak hour**.
4. On test days, we compare the forecast with what really happened.
"""
    )
