import streamlit as st
import pandas as pd
from predict import predict_next_24h

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


# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")
    chosen = st.date_input(
        "Pick a day", value=first_day, min_value=first_day, max_value=last_day
    )
    st.caption("Only test days are available (days the model never saw).")
    st.divider()
    st.markdown("**Model:** Weekly Naive")
    st.markdown("**Input:** last 168 hours (7 days)")
    st.markdown("**Output:** next 24 hours")


# ---------- Header ----------
st.title("⚡ PowerCast")
st.markdown("#### Forecasting a household's electricity use for the next 24 hours")
st.divider()


# ---------- Forecast ----------
start = pd.Timestamp(chosen)
history = series[start - pd.Timedelta(hours=168) : start - pd.Timedelta(hours=1)]
actual = series[start : start + pd.Timedelta(hours=23)]
forecast = predict_next_24h(history.values)

result = pd.DataFrame(
    {"Actual": actual.values, "Forecast": forecast}, index=actual.index.hour
)
result.index.name = "Hour"

peak = result["Forecast"].idxmax()
actual_peak = result["Actual"].idxmax()
mae = (result["Actual"] - result["Forecast"]).abs().mean()
total = result["Forecast"].sum()


# ---------- KPI cards ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("⚡ Predicted peak", f"{peak:02d}:00")
c2.metric("✅ Actual peak", f"{actual_peak:02d}:00")
c3.metric("📏 Avg error (MAE)", f"{mae:.2f} kW")
c4.metric("🔋 Predicted total", f"{total:.1f} kWh")


# ---------- Chart ----------
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
else:
    st.warning(
        f"Predicted peak at {peak:02d}:00, but the real peak was at {actual_peak:02d}:00"
    )


# ---------- Details ----------
with st.expander("📋 See hourly numbers"):
    st.dataframe(result.round(2), width="stretch")

with st.expander("❓ How does it work?"):
    st.markdown(
        """
1. You pick a day.
2. We take the **168 hours (7 days)** before it.
3. The model predicts the next **24 hours**. Weekly Naive says: *tomorrow looks like the same day last week*.
4. We compare the forecast with what really happened and find the **peak hour**.
"""
    )
