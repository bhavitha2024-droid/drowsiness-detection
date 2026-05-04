from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Driver Drowsiness Dashboard", layout="wide")
st.title("Personalized Temporal Driver Drowsiness Dashboard")

csv_path = Path("logs/session_features.csv")
if not csv_path.exists():
    st.warning("No session log found yet. Run app.py first.")
    st.stop()

df = pd.read_csv(csv_path)
if df.empty:
    st.warning("The session log is empty.")
    st.stop()

latest = df.iloc[-1]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Severity", str(latest["severity"]).upper())
col2.metric("EAR", f"{latest['ear']:.3f}")
col3.metric("MAR", f"{latest['mar']:.3f}")
col4.metric("FPS Proxy", "Logged stream")

st.subheader("Temporal Signals")
st.line_chart(df[["ear", "mar", "severity_score"]])

st.subheader("Pose Trends")
st.line_chart(df[["pitch", "roll", "yaw"]])

st.subheader("Event Counts")
st.line_chart(df[["blink_count", "yawn_count", "eye_closed_frames"]])

st.subheader("Recent Samples")
st.dataframe(df.tail(50), use_container_width=True)
