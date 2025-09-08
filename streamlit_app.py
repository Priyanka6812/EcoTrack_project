import os
import pandas as pd
import streamlit as st
import numpy as np

st.set_page_config(page_title="EcoTrack", layout="wide")
st.title("EcoTrack — AI for Climate & Green Planning")

uploaded = st.file_uploader("Upload Climate CSV (or use sample)", type=["csv"])
use_sample = False
if uploaded is None and os.path.exists("climate_data.csv"):
    use_sample = st.button("Use sample dataset (climate_data.csv)")

if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_sample:
    df = pd.read_csv("climate_data.csv")
else:
    st.info("Upload a CSV or click 'Use sample dataset' if available.")
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Basic cleaning (same as ecotrack)
for col in ['Temperature','CO2','Aerosol']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
df[['Temperature','CO2','Aerosol']] = df[['Temperature','CO2','Aerosol']].interpolate().fillna(method='bfill').fillna(method='ffill')

st.subheader("CO₂ Levels")
st.line_chart(df['CO2'])

st.subheader("Temperature Trend")
st.line_chart(df['Temperature'])

st.subheader("Aerosol Index")
st.line_chart(df['Aerosol'])

# Stats & insights
avg_temp = df['Temperature'].mean()
avg_co2 = df['CO2'].mean()
avg_aero = df['Aerosol'].mean()

st.markdown("### AI Insights (automated)")
st.write(f"🌡 **Average Temperature:** {avg_temp:.2f} °C")
st.write(f"🟢 **Average CO₂ Level:** {avg_co2:.2f} ppm")
st.write(f"🔵 **Average Aerosol Index:** {avg_aero:.3f}")

if avg_temp > 30:
    st.warning("⚠ High average temperature → Suggest green cover, reflective roofs, cool pavements.")
else:
    st.success("✔ Temperature within moderate range for sampled period.")

if avg_co2 > 412:
    st.warning("⚠ Elevated CO₂ → Suggest electric transport & congestion charging.")
else:
    st.success("✔ CO₂ within moderate range.")

if avg_aero > 0.15:
    st.warning("⚠ Higher aerosol index → Investigate local pollution sources.")
else:
    st.info("✔ Aerosol index low for sampled period.")

# Option to download summary as text
summary_text = (
    f"Average Temperature: {avg_temp:.2f} °C\n"
    f"Average CO2: {avg_co2:.2f} ppm\n"
    f"Average Aerosol: {avg_aero:.3f}\n"
)
st.download_button("Download summary (text)", summary_text, file_name="ecotrack_summary.txt")
