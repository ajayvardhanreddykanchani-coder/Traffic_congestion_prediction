import streamlit as st
import pandas as pd
import numpy as np
import joblib
import glob
import os

st.set_page_config(
    page_title="Traffic Congestion Prediction",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 Traffic Congestion Prediction System")
st.write("Machine Learning Based Traffic Analysis Dashboard")

st.divider()

# Load processed dataset
data_path = "processed_traffic.csv"

if not os.path.exists(data_path):
    st.error("Processed dataset not found. Please add data/processed_traffic.csv.")
    st.stop()

data = pd.read_csv(data_path)

# Find saved models
model_files = glob.glob("models/*.pkl")

if not model_files:
    st.error("No trained models found in the models folder.")
    st.info("Run your model-training module and save the trained models first.")
    st.stop()

model_names = [os.path.basename(f) for f in model_files]

st.sidebar.header("Dashboard Menu")
selected_model = st.sidebar.selectbox(
    "Choose Machine Learning Model",
    model_names
)

model_path = os.path.join("models", selected_model)
saved = joblib.load(model_path)

model = saved["model"]
features = saved["features"]

st.subheader("📊 Dataset Overview")

c1, c2, c3 = st.columns(3)

c1.metric("Total Records", f"{len(data):,}")
c2.metric("Input Features", len(features))
c3.metric("Available Models", len(model_files))

st.subheader("🛣️ Traffic Prediction")

st.write("Enter the traffic conditions below.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        hour = st.slider("Hour of Day", 0, 23, 8)
        day = st.slider("Day of Month", 1, 31, 15)
        month = st.selectbox("Month", list(range(1, 13)), index=5)
        year = st.number_input("Year", 2012, 2035, 2018)

    with col2:
        temp = st.number_input("Temperature (K)", value=290.0)
        rain = st.number_input("Rain in Last Hour", min_value=0.0, value=0.0)
        snow = st.number_input("Snow in Last Hour", min_value=0.0, value=0.0)
        clouds = st.slider("Cloud Coverage (%)", 0, 100, 40)

    weather_cols = [
        f.replace("weather_main_", "")
        for f in features
        if f.startswith("weather_main_")
    ]

    weather = st.selectbox(
        "Weather",
        weather_cols if weather_cols else ["Not specified"]
    )

    predict_button = st.form_submit_button("🚦 Predict Traffic")

if predict_button:
    # Start with typical values from the dataset
    row = {}

    for feature in features:
        if feature in data.columns and pd.api.types.is_numeric_dtype(data[feature]):
            row[feature] = float(data[feature].median())
        else:
            row[feature] = 0.0

    # Set user-entered conditions
    values = {
        "hour": hour,
        "day": day,
        "month": month,
        "year": year,
        "temp": temp,
        "rain_1h": rain,
        "snow_1h": snow,
        "clouds_all": clouds
    }

    for feature, value in values.items():
        if feature in row:
            row[feature] = value

    # Reset weather indicators
    for feature in features:
        if feature.startswith("weather_main_"):
            row[feature] = 1.0 if feature == f"weather_main_{weather}" else 0.0

    input_data = pd.DataFrame([row], columns=features)
    prediction = float(model.predict(input_data)[0])
    prediction = max(0, prediction)

    if prediction < 1000:
        level = "Low"
    elif prediction < 3000:
        level = "Moderate"
    else:
        level = "High"

    st.divider()
    st.subheader("✅ Prediction Results")

    a, b = st.columns(2)
    a.metric("Predicted Traffic Volume", f"{prediction:,.0f}")
    b.metric("Congestion Level", level)

    st.info(
        "Congestion categories are simplified demo thresholds, "
        "not official traffic classifications."
    )

st.divider()

st.subheader("📈 Model Performance")

metrics_file = "model_metrics.csv"

if os.path.exists(metrics_file):
    metrics = pd.read_csv(metrics_file)
    st.dataframe(metrics, use_container_width=True)

st.subheader("🔍 Dataset Preview")
st.dataframe(data.head(10), use_container_width=True)

st.caption("Traffic Congestion Prediction | Machine Learning Project")
