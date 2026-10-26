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

# ============================================================
# DATASET PATH
# Your GitHub screenshot shows processed_traffic.csv in the
# MAIN REPOSITORY FOLDER, not inside a data/ folder.
# ============================================================
data_path = "processed_traffic.csv"  # <-- DATASET PATH

if not os.path.exists(data_path):
    st.error(
        "Processed dataset not found. Upload processed_traffic.csv "
        "to the same folder as app.py in your GitHub repository."
    )
    st.stop()

data = pd.read_csv(data_path)

# ============================================================
# MODEL PATH
# Your GitHub screenshot shows .pkl files in the MAIN REPOSITORY
# FOLDER, not inside models/. Search there first, then models/.
# ============================================================
model_files = glob.glob("*.pkl") + glob.glob("models/*.pkl")  # <-- MODEL SEARCH PATH
model_files = sorted(set(model_files))

if not model_files:
    st.error(
        "No trained .pkl models found. Upload a model file such as "
        "random_forest_compressed.pkl beside app.py, or inside models/."
    )
    st.stop()

# Display short names in the dropdown while keeping full paths for loading
model_names = {os.path.basename(path): path for path in model_files}

st.sidebar.header("Dashboard Menu")
selected_model_name = st.sidebar.selectbox(
    "Choose Machine Learning Model",
    list(model_names.keys())
)

model_path = model_names[selected_model_name]  # <-- EXACT SELECTED MODEL PATH
saved = joblib.load(model_path)

# Support either the expected dictionary format or a directly saved model
if isinstance(saved, dict) and "model" in saved and "features" in saved:
    model = saved["model"]
    features = saved["features"]
else:
    model = saved
    st.error(
        "The selected model file does not contain the expected "
        "{'model': ..., 'features': ...} dictionary. Please save it "
        "in that format from the training notebook."
    )
    st.stop()

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
        feature.replace("weather_main_", "")
        for feature in features
        if feature.startswith("weather_main_")
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

    # Set user-entered conditions only for features the model expects
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

    # Set one-hot encoded weather indicators
    for feature in features:
        if feature.startswith("weather_main_"):
            row[feature] = 1.0 if feature == f"weather_main_{weather}" else 0.0

    input_data = pd.DataFrame([row], columns=features)

    try:
        prediction = float(model.predict(input_data)[0])
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

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
