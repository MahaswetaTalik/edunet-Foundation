import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model

# Page config and background styling
st.set_page_config(page_title="Climate Tipping Point Predictor", layout="wide")
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://as2.ftcdn.net/v2/jpg/05/52/22/47/1000_F_552224737_ZaakNnXCzGaE6Zt7WMVaxJsAv1sI4UPj.jpg");
        background-size: cover;
        background-attachment: fixed;
        position: relative;
    }
    .stApp::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background-color: rgba(255, 255, 255, 0.6);
        z-index: -1;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🌍 Environmental Impact Predictor")
st.markdown("Enter environmental parameters to predict **Temperature**, **CO₂ Emissions**, and **Sea Level Rise**.")

# Sidebar inputs
with st.sidebar:
    st.header("🔧 Input Parameters")
    debug_mode = st.checkbox("Enable Debug Mode")
    humidity = st.number_input("Humidity", value=0.0)
    wind_speed = st.number_input("Wind Speed", value=0.0)
    solar_radiation = st.number_input("Solar Radiation", value=0.0)
    industrial_output = st.number_input("Industrial Output", value=0.0)
    vehicle_count = st.number_input("Vehicle Count", value=0.0)
    global_temp = st.number_input("Global Temperature", value=0.0)

    input_data = np.array([
        humidity, wind_speed, solar_radiation,
        industrial_output, vehicle_count, global_temp
    ]).reshape(1, -1)

# Load models safely
try:
    multi_model = joblib.load('multi_model.pkl')
    multi_scaler = joblib.load('multi_scaler.pkl')
    dl_model = load_model('ai_sea_level_model.keras')
    dl_scaler = joblib.load('dl_scaler.pkl')
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# Predict button
if st.button("Predict"):
    try:
        scaled_ml = multi_scaler.transform(input_data)
        ml_preds = multi_model.predict(scaled_ml)
    except Exception as e:
        st.error(f"❌ ML prediction failed: {e}")
        if debug_mode: st.exception(e)
        st.stop()

    try:
        scaled_dl = dl_scaler.transform(input_data)
        dl_pred = dl_model.predict(scaled_dl)
    except Exception as e:
        st.error(f"❌ DL prediction failed: {e}")
        if debug_mode: st.exception(e)
        st.stop()

    # Display predictions
    st.subheader("📈 Predicted Values")
    col1, col2, col3 = st.columns(3)
    col1.metric("🌡️ Temperature", f"{ml_preds[0][0]:.2f}")
    col2.metric("🫁 CO₂ Emissions", f"{ml_preds[0][1]:.2f}")
    col3.metric("🌊 Sea Level Rise (DL)", f"{dl_pred[0][0]:.2f}")

    pred_df = pd.DataFrame({
        'Target': ['Temperature', 'CO₂ Emissions', 'Sea Level Rise'],
        'Predicted Value': [ml_preds[0][0], ml_preds[0][1], dl_pred[0][0]]
    })

    # Charts (wrapped for safety)
    try:
        st.subheader("📊 Prediction Overview")
        fig1, ax1 = plt.subplots()
        ax1.barh(pred_df['Target'], pred_df['Predicted Value'], color=['#FF6F61', '#6B5B95', '#88B04B'])
        ax1.set_xlabel("Predicted Value")
        ax1.set_title("Environmental Predictions Overview")
        st.pyplot(fig1)
    except Exception as e:
        st.error("❌ Bar chart failed.")
        if debug_mode: st.exception(e)

    try:
        st.subheader("🧭 Prediction Spread (Radar View)")
        labels = pred_df['Target'].tolist()
        values = pred_df['Predicted Value'].tolist()
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        fig2, ax2 = plt.subplots(subplot_kw={'polar': True})
        ax2.plot(angles, values, color='teal', linewidth=2)
        ax2.fill(angles, values, color='teal', alpha=0.25)
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(labels)
        ax2.set_title("Prediction Spread", y=1.1)
        st.pyplot(fig2)
    except Exception as e:
        st.error("❌ Radar chart failed.")
        if debug_mode: st.exception(e)

    try:
        st.subheader("🌊 Sea Level Gauge")
        fig3, ax3 = plt.subplots(figsize=(4, 2))
        ax3.barh(['Sea Level'], [dl_pred[0][0]], color='dodgerblue')
        ax3.set_xlim(0, 100)
        ax3.set_title("Sea Level Rise Indicator")
        ax3.set_xlabel("Rise (cm)")
        st.pyplot(fig3)
    except Exception as e:
        st.error("❌ Gauge chart failed.")
        if debug_mode: st.exception(e)

    try:
        st.subheader("🧮 Prediction Proportions")
        fig4, ax4 = plt.subplots()
        ax4.pie(pred_df['Predicted Value'], labels=pred_df['Target'], autopct='%1.1f%%',
                colors=['#FF6F61', '#6B5B95', '#88B04B'], startangle=90)
        ax4.set_title("Proportional Impact")
        st.pyplot(fig4)
    except Exception as e:
        st.error("❌ Pie chart failed.")
        if debug_mode: st.exception(e)

    try:
        baseline = [25, 300, 20]
        delta_df = pd.DataFrame({
            'Target': ['Temperature', 'CO₂ Emissions', 'Sea Level Rise'],
            'Predicted': [ml_preds[0][0], ml_preds[0][1], dl_pred[0][0]],
            'Baseline': baseline
        })
        delta_df['Delta'] = delta_df['Predicted'] - delta_df['Baseline']
        st.subheader("📉 Change from Baseline")
        st.dataframe(delta_df[['Target', 'Predicted', 'Baseline', 'Delta']])
    except Exception as e:
        st.error("❌ Delta chart failed.")
        if debug_mode: st.exception(e)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Mahasweta | [GitHub](https://github.com/MahaswetaTalik/edunet-Foundation.git)")