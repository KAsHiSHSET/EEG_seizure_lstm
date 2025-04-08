import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Initialize dark mode state
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Toggle button
st.sidebar.title("🌓 Theme")
if st.sidebar.button("Toggle Dark Mode"):
    st.session_state.dark_mode = not st.session_state.dark_mode

# Apply theme colors
if st.session_state.dark_mode:
    bg_color = "#121212"
    font_color = "#FFFFFF"
    chart_colors = ["#888", "#FF5C5C"]
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {bg_color};
            color: {font_color};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    bg_color = "#FFFFFF"
    font_color = "#000000"
    chart_colors = ["#6c757d", "#FF4B4B"]

# Load model
@st.cache_resource
def load_trained_model():
    return load_model("seizure_model.h5")  # Ensure this file exists

model = load_trained_model()

# App title and description
st.markdown(f"""
    <h1 style='text-align: center; color: #FF4B4B;'>
        🧠 EEG Seizure Detection App
    </h1>
    <p style='text-align: center; font-size:18px; color: {font_color};'>
        Upload EEG data (178 features per row) to detect seizures using an LSTM model.
    </p>
""", unsafe_allow_html=True)

# Upload section
st.markdown("### 📂 Upload EEG CSV File")
uploaded_file = st.file_uploader("Drag and drop or browse your EEG CSV file", type="csv")

# Prediction logic
if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")

    try:
        data = pd.read_csv(uploaded_file)

        if data.shape[1] != 178:
            st.error(f"❌ Invalid file format. Expected 178 features per row, but got {data.shape[1]}")
        else:
            predictions = model.predict(data)
            predicted_classes = (predictions > 0.5).astype("int32").flatten()

            prediction_df = pd.DataFrame({
                "Seizure_Predicted": predicted_classes
            })

            st.markdown("### 🧾 Prediction Results")
            st.dataframe(prediction_df.style.highlight_max(axis=0, color='lightcoral'), height=300)

            # Plotting summary
            st.markdown("### 📊 Seizure Count Summary")
            counts = prediction_df["Seizure_Predicted"].value_counts().sort_index()
            labels = ["No Seizure", "Seizure"]

            fig, ax = plt.subplots()
            ax.bar(labels, counts, color=chart_colors)
            ax.set_ylabel("Count", color=font_color)
            ax.set_title("Seizure Prediction Summary", color=font_color)
            ax.tick_params(axis='x', colors=font_color)
            ax.tick_params(axis='y', colors=font_color)
            fig.patch.set_facecolor(bg_color)
            ax.set_facecolor(bg_color)
            st.pyplot(fig)

            # Download
            st.download_button("📥 Download Predictions", prediction_df.to_csv(index=False), "seizure_predictions.csv")

    except Exception as e:
        st.error(f"⚠️ Error reading the file: {e}")

# Footer
st.markdown(f"""<hr style="margin-top:50px;">
    <p style="text-align:center; font-size:14px; color: {font_color};">
        Built with ❤️ by <b>Kashish Seth</b> | Powered by LSTM & Streamlit
    </p>
""", unsafe_allow_html=True)
