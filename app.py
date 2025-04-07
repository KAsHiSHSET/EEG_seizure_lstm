import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
@st.cache_resource
def load_trained_model():
    return load_model("seizure_model (1).h5")

model = load_trained_model()

# App Title
st.title("🧠 EEG Seizure Detection App")
st.write("Upload EEG data (178 features per row) to detect seizures using an LSTM model.")

# File Upload
uploaded_file = st.file_uploader("📁 Upload your EEG CSV file", type="csv")

# Prediction Logic
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        if data.shape[1] != 178:
            st.error("❌ The file must have exactly 178 features (columns) per row.")
        else:
            # Reshape for LSTM input
            X_input = data.values.reshape(data.shape[0], data.shape[1], 1)

            # Predict
            predictions = model.predict(X_input)
            predicted_classes = (predictions > 0.5).astype(int)

            # Add predictions to DataFrame
            data['Seizure_Predicted'] = predicted_classes

            # Display and download
            st.success("✅ Prediction Complete!")
            st.dataframe(data[['Seizure_Predicted']])
            st.download_button("📥 Download Predictions", data.to_csv(index=False), "predictions.csv")

    except Exception as e:
        st.error(f"⚠️ Error reading the file: {e}")

