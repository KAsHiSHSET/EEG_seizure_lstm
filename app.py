
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("seizure_model.h5")

st.title("🧠 EEG Seizure Detection App")
st.write("Upload EEG data (178 features per row) to detect seizures using an LSTM model.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    if data.shape[1] != 178:
        st.error("❌ The file must have exactly 178 features per row.")
    else:
        X_input = data.values.reshape(data.shape[0], data.shape[1], 1)
        predictions = model.predict(X_input)
        predicted_classes = (predictions > 0.5).astype(int)
        data['Seizure_Predicted'] = predicted_classes
        st.success("✅ Prediction Complete!")
        st.dataframe(data[['Seizure_Predicted']])
        st.download_button("📥 Download Predictions", data.to_csv(index=False), "predictions.csv")
