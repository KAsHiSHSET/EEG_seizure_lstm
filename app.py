import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from datetime import datetime
from fpdf import FPDF
import base64
import tempfile

# Load the trained model
@st.cache_resource
def load_trained_model():
    return load_model("seizure_model (1).h5")

model = load_trained_model()

# App Title and Description
st.markdown("""
    <h1 style='text-align: center; color: #FF4B4B;'>
        🧠 EEG Seizure Detection App
    </h1>
    <p style='text-align: center; font-size:18px;'>
        Upload EEG data (178 features per row) to detect seizures using an LSTM model.
    </p>
""", unsafe_allow_html=True)

# File Upload Section
st.markdown("### 📂 Upload EEG CSV File")
uploaded_file = st.file_uploader("Drag and drop or browse your EEG CSV file", type="csv")

# Prediction Logic
if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")

    try:
        data = pd.read_csv(uploaded_file)

        if data.shape[1] != 178:
            st.error(f"❌ Invalid file format. Expected 178 features per row, but got {data.shape[1]}")
        else:
            # Make predictions
            predictions = model.predict(data)
            predicted_classes = (predictions > 0.5).astype("int32").flatten()

            # Display results
            prediction_df = pd.DataFrame({
                "Seizure_Predicted": predicted_classes
            })

            st.markdown("### 🧾 Prediction Results")
            st.dataframe(prediction_df.style.highlight_max(axis=0, color='lightcoral'), height=300)

            # Bar Chart Summary
            st.markdown("### 📊 Seizure Count Summary")
            counts = prediction_df["Seizure_Predicted"].value_counts().sort_index()
            labels = ["No Seizure", "Seizure"]

            fig, ax = plt.subplots()
            ax.bar(labels, counts, color=["#6c757d", "#FF4B4B"])
            ax.set_ylabel("Count")
            ax.set_title("Seizure Prediction Summary")
            st.pyplot(fig)

            # PDF Report Generator
            st.markdown("### 📝 Generate Patient Report (PDF)")
            patient_id = st.text_input("Enter Patient ID")
            doctor_note = st.text_area("Doctor's Note")
            if st.button("📄 Generate PDF Report"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.set_text_color(33, 33, 33)

                pdf.cell(200, 10, txt="EEG Seizure Detection Report", ln=1, align='C')
                pdf.cell(200, 10, txt=f"Patient ID: {patient_id}", ln=2)
                pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=3)
                pdf.cell(200, 10, txt=f"Seizure Count: {int(counts[1]) if 1 in counts else 0}", ln=4)
                pdf.cell(200, 10, txt=f"Model Confidence: {round(predictions.mean() * 100, 2)}%", ln=5)
                pdf.multi_cell(0, 10, txt=f"Doctor Note: {doctor_note}")

                # Save to temp file and make download link
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    pdf.output(tmp.name)
                    with open(tmp.name, "rb") as f:
                        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                    href = f'<a href="data:application/pdf;base64,{base64_pdf}" download="{patient_id}_seizure_report.pdf">📥 Download Report</a>'
                    st.markdown(href, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ Error reading the file: {e}")

# Footer
st.markdown("""<hr style="margin-top:50px;">
    <p style="text-align:center; font-size:14px;">
        Built with ❤️ by <b>Kashish Seth</b> | Powered by LSTM & Streamlit
    </p>
""", unsafe_allow_html=True)
