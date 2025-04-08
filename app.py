import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from transformers import pipeline
import PyPDF2
import io

# ---------------------------
# SECTION 1: LOAD EEG SEIZURE DETECTION MODEL & DEFINE UI
# ---------------------------

@st.cache_resource
def load_trained_model():
    # Use the correctly named model file on GitHub, e.g., "seizure_model.h5"
    return load_model("seizure_model (1).h5")

model = load_trained_model()

# Main App Title and Description
st.markdown("""
    <h1 style='text-align: center; color: #FF4B4B;'>🧠 EEG Seizure Detection App</h1>
    <p style='text-align: center; font-size:18px;'>
        Upload EEG data (178 features per row) to detect seizures using an LSTM model.
    </p>
""", unsafe_allow_html=True)

# File uploader for EEG data
st.markdown("### 📂 Upload EEG CSV File")
uploaded_file = st.file_uploader("Drag and drop or browse your EEG CSV file", type="csv")

# Process EEG file and make predictions
if uploaded_file is not None:
    st.success("✅ EEG file uploaded successfully!")
    try:
        data = pd.read_csv(uploaded_file)
        if data.shape[1] != 178:
            st.error(f"❌ Invalid file: Expected 178 features per row, but got {data.shape[1]}.")
        else:
            # Reshape data for LSTM input
            X_input = data.values.reshape(data.shape[0], data.shape[1], 1)
            predictions = model.predict(X_input)
            predicted_classes = (predictions > 0.5).astype("int32").flatten()
            prediction_df = pd.DataFrame({"Seizure_Predicted": predicted_classes})
            
            st.markdown("### 🧾 EEG Prediction Results")
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
            
    except Exception as e:
        st.error(f"⚠️ Error processing EEG file: {e}")

# ---------------------------
# SECTION 2: CHATBOT INTEGRATION BASED ON UPLOADED REPORTS
# ---------------------------

st.markdown("---")
st.sidebar.header("🤖 Chat with AI Assistant (Based on Your Reports)")

# Load a Hugging Face transformer QA pipeline
@st.cache_resource
def load_qa_pipeline():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

qa_pipeline = load_qa_pipeline()

# Function to extract text from PDF files
def extract_text_from_pdf(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        return ""

# Create a session variable to hold the combined report text
if 'report_text' not in st.session_state:
    st.session_state.report_text = ""

st.sidebar.markdown("### 📄 Upload Your Reports")
uploaded_reports = st.sidebar.file_uploader(
    "Upload report files (PDF or TXT)", type=["pdf", "txt"], accept_multiple_files=True, key="reports")

if uploaded_reports:
    combined_text = ""
    for report in uploaded_reports:
        if report.type == "application/pdf":
            # Extract text from PDF
            combined_text += extract_text_from_pdf(report) + "\n"
        else:  # Assume text file (TXT)
            text = report.read().decode("utf-8")
            combined_text += text + "\n"
    st.session_state.report_text = combined_text
    st.sidebar.success("Reports uploaded and processed!")

st.sidebar.markdown("### ❓ Ask a Question")
user_query = st.sidebar.text_input("Ask a question based on the reports:")

if user_query:
    if st.session_state.report_text:
        try:
            result = qa_pipeline(question=user_query, context=st.session_state.report_text)
            st.sidebar.write("🤖 **Answer:**", result["answer"])
        except Exception as e:
            st.sidebar.error(f"Error getting answer: {e}")
    else:
        st.sidebar.error("Please upload reports to enable question answering.")

# ---------------------------
# SECTION 3: APP FOOTER
# ---------------------------
st.markdown("""<hr style="margin-top:50px;">
    <p style="text-align:center; font-size:14px;">
        Built with ❤️ by <b>Kashish Seth</b> | Powered by LSTM, Transformers & Streamlit
    </p>
""", unsafe_allow_html=True)
