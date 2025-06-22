# 🧠 EEG-Based Seizure Detection using LSTM

This project is a **medical AI application** that analyzes EEG (Electroencephalography) signals to detect **epileptic seizures** using a deep learning model (LSTM). It offers real-time classification and generates **doctor-ready PDF reports** with seizure count and confidence scores.

---

## 📌 What is EEG?

**EEG (Electroencephalography)** is a technique used to record the brain’s electrical activity over time. It is widely used in neuroscience and medical diagnostics to detect abnormalities in brain function.

### 🧬 How Does It Work?
- The human brain communicates using electrical impulses.
- EEG uses electrodes placed on the scalp to capture these signals.
- The recorded signals are visualized as waveforms representing brain activity.

---

## 📊 EEG in This Project

In this project, EEG data is used to classify whether a person is experiencing a **seizure** or not.

Since EEG signals are **time-series data**, a **Long Short-Term Memory (LSTM)** neural network was used to model temporal dependencies and patterns in the signal. This architecture is well-suited for analyzing sequential biomedical data like EEG.

---

## 💡 Why is EEG Important?

EEG is essential for diagnosing and monitoring:
- Epileptic seizures
- Sleep disorders
- Coma and brain death
- Encephalopathies
- Neurodegenerative diseases

---

## ⚙️ Features

- 📤 EEG file upload
- 🤖 Seizure detection using LSTM model
- 📈 Real-time classification with confidence scores
- 📄 Downloadable PDF reports (doctor-ready format)

---
### Screenshots
![image](https://github.com/user-attachments/assets/b69f082b-36d5-4b18-9b3a-76d5cae184b5)
![image](https://github.com/user-attachments/assets/cd11ecea-fe62-47c4-9698-92da20d60284)
![image](https://github.com/user-attachments/assets/5afce8dc-b467-478b-8098-9c24401bdb82)
![image](https://github.com/user-attachments/assets/c6550a59-a9f8-4b90-ae4d-759c2279cdec)
![image](https://github.com/user-attachments/assets/ff9e7b17-e5c7-4bf5-9af2-82bcd2948fc1)

## 💻 Tech Stack

| Component        | Technology               |
|------------------|---------------------------|
| Frontend         | Streamlit (Interactive UI)|
| Backend / Model  | Python, TensorFlow/Keras (LSTM) |
| Report Generation| ReportLab (PDF output)    |

---

## 🚀 How to Run

1. **Clone the repository**:
  - git clone https://github.com/yourusername/eeg-seizure-detection.git
  - cd eeg-seizure-detection


2. **Install dependencies**:

  - pip install -r requirements.txt


3. **Run the app**:
  -streamlit run app.py

### 📂 Sample Output
Prediction screen: Shows seizure status and confidence score.

PDF Report: Includes patient info, seizure count, model prediction, and timestamp.



### 📬 AUTHOR
For queries or collaborations, feel free to reach out !
kseth9852@gmail.com










