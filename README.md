🧠 EEG Seizure Detection using LSTM Neural Network
This project is a deep learning-based web app built to detect epileptic seizures from EEG signals. It uses a trained LSTM model to classify seizure and non-seizure events based on 178-feature EEG input data. The frontend is built with Streamlit for easy interaction and deployment.

🔍 Project Overview
Objective: Detect epileptic seizures using EEG data.

Approach: Train an LSTM (Long Short-Term Memory) model on EEG time-series data.

Deployment: Streamlit app for public access and user testing.

Input: CSV files with EEG readings (178 features per row).

Output: Seizure prediction (0 = No Seizure, 1 = Seizure).

🚀 Tech Stack
Tech	Usage
Python	Core Programming Language
TensorFlow/Keras	LSTM Model Training
Pandas & NumPy	Data Handling and Processing
Streamlit	Web App Interface
Matplotlib	Prediction Visualization
🧠 Model Details
Type: LSTM (Long Short-Term Memory)

Input Shape: (178, 1)

Layers: LSTM → Dense → Sigmoid Output

Loss: Binary Crossentropy

Optimizer: Adam

The model is trained to recognize patterns in EEG signals that correlate with seizures.

📂 File Structure
graphql
Copy
Edit
├── app.py                   # Streamlit App Script
├── seizure_model.h5         # Trained LSTM Model
├── requirements.txt         # Required Python Libraries
├── synthetic_eeg_data.csv   # Sample EEG Test File
└── README.md                # GitHub Readme
⚙️ How to Use
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/yourusername/eeg-seizure-detector.git
cd eeg-seizure-detector
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Run the Streamlit App
bash
Copy
Edit
streamlit run app.py
📄 Input Format
Your input file should be a CSV where:

Each row is one EEG signal sample

It must contain exactly 178 columns

No header is required

✅ Example:

Copy
Edit
0.12, 0.23, ..., 0.04
0.31, 0.35, ..., 0.11
📈 Output
The app predicts for each row whether it's a seizure or not.

Results are shown in a table.

A download button allows users to export predictions.

🌐 Deployment Options
Streamlit Cloud (Free hosting)

Hugging Face Spaces (optional)

Docker (for advanced deployment)

🖼 Sample Screenshot

👩‍💻 Author
 KASHISH SETH
