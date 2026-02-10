# 🩺 Multiple Disease Prediction System

Educational ML-based disease risk estimation system built using Python and Streamlit.

---

## 📌 Project Overview

This project predicts the **risk probability** of multiple diseases using machine learning models:

- Kidney Disease
- Liver Disease
- Parkinson’s Disease

The system assists in **early disease detection**, supports **faster decision-making**, and demonstrates a **real-world ML pipeline** from preprocessing to deployment.

---

## 🧠 Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn

---

## 🏗️ Project Structure
app/
└── app.py # Streamlit application
data/
└── cleaned/ # Preprocessed datasets
models/ # Trained models, scalers, imputers
train_models.py # Model training script
requirements.txt
.gitignore
README.md
---

## 🔄 Workflow

1. User enters medical test values and demographic details
2. Data is preprocessed (imputation + scaling)
3. Disease-specific ML models predict risk probability
4. Risk level is displayed (Low / Moderate / High)

---

## 🚀 How to Run the Project

### 1️⃣ Clone the repository
```bash
git clone https://github.com/wilson-in/mutiple_disease_prediction.git
cd mutiple_disease_prediction

2️⃣ Create virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate   # Windows
3️⃣ Install dependencies
pip install -r requirements.txt
4️⃣ Train the models
python train_models.py
5️⃣ Run the Streamlit app
streamlit run app/app.py

⚠️ Disclaimer

This system is for educational purposes only and does not provide medical diagnoses.