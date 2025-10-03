# 🩺 Multiple Disease Prediction System

Multiple Disease Prediction is an advanced machine learning project designed to predict the likelihood of **Kidney Disease, Liver Disease, and Parkinson’s Disease**. The system supports early diagnosis, enhances healthcare provider decision-making, and reduces diagnostic time and cost. It integrates data preprocessing, model training, evaluation, and visualization using Streamlit.

---

## 🔧 Tech Stack

![Python](https://img.shields.io/badge/Python-3.8%2B-gray?logo=python&logoColor=white&labelColor=3776AB)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-gray?logo=pandas&logoColor=white&labelColor=150458)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-gray?logo=numpy&logoColor=white&labelColor=013243)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML%20Models-gray?logo=scikit-learn&logoColor=white&labelColor=f89939)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-gray?logo=python&logoColor=white&labelColor=4187f6)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-gray?logo=streamlit&logoColor=white&labelColor=FF4B4B)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualizations-gray?logo=matplotlib&logoColor=white&labelColor=11557c)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualizations-gray?logo=python&logoColor=white&labelColor=8C5E9C)

---

## 📁 Project Structure
📂 multiple-disease-prediction
|
├── 📁 app/                           # Streamlit application code
│   └── app.py
|
├── 📁 data/
│   ├── 📁 raw/                       # Original/raw datasets
│   │   ├── parkinsons.csv
│   │   ├── kidney_disease.csv
│   │   └── indian_liver_patient.csv
│   │
│   ├── 📁 cleaned/                   # Cleaned/preprocessed datasets
│   │   ├── parkinsons_preprocessed.csv
│   │   ├── kidney_preprocessed.csv
│   │   └── liver_preprocessed.csv
│
├── 📁 models/                        # Trained ML models (saved as pickle files)
│   ├── kidney_model.pkl
│   ├── liver_model.pkl
│   └── parkinsons_model.pkl
|
├── 📁 src/                           # Source code for preprocessing and training
│   ├── preprocess_kidney.py
│   ├── preprocess_liver.py
│   ├── preprocess_parkinsons.py
│   └── train_models.py
|
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
├── .gitignore                        # Ignore unnecessary files in git
└── LICENSE                           # Open-source license for project

## 🚀 How to Run

1. Clone the repository  
bash
git clone https://github.com/wilson-in/multiple_disease_prediction.git
cd multiple-disease-prediction

Install dependencies

bashpip install -r requirements.txt

Run the Streamlit app

bashstreamlit run app/app.py

📊 Features

Multi-Disease Prediction: Predicts Kidney, Liver, and Parkinson’s disease probability using user-provided data (symptoms, demographics, test results).
Data Preprocessing: Manages missing values, encodes categorical features, and scales numerical data for model accuracy.
Machine Learning Models: Trained with Logistic Regression, Random Forest, and XGBoost for robust predictions.
Interactive Streamlit App: Allows real-time input of health data with instant probability and risk level outputs.
Model Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
Scalable & Secure: Supports multiple users with in-memory data processing for privacy.
Visual Insights: Displays probability charts and confusion matrices for better understanding.
