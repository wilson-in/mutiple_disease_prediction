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
# 🩺 Multiple Disease Prediction System  

A machine learning project that predicts the probability of **Kidney Disease**, **Liver Disease**, and **Parkinson’s Disease** using medical datasets.  
Built with **Python, scikit-learn, pandas/numpy, and Streamlit**.  

---

## 📂 Project Structure  

```bash
multiple-disease-prediction
├── 📁 app/                      # Streamlit application code
│   └── app.py
│
├── 📁 data/
│   ├── 📁 raw/                  # Original/raw datasets
│   │   ├── parkinsons.csv
│   │   ├── kidney_disease.csv
│   │   └── indian_liver_patient.csv
│   │
│   ├── 📁 cleaned/              # Cleaned/preprocessed datasets
│   │   ├── parkinsons_preprocessed.csv
│   │   ├── kidney_preprocessed.csv
│   │   └── liver_preprocessed.csv
│
├── 📁 models/                   # Trained ML models (saved as pickle files)
│   ├── kidney_model.pkl
│   ├── liver_model.pkl
│   └── parkinsons_model.pkl
│
├── 📁 src/                      # Source code for preprocessing and training
│   ├── preprocess_kidney.py
│   ├── preprocess_liver.py
│   ├── preprocess_parkinsons.py
│   └── train_models.py
│
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── .gitignore                   # Ignore unnecessary files in git
└── LICENSE                      # Open-source license


🚀 How to Run

Clone the repository

git clone https://github.com/wilson-in/multiple_disease_prediction.git
cd multiple-disease-prediction


Install dependencies

pip install -r requirements.txt


Run the Streamlit app

streamlit run app/app.py
