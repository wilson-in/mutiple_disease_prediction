Multiple Disease Prediction System
A machine learning-based web application that predicts multiple diseases including Diabetes, Heart Disease, and Parkinson's Disease using patient data and symptoms.

Features
Multi-Disease Prediction - Single platform for three different disease predictions

Machine Learning Models - Trained on verified medical datasets

User-Friendly Interface - Simple web interface built with Streamlit

Real-time Predictions - Instant results with probability scores

Privacy Focused - Local processing of sensitive health data

Installation
Prerequisites
Python 3.8 or higher

pip package manager

Setup Instructions
Clone the repository

bash
git clone https://github.com/wilson-in/multiple_disease_prediction.git
cd multiple_disease_prediction
Create virtual environment (recommended)

bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies

bash
pip install -r requirements.txt
Usage
Start the application

bash
streamlit run app.py
Access the web interface

Open your browser and navigate to http://localhost:8501

Make predictions

Select the disease type from the sidebar

Input the required medical parameters

Click predict to get results

Project Structure
text
multiple_disease_prediction/
├── data/                    # Dataset files
├── models/                  # Trained ML models
├── notebooks/               # Jupyter notebooks
├── src/                     # Source code
├── app.py                   # Main application
├── requirements.txt         # Dependencies
└── README.md                # Documentation
Machine Learning Models
Algorithms Used
Random Forest Classifier

Support Vector Machine (SVM)

Logistic Regression

XGBoost

Model Performance
Diabetes: Random Forest (92% accuracy)

Heart Disease: SVM (87% accuracy)

Parkinson's: XGBoost (95% accuracy)
