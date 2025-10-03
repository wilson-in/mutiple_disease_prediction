# Multiple Disease Prediction System

A comprehensive system that predicts **Kidney, Liver, and Parkinson’s diseases** using machine learning. Includes data preprocessing, model training, evaluation, a Streamlit web app for inference, and Power BI dashboards for visualization.

---

## 🧠 Overview

Healthcare datasets often focus on individual diseases. This project integrates multiple disease prediction in one system, allowing users to input clinical/lab features and predict risk for multiple conditions. The system also provides visual analytics to interpret data, model performance, and feature insights.

---

## 🔍 Features

- Preprocess raw datasets: cleaning, encoding, scaling  
- Train separate models per disease  
- Save trained models, scalers, metadata & evaluation reports  
- Deploy a **Streamlit web app** to take inputs and show predictions  
- Build **Power BI dashboards** for EDA and model performance  
- Documentation, reproducibility, and modular code

---

## 📁 Project Structure

## 📂 Project Structure

bash
4-multiple-disease-prediction/
│── app/                      # Streamlit app for disease prediction
│   └── streamlit_app.py
│
│── data/                     # Datasets
│   ├── raw/                  # Raw datasets (Kidney, Liver, Parkinson’s)
│   │   ├── kidney_disease.csv
│   │   ├── indian_liver_patient.csv
│   │   └── parkinsons.csv
│   └── cleaned/              # Preprocessed datasets
│       ├── kidney_preprocessed.csv
│       ├── liver_preprocessed.csv
│       └── parkinsons_preprocessed.csv
│
│── models/                   # Trained ML models, scalers & evaluation reports
│   ├── kidney_model.pkl
│   ├── liver_model.pkl
│   ├── parkinsons_model.pkl
│   ├── kidney_scaler.pkl
│   ├── liver_scaler.pkl
│   ├── parkinsons_scaler.pkl
│   ├── kidney_confusion.png
│   ├── liver_confusion.png
│   ├── parkinsons_confusion.png
│   ├── metadata.json
│   └── training_reports.json
│
│── notebooks/                # Training and experimentation notebooks/scripts
│   ├── Multiple_Disease_Trainer.py
│   └── train_models.py
│
│── powerbi_report/            # Power BI dashboards & exports
│   ├── Kidney_Report.pbix
│   ├── Liver_Report.pbix
│   ├── Parkinsons_Report.pbix
│   └── dashboards.pdf
│
│── requirements.txt           # Python dependencies
│── .gitignore                 # Ignored files & folders
│── README.md                  # Project documentation



---

## 🧪 Datasets

Used 3 datasets from public sources (e.g. UCI / Kaggle):

- **Kidney Disease**  
- **Indian Liver Patient**  
- **Parkinson’s Disease**

Raw files stored in `data/raw/`; processed & cleaned data in `data/processed/`.

---

## 📋 Methodology

**1. Data Preprocessing**  
- Handle missing values  
- Encode categorical features (e.g. gender)  
- Standard scaling of numerical features  
- Save cleaned datasets for reproducibility  

**2. Model Training**  
- Split each disease dataset into train/test (80/20)  
- Try algorithms like RandomForest, XGBoost, Logistic Regression  
- Select best model per disease  
- Save models, scalers, and metadata (feature order + encodings)  

**3. Evaluation**  
- Compute Accuracy, ROC-AUC, Confusion Matrix  
- Save evaluation reports in `training_reports.json`  
- Save confusion matrix plots  

**4. Deployment**  
- Streamlit app loads metadata and models  
- User selects disease, enters input values, and obtains prediction  
- Confusion matrix image shown alongside prediction  

**5. Visualization (Power BI)**  
- Load raw datasets and metrics  
- Build dashboards per disease: class distribution, feature comparison, model performance  
- Export dashboards to PDF for reporting  

---

## 🚀 Running the Project Locally

# 1. Create & activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train models (if not provided)
python notebooks/train_models.py

# 4. Run the Streamlit app
cd app
streamlit run streamlit_app.py
# Then open: http://localhost:8501


## 📊 Using Power BI Dashboards

# 1. Open files in powerbi_report/ using Power BI Desktop

# 2. Dashboards include:
#    - Disease class distribution
#    - Age distribution
#    - Feature comparisons (e.g., Blood Urea, Bilirubin, Jitter)
#    - Confusion matrix visuals

# 3. Export dashboards to PDF or embed visualizations into your report


## 🏆 Results & Performance

# The models/training_reports.json contains performance for each disease model:
# - Kidney: Accuracy = ..., ROC-AUC = ...
# - Liver: Accuracy = ..., ROC-AUC = ...
# - Parkinson’s: Accuracy = ..., ROC-AUC = ...

# Confusion matrix images are saved in /models


## ⚙️ Requirements & Dependencies

# Key packages required:
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
streamlit
joblib

# Install them all at once:
pip install -r requirements.txt


## 🎯 Future Work

# Planned enhancements:
# - Include more diseases (Diabetes, Heart, Lung)
# - Use deep learning / ensemble models
# - Deploy on cloud services (AWS, Heroku)
# - Add frontend enhancements (charts, explanations)


## 📂 Usage Example

# 1. Launch the Streamlit app
# 2. Select a disease (Kidney / Liver / Parkinson’s)
# 3. Enter patient parameter values
# 4. Click “Predict” → result displayed + confusion matrix
