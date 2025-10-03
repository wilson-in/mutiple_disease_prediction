# 🩺 Multiple Disease Prediction System

A comprehensive system that predicts **Kidney, Liver, and Parkinson’s diseases** using machine learning.  
Includes data preprocessing, model training, evaluation, a Streamlit web app for inference, and Power BI dashboards for visualization.

---

## 📂 Project Structure

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

## 📊 Datasets

Used **3 datasets** from public sources (e.g., UCI / Kaggle):

- Kidney Disease  
- Indian Liver Patient  
- Parkinson’s Disease  

Raw files stored in `data/raw/`; preprocessed data in `data/cleaned/`.

---

## 🔬 Methodology

1. **Data Preprocessing**  
   - Handle missing values  
   - Encode categorical features (e.g., gender)  
   - Standard scaling for numerical features  
   - Save cleaned datasets  

2. **Model Training**  
   - Train/test split (80/20)  
   - Algorithms: RandomForest, XGBoost, Logistic Regression  
   - Save best models, scalers, and metadata  

3. **Evaluation**  
   - Accuracy, ROC-AUC, Confusion Matrix  
   - Reports saved in `training_reports.json`  
   - Confusion matrix plots saved as `.png`  

4. **Deployment**  
   - Streamlit app loads models & metadata  
   - User selects disease, enters values, gets prediction  
   - Confusion matrix image shown  

5. **Visualization (Power BI)**  
   - Load datasets & metrics  
   - Build dashboards: distribution, features, performance  
   - Export dashboards to PDF  

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

---

## 📊 Using Power BI Dashboards

# 1. Open files in powerbi_report/ using Power BI Desktop

# 2. Dashboards include:
#    - Disease class distribution
#    - Age distribution
#    - Feature comparisons (e.g., Blood Urea, Bilirubin, Jitter)
#    - Confusion matrix visuals

# 3. Export dashboards to PDF or embed in reports

---

## 🏆 Results & Performance

# Reports saved in models/training_reports.json:
# - Kidney: Accuracy = ..., ROC-AUC = ...
# - Liver: Accuracy = ..., ROC-AUC = ...
# - Parkinson’s: Accuracy = ..., ROC-AUC = ...

# Confusion matrix plots saved in /models

---

## ⚙️ Requirements

# Core packages:
pandas  
numpy  
scikit-learn  
xgboost  
matplotlib  
seaborn  
streamlit  
joblib  

# Install all:
pip install -r requirements.txt

---

## 🎯 Future Work

# - Add more diseases (Diabetes, Heart, Lung)  
# - Use deep learning & ensembles  
# - Cloud deployment (AWS / Heroku)  
# - Advanced front-end with interactive charts  

---

## 📂 Usage Example

# 1. Launch Streamlit app  
# 2. Select a disease  
# 3. Enter patient parameters  
# 4. Get prediction + confusion matrix  

---

## 👨‍💻 Author

**Author**: Wilson  
GitHub: [wilson-in](https://github.com/wilson-in/mutiple_disease_prediction)  
License: MIT
