# 🩺 Multiple Disease Prediction System (Kidney, Liver, Parkinson’s)

This repository implements a **Multiple Disease Prediction System** using **Machine Learning models** to predict the likelihood of:
- Chronic Kidney Disease (CKD)  
- Liver Disease  
- Parkinson’s Disease  

It also includes **interactive Power BI dashboards** for visualization and **a Streamlit web app** for deployment.

---

## 📂 Project Structure
4-multiple-disease-prediction/
│── app/ # Streamlit app for disease prediction
│ └── streamlit_app.py
│── data/ # Datasets
│ ├── raw/ # Raw datasets (Kidney, Liver, Parkinson’s)
│ └── processed/ # Cleaned datasets
│── models/ # Trained ML models & metadata
│ ├── kidney_model.pkl
│ ├── liver_model.pkl
│ ├── parkinsons_model.pkl
│ └── metadata.json
│── notebooks/ # Jupyter notebooks for EDA & model training
│── powerbi_report/ # Power BI dashboards (.pbix, .pdf exports)
│── requirements.txt # Python dependencies
│── .gitignore # Files & folders ignored in Git
│── README.md # Project documentation

yaml
Copy code

---

## 📊 Datasets Used
All datasets are sourced from the **UCI Machine Learning Repository**:  
- **Kidney Disease Dataset**  
- **Indian Liver Patient Dataset**  
- **Parkinson’s Disease Dataset**  

Each dataset was cleaned, normalized, and preprocessed for model training.

---

## ⚙️ Methodology
1. **Data Preprocessing**
   - Missing values handled (mean/median/mode imputation)
   - Categorical variables label encoded
   - StandardScaler applied to numerical features

2. **Model Training**
   - Logistic Regression, Random Forest, and SVM tested
   - Best-performing models saved in `/models`

3. **Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1-score
   - Confusion matrices visualized in Power BI

4. **Deployment**
   - Streamlit app for interactive predictions
   - Power BI dashboards for data insights

---

## 🚀 Running the Project Locally

### 1. Clone the Repository
```bash
git clone https://github.com/wilson-in/mutiple_disease_prediction.git
cd mutiple_disease_prediction
2. Create a Virtual Environment
bash
Copy code
python -m venv venv
# Activate
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux/Mac
3. Install Dependencies
bash
Copy code
pip install -r requirements.txt
4. Train Models (if not already present)
bash
Copy code
python notebooks/Multiple_Disease_Trainer.py
This generates trained models and metadata.json inside /models.

5. Run the Streamlit App
bash
Copy code
cd app
streamlit run streamlit_app.py
The app will open at: http://localhost:8501/

📈 Power BI Dashboards
Navigate to powerbi_report/ and open .pbix files in Power BI Desktop.

Dashboards include:

Kidney Disease Dashboard

Pie chart: distribution of CKD vs non-CKD

Histogram: Age distribution

Bar chart: Biomarkers (Hemoglobin, Blood Urea, etc.)

Confusion matrix

Liver Disease Dashboard

Pie chart: Diseased vs Healthy

Histogram: Age distribution

Bar chart: Average Bilirubin by class

Confusion matrix

Parkinson’s Disease Dashboard

Pie chart: Disease vs Healthy

Histogram: Distribution of fundamental frequency (MDVP:Fo)

Bar chart: Key features (Shimmer, Jitter, etc.)

Confusion matrix

Export dashboards to PDF for reporting.

📜 Requirements
Install all dependencies with:

bash
Copy code
pip install -r requirements.txt
Main libraries:

scikit-learn

pandas

numpy

matplotlib

seaborn

streamlit

joblib

