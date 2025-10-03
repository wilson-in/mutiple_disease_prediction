# Multiple Disease Prediction

## Project Title
Multiple Disease Prediction

## Objective
Build a scalable, accurate system to:
- Aid early disease detection.
- Enhance healthcare decision-making.
- Reduce diagnostic time/cost with quick predictions.

## System Architecture
- **Frontend**: Streamlit UI for input (symptoms, test results).
- **Backend**: Python processes inputs and runs ML models.
- **ML Models**: Logistic Regression, Random Forest, XGBoost.

## Features
- Predicts Kidney, Liver, Parkinson's diseases.
- User-friendly interface.
- Interactive visualizations.
- Secure data handling.
- Scalable for many users.

## Workflow
- **Input**: User enters symptoms, demographics, test results.
- **Preprocessing**: Handles missing data, encodes/scales features.
- **Inference**: ML models predict disease probabilities.
- **Output**: Shows predicted diseases with probabilities/risk levels.

## Data Collection
- **Sources**: Parkinson's, Kidney, Indian Liver Patient datasets.
- **Features**: Symptoms, test results, demographics.

## Preprocessing
- Handles missing data, encodes categoricals, scales features.

## Training
- Trains separate models per disease.
- Uses cross-validation.

## Evaluation
- Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC.
- Confusion Matrix.

## Tools
- **Language**: Python
- **Libs**: Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
- **Frontend**: Streamlit

## Results
Enhances healthcare access/efficiency with ML and user-friendly UI.

## Evaluation Metrics
- **Regression**: MAE, RMSE
- **Classification**: Accuracy, Precision, Recall, F1-score
- **App**: Streamlit responsiveness, visualization quality

## Tags
Streamlit, Python, Machine Learning, Visualization

## Deliverables
- **Code**: Preprocessing, modeling, Streamlit scripts.
- **App**: Functional web app (local/web).
- **Docs**: Model explanations, run instructions.
- **Presentation**: Results/business insights.

## Guidelines
- Follow Python standards.
- Use GitHub/GitLab.
- Ensure modularity.
- Optimize with caching.
- Validate models/app regularly.

## Getting Started
### Prerequisites
- Python 3.8+
- pip install -r requirements.txt

### Installation
1. git clone https://github.com/wilson-in/multiple_disease_prediction.git
2. cd multiple_disease_prediction
3. python -m venv venv
4. Activate: venv\Scripts\activate (Windows) or source venv/bin/activate (macOS/Linux)
5. pip install -r requirements.txt

### Running
1. python src/preprocess_kidney.py
2. python src/preprocess_liver.py
3. python src/preprocess_parkinsons.py
4. python src/train_models.py
5. streamlit run app.py (open http://localhost:8501)


