# Inttrvu Project 3 - Predicting Sales - Regression

[Banner](banner.png)

> Capstone Project 3: End-to-end Rossmann Store Sales prediction using GradientBoosting (R² 0.6601) with 15 engineered features + FastAPI deployment.

**Live Repo:** https://github.com/shrinsharma/predicting-sales-regression

### 📊 Overview
This project predicts daily sales for Rossmann stores based on store, promo, competition and time-based features. The final pipeline includes custom preprocessing, feature engineering, and a production-ready FastAPI app.

**Prediction Example:** `₹6992.38`

### 🎯 Problem Statement
Predict `Sales` (regression) using store characteristics, promotions, competition metrics, and calendar features.

### 📁 Dataset
- Rossmann Store Sales (Kaggle)
- Train/Test split with time-based features engineered in `src/feature_engineering.py`

### ✨ Features (15 Final)
Core: `Store, DayOfWeek, Promo, CompetitionDistance`
Engineered: 11 features from `src/feature_engineering.py` including:
- `Year, Month, WeekOfYear, IsMonthStart/End, CompetitionOpen, Promo2Active, IsWeekend, AvgSalesPerCustomer (lag), etc.`

Full list saved in `models/columnName.joblib`

### 🤖 Models Compared (7)
All models trained in `train.py` and compared in `models/model_comparison_FINAL_7_EXACT.csv`:

| Model | File |
| :--- | :--- |
| Linear Regression | `model_step6_linear.pkl` |
| Ridge | `model_step6_ridge.pkl` |
| Lasso | `model_step6_lasso.pkl` |
| Decision Tree (depth=10) | `model_step6_decisiontree_fix1_depth10.pkl` |
| Random Forest | `model_step6_randomforest.pkl` |
| **Gradient Boosting (BEST)** | `model_step6_gradientboosting.pkl` |
| XGBoost | `model_step6_xgboost.pkl` |

**Best Model:** `GradientBoostingRegressor`
- **R² Score: 0.6601**
- Scaler: `scalerDict.joblib`
- Encoder: `encoder.joblib`

### 📂 Project Structure
```
.
├── app.py                              # FastAPI app - production API
├── train.py                            # Training pipeline (final)
├── Project_3_Predicting_Sales.ipynb    # EDA + Experimentation
├── requirements.txt
├── models/
│   ├── model_step6_gradientboosting.pkl # BEST MODEL
│   ├── model_step6_*.pkl               # Other 6 models
│   ├── scalerDict.joblib
│   ├── encoder.joblib
│   ├── columnName.joblib
│   └── model_comparison_FINAL_7_EXACT.csv
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── predict.py
├── templates/
│   └── index.html                      # Frontend
└── static/
    └── style.css
```

### 🚀 Installation

```bash
# Clone
git clone https://github.com/shrinsharma/predicting-sales-regression.git
cd predicting-sales-regression

# Create venv (Windows)
python -m venv venv
.\venv\Scripts\activate

# Install
pip install -r requirements.txt
```

### 🏋️ Training

```bash
python train.py
```
This will retrain all 7 models, save .pkl files to `models/` and generate `model_comparison_FINAL_7_EXACT.csv`

### 🌐 Run FastAPI App

```bash
uvicorn app:app --reload --port 8000
```
Open: http://127.0.0.1:8000

**API Endpoint:**
- `POST /predict`
```json
{
  "Store": 1,
  "DayOfWeek": 5,
  "Promo": 1,
  "CompetitionDistance": 1270.0
  // ... + 11 engineered features auto-handled by src/predict.py
}
```
Response:
```json
{
  "prediction": 6992.38,
  "currency": "INR"
}
```

### 🛠️ Tech Stack
- **Python:** 3.11.9
- **ML:** scikit-learn, XGBoost, pandas, numpy, joblib
- **API:** FastAPI, Uvicorn, Jinja2
- **Frontend:** HTML, CSS

### 📈 Results
- Cleaned .gitignore - venv, __pycache__, .zip, data_temp ignored
- Reproducible training
- Deployed API with ₹ formatting

### 👨‍💻 Author
**Shrin Sharma** - [GitHub](https://github.com/shrinsharma)

---
*Capstone Project 3 - INTTRVU*
