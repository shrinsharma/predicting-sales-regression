# train.py - One-click training for Project 3 - Mentor style same as Medicine Review
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from src.preprocessing import preprocess_sales_data
from src.feature_engineering import create_features, save_encoder, save_scaler_dict, save_column_names
from src.model_training import (
    train_linear, train_lasso, train_ridge,
    train_decision_tree, train_random_forest,
    train_gradient_boosting, train_xgboost,
    evaluate_model, save_model
)

# Dataset path and model folder - same pattern as mentor
DATA_PATH = "data/regression.csv"  # Place your CSV here OR use sqlite in Colab
MODEL_FOLDER = "models"

os.makedirs(MODEL_FOLDER, exist_ok=True)

print("Loading data...")
# Try to load CSV - if not present, instruction for Colab DB
try:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded from {DATA_PATH}: {df.shape}")
except FileNotFoundError:
    print(f"File not found: {DATA_PATH}")
    print("For this project: Either place regression.csv in data/ folder")
    print("OR use Colab path: /content/drive/MyDrive/INTTRVU/CAPSTONE/Project 3/regression.db via sqlite3")
    print("Example:")
    print("import sqlite3; conn=sqlite3.connect('/content/drive/MyDrive/.../regression.db'); df=pd.read_sql('SELECT * FROM regression', conn)")
    exit()

print(df.head())

# Preprocessing - Date conversion + drop Customers leakage
df = preprocess_sales_data(df)

# Feature Engineering - Encoder + Scaler + Columns
print("Creating features...")
df_features, encoder, scaler_dict, columns = create_features(df, is_train=True)

# Separate X and y
TARGET_COLUMN = "Sales"
X = df_features.drop(columns=[TARGET_COLUMN])
y = df_features[TARGET_COLUMN]

# Train-test split - time-based better but using random for simplicity like mentor
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Train 7 models - exact same as Colab STEP 6
print("\nTraining Linear...")
linear = train_linear(X_train, y_train)
evaluate_model(linear, X_test, y_test)
save_model(linear, "models/model_step6_linear.pkl")

print("\nTraining Lasso...")
lasso = train_lasso(X_train, y_train)
evaluate_model(lasso, X_test, y_test)
save_model(lasso, "models/model_step6_lasso.pkl")

print("\nTraining Ridge...")
ridge = train_ridge(X_train, y_train)
evaluate_model(ridge, X_test, y_test)
save_model(ridge, "models/model_step6_ridge.pkl")

print("\nTraining DecisionTree depth10...")
dt = train_decision_tree(X_train, y_train)
evaluate_model(dt, X_test, y_test)
save_model(dt, "models/model_step6_decisiontree_fix1_depth10.pkl")

print("\nTraining RandomForest...")
rf = train_random_forest(X_train, y_train)
evaluate_model(rf, X_test, y_test)
save_model(rf, "models/model_step6_randomforest.pkl")

print("\nTraining GradientBoosting - BEST MODEL...")
gb = train_gradient_boosting(X_train, y_train)
result = evaluate_model(gb, X_test, y_test)
save_model(gb, "models/model_step6_gradientboosting.pkl")
print(f"BEST Model R2: {result['r2']:.4f} - Expected 0.6601")

print("\nTraining XGBoost...")
xgb_model = train_xgboost(X_train, y_train)
evaluate_model(xgb_model, X_test, y_test)
save_model(xgb_model, "models/model_step6_xgboost.pkl")

# Save encoder, scaler, columns - critical for app.py
print("\nSaving Encoder, ScalerDict, ColumnNames...")
save_encoder(encoder, "models/encoder.joblib")
save_scaler_dict(scaler_dict, "models/scalerDict.joblib")
save_column_names(columns, "models/columnName.joblib")

print("\nTraining complete. All 11 files saved in models/ folder.")