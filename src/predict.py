# src/predict.py - FIXED - 10 features, final retrained GB 844k, MODELS_DIR dynamic
import joblib
import pandas as pd
from pathlib import Path
import sys

# Compat shim for sklearn unpickle
try:
    import sklearn.ensemble._gb_losses
    sys.modules['_loss'] = sklearn.ensemble._gb_losses
except Exception:
    pass

BASE_DIR = Path(__file__).parent.parent.resolve()
MODELS_DIR = BASE_DIR / "models"

def load_model_and_encoder():
    model_path_final = MODELS_DIR / "model_step6_gradientboosting_final_retrained_complete_open.pkl"
    model_path = MODELS_DIR / "model_step6_gradientboosting.pkl"
    use_path = model_path_final if model_path_final.exists() else model_path
    model = joblib.load(use_path)
    encoder = joblib.load(MODELS_DIR / "encoder.joblib")
    column_names = joblib.load(MODELS_DIR / "columnName.joblib")
    return model, encoder, column_names

def preprocess_input_fixed(input_dict, encoder, column_names):
    # Date parsing
    date_str = input_dict.get("Date", "2015-07-31")
    try:
        dt = pd.to_datetime(date_str)
    except:
        import datetime
        dt = datetime.datetime.now()

    base = {
        "Store": int(input_dict.get("Store", 1)),
        "DayOfWeek": int(input_dict.get("DayOfWeek", 1)),
        "Open": int(input_dict.get("Open", 1)),
        "Promo": int(input_dict.get("Promo", 0)),
        "SchoolHoliday": int(input_dict.get("SchoolHoliday", 0)),
        "Year": dt.year,
        "Month": dt.month,
        "Day": dt.day,
        "WeekOfYear": int(dt.isocalendar().week) if hasattr(dt.isocalendar(), 'week') else dt.isocalendar()[1]
    }

    # Handle StateHoliday encoding to match train.py
    state = str(input_dict.get("StateHoliday", "0"))

    # Try encoder (OneHot)
    if encoder is not None and hasattr(encoder, 'transform'):
        try:
            cat_df = pd.DataFrame([[state]], columns=["StateHoliday"])
            encoded = encoder.transform(cat_df)
            encoded_cols = encoder.get_feature_names_out(["StateHoliday"])
            for i, col in enumerate(encoded_cols):
                base[col] = float(encoded[0][i])
        except Exception:
            # fallback: if column_names contains StateHoliday raw, keep raw mapping
            base["StateHoliday"] = state
    else:
        base["StateHoliday"] = state

    # Build row in expected order
    if column_names:
        row = []
        for col in column_names:
            if col in base:
                row.append(base[col])
            else:
                # If col is StateHoliday_a etc, check
                row.append(base.get(col, 0))
        df = pd.DataFrame([row], columns=column_names)
    else:
        df = pd.DataFrame([base])

    # If StateHoliday column is still string and model expects numeric, map it
    if "StateHoliday" in df.columns and df["StateHoliday"].dtype == object:
        mapping = {"0": 0, "a": 1, "b": 2, "c": 3, "0.0": 0}
        df["StateHoliday"] = df["StateHoliday"].map(lambda x: mapping.get(str(x), 0))

    return df

def predict_sales(input_dict: dict):
    model, encoder, column_names = load_model_and_encoder()

    df = preprocess_input_fixed(input_dict, encoder, column_names)

    # Ensure correct column order from model
    expected = list(getattr(model, 'feature_names_in_', column_names if column_names else []))
    if expected:
        for col in expected:
            if col not in df.columns:
                df[col] = 0
        df = df[expected]

    pred = float(model.predict(df)[0])

    # If closed, sales 0
    if int(input_dict.get('Open', 1)) == 0:
        pred = 0.0

    return pred