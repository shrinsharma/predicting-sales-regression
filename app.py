# app.py - FIXED v2 - MODELS_DIR + final retrained + sklearn compat shim
from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import sys
import joblib
import pandas as pd
from datetime import datetime

# === SKLEARN COMPAT SHIM FOR _loss MODULE ===
# Fix for pickle created in Colab vs Python 3.14 local
try:
    import sklearn.ensemble._gb_losses
    sys.modules['_loss'] = sklearn.ensemble._gb_losses
except Exception:
    pass
try:
    import sklearn.ensemble._hist_gradient_boosting.loss
    sys.modules['sklearn.ensemble._loss'] = sklearn.ensemble._hist_gradient_boosting.loss
    sys.modules['sklearn.ensemble._gb_losses'] = sklearn.ensemble._gb_losses if 'sklearn.ensemble._gb_losses' in sys.modules else sys.modules.get('_loss')
except Exception:
    pass
# Additional shims for GradientBoosting
for mod_name in [
    'sklearn.ensemble.gradient_boosting',
    'sklearn.ensemble._gradient_boosting',
]:
    try:
        __import__(mod_name)
    except Exception:
        pass

BASE_DIR = Path(__file__).parent.resolve()
MODELS_DIR = BASE_DIR / "models"
print(f"MODELS_DIR: {MODELS_DIR}")

MODEL_PATH = MODELS_DIR / "model_step6_gradientboosting.pkl"
FINAL_MODEL_PATH = MODELS_DIR / "model_step6_gradientboosting_final_retrained_complete_open.pkl"
if FINAL_MODEL_PATH.exists():
    MODEL_PATH = FINAL_MODEL_PATH

print(f"Loading model: {MODEL_PATH}")

model = None
encoder = None
column_names = None

try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded OK: {type(model)}")
except Exception as e:
    print(f"Model load error: {e}")
    # Try alternative pickle loading with compat
    try:
        import pickle
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded via pickle OK: {type(model)}")
    except Exception as e2:
        print(f"Second load attempt failed: {e2}")

try:
    encoder = joblib.load(MODELS_DIR / "encoder.joblib")
    print(f"Encoder loaded")
except Exception as e:
    print(f"Encoder load error: {e}")

try:
    column_names = joblib.load(MODELS_DIR / "columnName.joblib")
    print(f"Columns: {column_names}")
except Exception as e:
    print(f"ColumnName load error: {e}")

app = FastAPI(title="Predicting Sales - Store Sales Prediction")

if (BASE_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")
if (BASE_DIR / "templates").exists():
    templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
else:
    templates = Jinja2Templates(directory="templates")

def preprocess_input(input_dict):
    date_str = input_dict["Date"]
    try:
        dt = pd.to_datetime(date_str)
    except:
        dt = datetime.now()

    data = {
        "Store": int(input_dict["Store"]),
        "DayOfWeek": int(input_dict["DayOfWeek"]),
        "Open": int(input_dict["Open"]),
        "Promo": int(input_dict["Promo"]),
        "SchoolHoliday": int(input_dict["SchoolHoliday"]),
        "Year": dt.year,
        "Month": dt.month,
        "Day": dt.day,
        "WeekOfYear": int(dt.isocalendar().week) if hasattr(dt.isocalendar(), 'week') else dt.isocalendar()[1]
    }

    state = input_dict["StateHoliday"]
    if encoder is not None and hasattr(encoder, 'get_feature_names_out'):
        try:
            cat_df = pd.DataFrame([[state]], columns=["StateHoliday"])
            encoded = encoder.transform(cat_df)
            encoded_cols = encoder.get_feature_names_out(["StateHoliday"])
            for i, col in enumerate(encoded_cols):
                data[col] = encoded[0][i]
        except Exception as e:
            print(f"Encoder error: {e}")
            for col in column_names if column_names else []:
                if "StateHoliday" in col:
                    data[col] = 1 if state in col else 0
    else:
        for col in column_names if column_names else []:
            if "StateHoliday" in col:
                data[col] = 1 if state in col else 0

    if column_names:
        row = []
        for col in column_names:
            row.append(data.get(col, 0))
        X = pd.DataFrame([row], columns=column_names)
    else:
        X = pd.DataFrame([data])

    return X

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )

@app.post("/predict")
async def predict(
    request: Request,
    Store: int = Form(...),
    DayOfWeek: int = Form(...),
    Date: str = Form(...),
    Open: int = Form(...),
    Promo: int = Form(...),
    StateHoliday: str = Form(...),
    SchoolHoliday: int = Form(...)
):
    input_dict = {
        "Store": Store,
        "DayOfWeek": DayOfWeek,
        "Date": Date,
        "Open": Open,
        "Promo": Promo,
        "StateHoliday": StateHoliday,
        "SchoolHoliday": SchoolHoliday
    }

    if model is None:
        prediction = 0.0
    else:
        X = preprocess_input(input_dict)
        prediction = float(model.predict(X)[0])
        if int(Open) == 0:
            prediction = 0.0

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "Store": Store,
            "DayOfWeek": DayOfWeek,
            "Date": Date,
            "Open": Open,
            "Promo": Promo,
            "StateHoliday": StateHoliday,
            "SchoolHoliday": SchoolHoliday,
            "prediction": round(prediction, 2)
        }
    )

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "models_dir": str(MODELS_DIR),
        "files": [p.name for p in MODELS_DIR.glob("*.pkl")] if MODELS_DIR.exists() else []
    }