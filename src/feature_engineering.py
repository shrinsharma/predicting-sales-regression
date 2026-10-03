# src/feature_engineering.py - FIXED 10 features, MODELS_DIR dynamic
import joblib
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).parent.parent.resolve()
MODELS_DIR = BASE_DIR / "models"

def create_encoder(df: pd.DataFrame):
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    if 'StateHoliday' in df.columns:
        encoder.fit(df[['StateHoliday']])
    return encoder

def transform_state_holiday(encoder, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'StateHoliday' not in df.columns or encoder is None:
        return df
    try:
        if hasattr(encoder, 'get_feature_names_out'):
            encoded = encoder.transform(df[['StateHoliday']])
            cols = encoder.get_feature_names_out(['StateHoliday'])
            enc_df = pd.DataFrame(encoded, columns=cols, index=df.index)
            df = df.drop(columns=['StateHoliday'])
            df = pd.concat([df, enc_df], axis=1)
        else:
            # fallback
            df['StateHoliday'] = df['StateHoliday'].map({"0":0,"a":1,"b":2,"c":3}).fillna(0)
    except Exception:
        df['StateHoliday'] = df['StateHoliday'].map({"0":0,"a":1,"b":2,"c":3}).fillna(0)
    return df

def save_encoder(encoder, path=None):
    if path is None:
        path = MODELS_DIR / "encoder.joblib"
    joblib.dump(encoder, path)
    return f"Saved encoder to {path}"

def load_encoder(path=None):
    if path is None:
        path = MODELS_DIR / "encoder.joblib"
    return joblib.load(path)

def create_scaler_dict(df: pd.DataFrame):
    scaler_dict = {}
    numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    exclude = ['Sales']
    numeric_cols = [c for c in numeric_cols if c not in exclude]
    for col in numeric_cols:
        scaler = StandardScaler()
        scaler.fit(df[[col]])
        scaler_dict[col] = scaler
    return scaler_dict

def transform_scaler(scaler_dict, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, scaler in scaler_dict.items():
        if col in df.columns:
            try:
                df[col] = scaler.transform(df[[col]])
            except:
                pass
    return df

def save_scaler_dict(scaler_dict, path=None):
    if path is None:
        path = MODELS_DIR / "scalerDict.joblib"
    joblib.dump(scaler_dict, path)
    return f"Saved scalerDict to {path}"

def load_scaler_dict(path=None):
    if path is None:
        path = MODELS_DIR / "scalerDict.joblib"
    return joblib.load(path)

def save_column_names(columns, path=None):
    if path is None:
        path = MODELS_DIR / "columnName.joblib"
    joblib.dump(columns, path)
    return f"Saved columns to {path}"

def load_column_names(path=None):
    if path is None:
        path = MODELS_DIR / "columnName.joblib"
    return joblib.load(path)