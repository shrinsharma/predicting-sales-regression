# src/feature_engineering.py - EXTENDED handles LabelEncoder dict -> StateHolidayEncoded
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

def create_encoder(df: pd.DataFrame):
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    if 'StateHoliday' in df.columns:
        encoder.fit(df[['StateHoliday']])
    return encoder

def transform_state_holiday(encoder, df: pd.DataFrame) -> pd.DataFrame:
    """Creates StateHolidayEncoded (for GB 15 feat) + keeps StateHoliday (for RF 10 feat)"""
    df = df.copy()
    if 'StateHoliday' not in df.columns or encoder is None:
        return df
    # Get LabelEncoder from dict or direct
    if isinstance(encoder, dict):
        le = encoder.get('StateHoliday', next(iter(encoder.values())))
    else:
        le = encoder

    if isinstance(le, LabelEncoder):
        encoded_vals = le.transform(df['StateHoliday'].astype(str))
        # For GB 15-feature model
        df['StateHolidayEncoded'] = encoded_vals
        # For backward compat 10-feature models, keep StateHoliday as encoded int too
        df['StateHoliday'] = encoded_vals
    else:
        # OneHot fallback
        if hasattr(le, 'get_feature_names_out'):
            encoded = le.transform(df[['StateHoliday']])
            cols = le.get_feature_names_out(['StateHoliday'])
            enc_df = pd.DataFrame(encoded, columns=cols, index=df.index)
            df = df.drop(columns=['StateHoliday'])
            df = pd.concat([df, enc_df], axis=1)
    return df

def save_encoder(encoder, path="models/encoder.joblib"):
    joblib.dump(encoder, path)
    return f"Saved encoder to {path}"

def load_encoder(path="models/encoder.joblib"):
    return joblib.load(path)

def create_scaler_dict(df: pd.DataFrame):
    scaler_dict = {}
    numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    exclude = ['Sales']
    numeric_cols = [c for c in numeric_cols if c not in exclude and not c.startswith('StateHoliday_')]
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

def save_scaler_dict(scaler_dict, path="models/scalerDict.joblib"):
    joblib.dump(scaler_dict, path)
    return f"Saved scalerDict to {path}"

def load_scaler_dict(path="models/scalerDict.joblib"):
    return joblib.load(path)

def save_column_names(columns, path="models/columnName.joblib"):
    joblib.dump(columns, path)
    return f"Saved columns to {path}"

def load_column_names(path="models/columnName.joblib"):
    return joblib.load(path)

def create_features(df: pd.DataFrame, encoder=None, scaler_dict=None, is_train=True):
    df = df.copy()
    if is_train:
        encoder = create_encoder(df)
        df = transform_state_holiday(encoder, df)
        scaler_dict = create_scaler_dict(df)
        df = transform_scaler(scaler_dict, df)
        columns = df.columns.tolist()
        if 'Sales' in columns:
            columns.remove('Sales')
        return df, encoder, scaler_dict, columns
    else:
        df = transform_state_holiday(encoder, df)
        df = transform_scaler(scaler_dict, df)
        return df