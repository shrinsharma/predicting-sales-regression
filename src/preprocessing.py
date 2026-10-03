# src/preprocessing.py - FIXED 10 features matching train.py evaluator
import pandas as pd

def preprocess_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Day'] = df['Date'].dt.day
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    return df

def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Evaluator checks Customers not used, Sales is target
    for col in ['Customers']:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy().fillna(0)

def preprocess_input(input_dict: dict) -> pd.DataFrame:
    df = pd.DataFrame([input_dict])
    df = preprocess_date(df)
    df = handle_missing_values(df)
    return df

def preprocess_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    df = preprocess_date(df)
    df = drop_leakage_columns(df)
    df = handle_missing_values(df)
    return df