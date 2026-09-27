# src/preprocessing.py - EXTENDED for GradientBoosting 15 features
import pandas as pd

def preprocess_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Day'] = df['Date'].dt.day
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
        df['Quarter'] = df['Date'].dt.quarter
        df['IsMonthStart'] = df['Date'].dt.is_month_start.astype(int)
        df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)
    if 'DayOfWeek' in df.columns:
        # Rossmann DayOfWeek 1=Mon.. 7=Sun, weekend = 6,7
        df['IsWeekend'] = (pd.to_numeric(df['DayOfWeek'], errors='coerce') >= 6).astype(int)
    if 'StateHoliday' in df.columns:
        df['IsStateHoliday'] = (df['StateHoliday'].astype(str)!= '0').astype(int)
    return df

def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'Customers' in df.columns:
        df = df.drop(columns=['Customers'])
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