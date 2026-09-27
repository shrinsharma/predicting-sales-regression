# src/predict.py - EXTENDED GradientBoosting 15 features BEST MODEL
import joblib
import pandas as pd
from src.preprocessing import preprocess_input
from src.feature_engineering import load_encoder

def predict_sales(input_dict: dict):
    # BEST MODEL - 15 features, R2 0.6601
    model = joblib.load("models/model_step6_gradientboosting.pkl")
    encoder = load_encoder("models/encoder.joblib")

    df = preprocess_input(input_dict)
    # Transform StateHoliday -> StateHolidayEncoded + StateHoliday (both)
    from src.feature_engineering import transform_state_holiday
    df = transform_state_holiday(encoder, df)

    # Drop leakage / raw cols
    for col in ['Date', 'Sales', 'Customers']:
        if col in df.columns:
            df = df.drop(columns=[col])

    # For GB, StateHoliday raw should NOT be used, use StateHolidayEncoded
    # Model expects 15 cols, NOT StateHoliday
    expected = list(getattr(model, 'feature_names_in_', []))
    if not expected:
        expected = ['Store','DayOfWeek','Open','Promo','SchoolHoliday','Year','Month','Day','WeekOfYear','Quarter','IsWeekend','IsMonthStart','IsMonthEnd','IsStateHoliday','StateHolidayEncoded']

    # Ensure all expected present
    for col in expected:
        if col not in df.columns:
            df[col] = 0

    df = df[expected]

    # Tree model - NO scaler needed (was trained without scalerDict 10-col)
    prediction = model.predict(df)[0]

    if input_dict.get('Open', 1) == 0 or str(input_dict.get('Open')) == '0':
        prediction = 0.0

    return float(prediction)