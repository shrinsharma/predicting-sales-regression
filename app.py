# app.py - FastAPI for Predicting Sales - mentor style same as Medicine Review
from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from src.predict import predict_sales

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

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

    prediction = predict_sales(input_dict)

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