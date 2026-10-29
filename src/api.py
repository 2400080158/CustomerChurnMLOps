from fastapi.responses import FileResponse
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Customer Churn Prediction API")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"

model = joblib.load(MODEL_PATH)


class Customer(BaseModel):
    tenure: float
    monthly_charges: float
    total_charges: float
    support_calls: float
    contract_type: float
    payment_delay: float
    usage_score: float
    complaints: float
    age: float
    service_count: float


@app.get("/")
def home():
    return FileResponse(
        BASE_DIR / "src" / "static" / "index.html"
    )


@app.post("/predict")
def predict(customer: Customer):
    data = pd.DataFrame([customer.model_dump()])
    prediction = int(model.predict(data)[0])

    return {
        "churn_prediction": prediction,
        "result": "Likely to churn" if prediction == 1
                  else "Unlikely to churn"
    }
