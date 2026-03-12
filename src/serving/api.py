"""
FastAPI inference endpoint for Churn Prediction.
Logs every request to PostgreSQL for drift monitoring.
"""
import os
import pickle
import logging
import numpy as np
import pandas as pd
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from prometheus_fastapi_instrumentator import Instrumentator
from src.database.db import get_db, CustomerRecord, create_tables

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Churn Prediction API", version="2.0")
Instrumentator().instrument(app).expose(app)

# Load model artifacts once at startup
MODEL_PATH = os.getenv("MODEL_PATH", "src/models/churn_model.pkl")
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", "src/models/preprocessor.pkl")

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
with open(PREPROCESSOR_PATH, 'rb') as f:
    preprocessor = pickle.load(f)

create_tables()

# ── Request Schema ─────────────────────────────────────────
class CustomerFeatures(BaseModel):
    gender: str = "Male"
    SeniorCitizen: int = 0
    Partner: str = "Yes"
    Dependents: str = "No"
    tenure: float = 12
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "Fiber optic"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 70.0
    TotalCharges: float = 840.0

def preprocess(customer: CustomerFeatures) -> pd.DataFrame:
    data = customer.model_dump()
    df = pd.DataFrame([data])
    
    # Engineer features (same as training)
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['total_services'] = df[service_cols].apply(
        lambda row: sum(v == 'Yes' for v in row), axis=1
    )
    df['tenure_bucket'] = pd.cut(
        df['tenure'], bins=[0, 12, 24, 48, 72, np.inf],
        labels=[0, 1, 2, 3, 4], include_lowest=True
    ).astype(int)
    df['charge_per_service'] = df['MonthlyCharges'] / (df['total_services'] + 1)
    df['customer_value'] = df['tenure'] * df['MonthlyCharges']
    df['has_premium'] = (df['total_services'] >= 3).astype(int)

    # Apply saved encoders
    encoders = preprocessor['encoders']
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str)
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
            df[col] = le.transform(df[col])

    # Scale numerics
    scaler = preprocessor['scaler']
    num_cols = preprocessor['num_cols']
    df[num_cols] = scaler.transform(df[num_cols])

    return df[preprocessor['feature_cols']]

# ── Endpoints ──────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": "XGBoost Churn v2.0"}

@app.post("/predict")
def predict(customer: CustomerFeatures, db: Session = Depends(get_db)):
    X = preprocess(customer)
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])

    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    total_services = sum(
        getattr(customer, col) == 'Yes' for col in service_cols
    )

    # Log to DB for drift monitoring
    record = CustomerRecord(
        tenure=customer.tenure,
        monthly_charges=customer.MonthlyCharges,
        total_charges=customer.TotalCharges,
        contract=customer.Contract,
        payment_method=customer.PaymentMethod,
        internet_service=customer.InternetService,
        total_services=total_services,
        customer_value=customer.tenure * customer.MonthlyCharges,
        churn_prediction=prediction,
        churn_probability=probability
    )
    db.add(record)
    db.commit()

    return {
        "churn_prediction": prediction,
        "churn_label": "CHURN" if prediction == 1 else "RETAIN",
        "churn_probability": round(probability, 4),
        "risk_tier": "HIGH" if probability > 0.7 else "MEDIUM" if probability > 0.4 else "LOW"
    }
