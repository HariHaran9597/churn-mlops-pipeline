# scripts/populate_db.py
"""
Seeds the PostgreSQL database with customer records from the telco churn dataset.
This provides a baseline of production-like data for drift monitoring.

Usage:
    docker exec -it drift_ml_app python scripts/populate_db.py
    # or locally:
    python scripts/populate_db.py
"""
import sys
import os
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.database.db import engine, create_tables, CustomerRecord, SessionLocal

DATA_PATH = os.getenv("DATA_PATH", "data/telco_churn.csv")
SEED_COUNT = 200  # Number of records to seed


def seed_from_csv(path: str, n: int = SEED_COUNT):
    """
    Reads the telco churn CSV and inserts N sampled records into the
    customer_records table — the same table the /predict endpoint logs to.
    This gives drift monitoring a baseline of realistic production data.
    """
    import pandas as pd

    if not os.path.exists(path):
        logger.error(f"Dataset not found at '{path}'. "
                     f"Make sure telco_churn.csv is in the data/ directory.")
        sys.exit(1)

    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    # Sample N records
    sample = df.sample(n=min(n, len(df)), random_state=42)

    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]

    create_tables()
    db = SessionLocal()

    try:
        inserted = 0
        for _, row in sample.iterrows():
            total_services = sum(row[col] == "Yes" for col in service_cols)
            customer_value = row["tenure"] * row["MonthlyCharges"]

            record = CustomerRecord(
                tenure=float(row["tenure"]),
                monthly_charges=float(row["MonthlyCharges"]),
                total_charges=float(row["TotalCharges"]),
                contract=row["Contract"],
                payment_method=row["PaymentMethod"],
                internet_service=row["InternetService"],
                total_services=total_services,
                customer_value=customer_value,
                churn_prediction=1 if row["Churn"] == "Yes" else 0,
                churn_probability=round(random.uniform(0.3, 0.9), 4)
                if row["Churn"] == "Yes"
                else round(random.uniform(0.05, 0.4), 4),
            )
            db.add(record)
            inserted += 1

        db.commit()
        logger.info(f"✓ Seeded {inserted} customer records into PostgreSQL.")
    except Exception as e:
        db.rollback()
        logger.error(f"✗ Failed to seed database: {e}")
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    seed_from_csv(DATA_PATH)