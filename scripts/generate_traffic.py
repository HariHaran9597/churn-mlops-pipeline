"""
Generates fake customer traffic to test drift detection.
"""
import random
import argparse
import httpx
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_URL = "http://localhost:8000/predict"

def normal_customer() -> dict:
    return {
        "gender": random.choice(["Male", "Female"]),
        "SeniorCitizen": random.choices([0, 1], weights=[84, 16])[0],
        "Partner": random.choice(["Yes", "No"]),
        "Dependents": random.choice(["Yes", "No"]),
        "tenure": random.randint(1, 72),
        "PhoneService": random.choice(["Yes", "No"]),
        "MultipleLines": random.choice(["Yes", "No", "No phone service"]),
        "InternetService": random.choices(["DSL", "Fiber optic", "No"], weights=[34, 44, 22])[0],
        "OnlineSecurity": random.choice(["Yes", "No", "No internet service"]),
        "OnlineBackup": random.choice(["Yes", "No", "No internet service"]),
        "DeviceProtection": random.choice(["Yes", "No", "No internet service"]),
        "TechSupport": random.choice(["Yes", "No", "No internet service"]),
        "StreamingTV": random.choice(["Yes", "No", "No internet service"]),
        "StreamingMovies": random.choice(["Yes", "No", "No internet service"]),
        "Contract": random.choices(["Month-to-month", "One year", "Two year"], weights=[55, 21, 24])[0],
        "PaperlessBilling": random.choice(["Yes", "No"]),
        "PaymentMethod": random.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]),
        "MonthlyCharges": round(random.uniform(18, 120), 2),
        "TotalCharges": round(random.uniform(18, 8000), 2),
    }

def drifted_customer() -> dict:
    customer = normal_customer()
    customer["Contract"] = "Month-to-month"
    customer["InternetService"] = "Fiber optic"
    customer["MonthlyCharges"] = round(random.uniform(90, 120), 2)
    return customer

def run(mode: str, n: int = 60):
    logger.info(f"Sending {n} requests in '{mode}' mode...")
    with httpx.Client(timeout=10) as client:
        for i in range(n):
            customer = drifted_customer() if mode == "drift" else normal_customer()
            try:
                resp = client.post(API_URL, json=customer)
                if resp.status_code == 200:
                    result = resp.json()
                    if i % 10 == 0:
                        logger.info(f"Req {i+1}: {result['churn_label']} (p={result['churn_probability']:.3f})")
            except Exception as e:
                logger.error(f"Error {i+1}: {e}")
            time.sleep(0.1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["normal", "drift"], default="normal")
    parser.add_argument("--n", type=int, default=60)
    args = parser.parse_args()
    run(args.mode, args.n)
