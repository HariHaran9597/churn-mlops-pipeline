# test_api.py
"""
API integration tests for the Churn Prediction service.
Run with: python test_api.py (requires the stack to be up via docker-compose)
"""
import requests
import json
import sys

BASE_URL = "http://localhost:8000"

# ── Test Payloads ──────────────────────────────────────────
HIGH_RISK_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 1,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.0,
    "TotalCharges": 190.0,
}

LOW_RISK_CUSTOMER = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 60,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Credit card (automatic)",
    "MonthlyCharges": 55.0,
    "TotalCharges": 3300.0,
}

REQUIRED_RESPONSE_FIELDS = [
    "churn_prediction",
    "churn_label",
    "churn_probability",
    "risk_tier",
]

passed = 0
failed = 0


def test(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✓ {name}")
    else:
        failed += 1
        print(f"  ✗ {name}  →  {detail}")


# ── Test 1: Health Check ───────────────────────────────────
print("\n" + "=" * 60)
print("Test Suite: Churn Prediction API")
print("=" * 60)

print("\n[1] GET /health")
try:
    resp = requests.get(f"{BASE_URL}/health", timeout=5)
    test("Status 200", resp.status_code == 200, f"got {resp.status_code}")
    body = resp.json()
    test("Has 'status' field", "status" in body)
    test("Status is 'ok'", body.get("status") == "ok", f"got '{body.get('status')}'")
except requests.exceptions.ConnectionError:
    print("  ✗ Connection failed — is the API running? (docker-compose up -d)")
    sys.exit(1)

# ── Test 2: Predict — High Risk Customer ──────────────────
print("\n[2] POST /predict — High-risk customer (short tenure, no services)")
try:
    resp = requests.post(f"{BASE_URL}/predict", json=HIGH_RISK_CUSTOMER, timeout=5)
    test("Status 200", resp.status_code == 200, f"got {resp.status_code}")
    body = resp.json()

    for field in REQUIRED_RESPONSE_FIELDS:
        test(f"Response has '{field}'", field in body)

    test(
        "churn_prediction is 0 or 1",
        body.get("churn_prediction") in (0, 1),
        f"got {body.get('churn_prediction')}",
    )
    test(
        "churn_label is CHURN or RETAIN",
        body.get("churn_label") in ("CHURN", "RETAIN"),
        f"got '{body.get('churn_label')}'",
    )
    test(
        "churn_probability is between 0 and 1",
        0 <= body.get("churn_probability", -1) <= 1,
        f"got {body.get('churn_probability')}",
    )
    test(
        "risk_tier is HIGH, MEDIUM, or LOW",
        body.get("risk_tier") in ("HIGH", "MEDIUM", "LOW"),
        f"got '{body.get('risk_tier')}'",
    )

    print(f"  → Prediction: {body['churn_label']} "
          f"(p={body['churn_probability']:.3f}, tier={body['risk_tier']})")
except Exception as e:
    print(f"  ✗ Request failed: {e}")

# ── Test 3: Predict — Low Risk Customer ───────────────────
print("\n[3] POST /predict — Low-risk customer (long tenure, many services)")
try:
    resp = requests.post(f"{BASE_URL}/predict", json=LOW_RISK_CUSTOMER, timeout=5)
    test("Status 200", resp.status_code == 200, f"got {resp.status_code}")
    body = resp.json()

    for field in REQUIRED_RESPONSE_FIELDS:
        test(f"Response has '{field}'", field in body)

    print(f"  → Prediction: {body['churn_label']} "
          f"(p={body['churn_probability']:.3f}, tier={body['risk_tier']})")
except Exception as e:
    print(f"  ✗ Request failed: {e}")

# ── Test 4: Invalid Payload ───────────────────────────────
print("\n[4] POST /predict — Empty payload (should use defaults)")
try:
    resp = requests.post(f"{BASE_URL}/predict", json={}, timeout=5)
    test("Status 200 (defaults applied)", resp.status_code == 200, f"got {resp.status_code}")
    body = resp.json()
    test("Response still has all fields", all(f in body for f in REQUIRED_RESPONSE_FIELDS))
except Exception as e:
    print(f"  ✗ Request failed: {e}")

# ── Summary ───────────────────────────────────────────────
print("\n" + "=" * 60)
total = passed + failed
print(f"Results: {passed}/{total} passed, {failed} failed")
if failed == 0:
    print("✅ All tests passed!")
else:
    print("⚠️  Some tests failed — check output above.")
print("=" * 60 + "\n")

sys.exit(1 if failed > 0 else 0)