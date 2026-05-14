# 🔄 Telecom Churn Prediction with Self-Healing MLOps Pipeline

> An end-to-end ML system that predicts telecom customer churn using XGBoost, automatically detects data drift in production traffic using Evidently AI, and triggers self-healing model retraining — fully containerized with Docker.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)](https://docker.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io)
[![Evidently](https://img.shields.io/badge/Evidently_AI-0.7-purple)](https://evidentlyai.com)
[![Prefect](https://img.shields.io/badge/Prefect-2.14-blue)](https://prefect.io)

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Pipeline Flow](#-pipeline-flow)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [API Documentation](#-api-documentation)
- [Drift Detection & Auto-Retraining](#-drift-detection--auto-retraining)
- [Model Performance](#-model-performance)
- [Feature Engineering](#-feature-engineering)
- [Monitoring](#-monitoring)
- [Key Design Decisions](#-key-design-decisions)

---

## 🎯 Problem Statement

Telecom companies lose **$65B annually** to customer churn. A static ML model deployed once will silently degrade as customer behavior shifts — new pricing plans, competitor launches, and seasonal patterns cause **data drift** that erodes prediction accuracy over time.

**This project solves two problems:**
1. **Churn Prediction** — Identify at-risk customers before they leave using XGBoost with SMOTE for class imbalance
2. **Model Reliability** — Automatically detect when production data drifts from training data and retrain the model. Retraining saves a new model artifact; an API restart loads the latest version.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DOCKER-COMPOSE STACK                        │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │   Client /    │    │   FastAPI     │    │    PostgreSQL        │  │
│  │   Traffic     │───▶│   /predict    │───▶│    (Feature Store)   │  │
│  │   Simulator   │    │   /health     │    │    customer_records  │  │
│  └──────────────┘    └──────┬───────┘    └──────────┬───────────┘  │
│                             │                        │              │
│                             │ Prometheus             │              │
│                             │ /metrics               │              │
│                             ▼                        ▼              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  Prometheus   │    │   Evidently   │◀──│  Reference Dataset   │  │
│  │  (Metrics)    │    │   Drift       │    │  (telco_churn.csv)   │  │
│  └──────┬───────┘    │   Monitor     │    └──────────────────────┘  │
│         │            └──────┬───────┘                               │
│         ▼                   │ Drift Detected?                       │
│  ┌──────────────┐           ▼                                       │
│  │   Grafana     │    ┌──────────────┐    ┌──────────────────────┐  │
│  │  (Dashboard)  │    │   Prefect     │───▶│   XGBoost Retrain   │  │
│  └──────────────┘    │   Flow        │    │   + MLflow Tracking  │  │
│                      └──────────────┘    └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **ML Model** | XGBoost + SMOTE | Churn classification with class imbalance handling |
| **Experiment Tracking** | MLflow | Track AUC, F1, model versions across retraining cycles |
| **Serving** | FastAPI + Uvicorn | REST API for real-time churn predictions |
| **Drift Detection** | Evidently AI | Statistical drift monitoring (PSI/KS tests) on production data |
| **Orchestration** | Prefect | Coordinates drift check → retrain → deploy pipeline |
| **Feature Store** | PostgreSQL | Logs every prediction request for drift analysis |
| **Monitoring** | Prometheus + Grafana | API latency, request count, error rate dashboards |
| **Infrastructure** | Docker Compose | One-command deployment of the entire 5-service stack |
| **Data Processing** | Pandas, Scikit-learn | Feature engineering, encoding, scaling |

---

## 🔄 Pipeline Flow

```
                    ┌─────────────────────┐
                    │  Customer Request    │
                    │  (21 raw features)   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Feature Engineering │
                    │  • total_services    │
                    │  • customer_value    │
                    │  • tenure_bucket     │
                    │  • charge_per_service│
                    │  • has_premium       │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  XGBoost Prediction  │──────▶  Response: {
                    │  (SMOTE-balanced)    │           churn_label: "CHURN",
                    └──────────┬──────────┘           probability: 0.78,
                               │                      risk_tier: "HIGH"
                               │                    }
                    ┌──────────▼──────────┐
                    │  Log to PostgreSQL   │
                    │  (for drift monitor) │
                    └──────────┬──────────┘
                               │
              ┌────────────────▼────────────────┐
              │     DRIFT CORRECTION FLOW        │
              │   (Runs on schedule / manual)     │
              │                                   │
              │  1. Fetch last 200 records from DB│
              │  2. Compare vs training baseline  │
              │  3. If drift_share ≥ 50%:         │
              │     → Retrain XGBoost             │
              │     → Save new model.pkl          │
              │     → Log to MLflow               │
              └──────────────────────────────────┘
```

---

## 📁 Project Structure

```
churn-mlops-pipeline/
│
├── docker-compose.yml              # Full 5-service stack
├── Dockerfile                      # ML App image (Python 3.11-slim)
├── requirements.txt                # All dependencies
│
├── data/
│   └── telco_churn.csv             # IBM Telco dataset (7,043 records)
│
├── monitoring/
│   ├── prometheus.yml              # Prometheus scrape config
│   └── drift_report.html           # Auto-generated Evidently report
│
├── scripts/
│   ├── generate_traffic.py         # Traffic simulator (normal + drift modes)
│   └── populate_db.py              # Database seeding script
│
└── src/
    ├── database/
    │   └── db.py                   # SQLAlchemy models + PostgreSQL connection
    ├── drift/
    │   └── monitor.py              # Evidently AI drift detection
    ├── models/
    │   ├── churn_model.pkl          # Trained XGBoost model
    │   └── preprocessor.pkl         # LabelEncoders + StandardScaler
    ├── orchestration/
    │   └── flow.py                 # Prefect drift → retrain pipeline
    ├── serving/
    │   └── api.py                  # FastAPI prediction endpoint
    └── training/
        └── train.py                # XGBoost + SMOTE training with MLflow
```

---

## 🚀 Quick Start

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- Git

### 1. Clone & Start

```bash
git clone https://github.com/HariHaran9597/churn-mlops-pipeline.git
cd churn-mlops-pipeline

# Set up environment variables
cp .env.example .env   # Edit .env if you need custom credentials

# Train the model (generates churn_model.pkl + logs to MLflow)
pip install -r requirements.txt
python src/training/train.py

# Start the full stack (FastAPI + PostgreSQL + Prometheus + Grafana)
docker-compose up --build -d
```

### 2. Test the API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": 0,
    "Partner": "Yes", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.0, "TotalCharges": 170.0
  }'
```

**Response:**
```json
{
  "churn_prediction": 1,
  "churn_label": "CHURN",
  "churn_probability": 0.7823,
  "risk_tier": "HIGH"
}
```

### 3. Simulate Drift & Auto-Retrain

```bash
# Send 60 biased requests (simulates a telecom pricing change)
python scripts/generate_traffic.py --mode drift --n 60

# Run the self-healing pipeline
docker exec -it drift_ml_app python -m src.orchestration.flow
```

**Output:**
```
STARTING DRIFT CORRECTION FLOW
Drift detected: True | Drifted columns: 6/6 (share: 100.0%)
⚠️  DRIFT DETECTED — triggering automatic retraining...
Training Complete. AUC: 0.8344 | F1: 0.6249 | Recall: 0.8150
✅ Model retrained and saved successfully.
✅ MLflow run logged to experiment 'churn-xgboost'
🔄 Restart API server to load new model weights.
```

> **Note:** Retraining saves a new `churn_model.pkl` artifact to disk. The API loads the model once at startup, so a container restart is required to serve the updated model:
> ```bash
> docker-compose restart ml-app
> ```

---

## 📡 API Documentation

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Predict churn for a single customer |
| `/health` | GET | API health check + model version |
| `/metrics` | GET | Prometheus metrics (latency, requests) |
| `/docs` | GET | Interactive Swagger documentation |

### Request Schema (`/predict`)

| Field | Type | Example |
|---|---|---|
| `gender` | string | "Male" / "Female" |
| `SeniorCitizen` | int | 0 / 1 |
| `tenure` | float | 24 (months) |
| `Contract` | string | "Month-to-month" / "One year" / "Two year" |
| `MonthlyCharges` | float | 85.0 |
| `TotalCharges` | float | 2040.0 |
| `InternetService` | string | "DSL" / "Fiber optic" / "No" |
| ... | ... | (21 total features — see `/docs` for full schema) |

### Response

```json
{
  "churn_prediction": 1,
  "churn_label": "CHURN",
  "churn_probability": 0.7823,
  "risk_tier": "HIGH"
}
```

Risk tiers: `HIGH` (>70%), `MEDIUM` (40-70%), `LOW` (<40%)

---

## 🔍 Drift Detection & Auto-Retraining

### Why Drift Matters

A churn model trained today will degrade silently when:
- The telecom launches a **new pricing plan** (MonthlyCharges distribution shifts)
- A **competitor enters the market** (Contract type distribution skews to Month-to-month)
- **Seasonal patterns** change customer behavior (tenure distributions shift)

### How We Detect It

Evidently AI compares the last 200 production requests (from PostgreSQL) against the original training data baseline using statistical tests:

| Feature | Test Used | Threshold |
|---|---|---|
| `tenure` | Kolmogorov-Smirnov | p < 0.05 |
| `MonthlyCharges` | Kolmogorov-Smirnov | p < 0.05 |
| `TotalCharges` | Kolmogorov-Smirnov | p < 0.05 |
| `Contract` | Chi-Square | p < 0.05 |
| `PaymentMethod` | Chi-Square | p < 0.05 |
| `InternetService` | Chi-Square | p < 0.05 |

**Dataset drift** is flagged when **≥50%** of monitored features have drifted.

### Traffic Simulator

```bash
# Normal traffic (realistic distribution)
python scripts/generate_traffic.py --mode normal --n 100

# Drifted traffic (simulates market disruption)
python scripts/generate_traffic.py --mode drift --n 100
```

The drift simulator injects bias: all customers switch to Month-to-month contracts, Fiber optic internet, and higher charges — mimicking a real-world competitive pricing war.

---

## 📊 Model Performance

### Best Model: XGBoost + SMOTE

| Metric | Value |
|---|---|
| **AUC-ROC** | 0.835 |
| **F1 Score** | 0.625 |
| **Recall** | 0.815 |
| **Precision** | 0.505 |
| **Accuracy** | 0.735 |

### Model Comparison

| Model | AUC | F1 | Recall | Why Not Chosen |
|---|---|---|---|---|
| Logistic Regression | 0.840 | 0.581 | 0.532 | Low recall — misses 47% of churners |
| XGBoost (no SMOTE) | 0.833 | 0.617 | 0.762 | Better, but still misses 24% of churners |
| **XGBoost + SMOTE** | **0.835** | **0.625** | **0.815** | **Catches 81.5% of churners** ✅ |

> **Design Decision:** We optimized for **Recall** over Accuracy. Missing a churning customer (False Negative) costs the business more than a false alarm (False Positive). With 81.5% recall, we identify 4 out of 5 at-risk customers.

---

## ⚙️ Feature Engineering

Beyond raw features from the IBM Telco dataset, we engineered 5 business-contextual features:

| Feature | Formula | Business Meaning |
|---|---|---|
| `total_services` | Count of "Yes" in service columns | How many products the customer uses |
| `tenure_bucket` | Binned tenure (0-12, 12-24, ...) | Customer lifecycle stage |
| `charge_per_service` | MonthlyCharges / (total_services + 1) | Value-for-money ratio |
| `customer_value` | tenure × MonthlyCharges | Estimated lifetime value |
| `has_premium` | total_services ≥ 3 | Premium customer indicator |

---

## 📈 Monitoring

### Prometheus Metrics (port 9090)

The FastAPI app exposes real-time metrics via `prometheus-fastapi-instrumentator`:
- `http_request_duration_seconds` — API latency histogram
- `http_requests_total` — Total request count by status code
- `http_request_size_bytes` — Request payload sizes

### Grafana Dashboards (port 3000)

Access Grafana at `http://localhost:3000` (default credentials: `admin/admin`).

### Evidently Drift Report

After running `monitor.py`, a comprehensive HTML drift report is saved to `monitoring/drift_report.html` with per-feature distribution comparisons and statistical test results.

---

## 🧠 Key Design Decisions

| Decision | Rationale |
|---|---|
| **XGBoost + SMOTE over Logistic Regression** | 81.5% recall vs 53.2% — business cost of missing churners outweighs false positives |
| **PostgreSQL as feature store** | Every prediction is logged; drift detection queries the last N records — simulates production data warehousing |
| **Evidently over custom drift scripts** | Industry-standard library; generates publication-quality reports; supports KS, PSI, Chi-square tests out of the box |
| **Prefect over cron jobs** | Retry logic, task dependencies, observability dashboard — production-grade orchestration |
| **Docker Compose over single container** | Decoupled services (API, DB, monitoring) — mirrors real microservice architecture |
| **SMOTE over class weights** | Generates synthetic minority samples rather than just reweighting loss — produces better decision boundaries for imbalanced data |

---

## 🛣️ Future Improvements

- [ ] Add Grafana dashboard JSON (auto-provisioned)
- [ ] Implement A/B testing between old and retrained models
- [ ] Add Prefect scheduled deployments (e.g., drift check every 6 hours)
- [ ] Add model versioning with MLflow Model Registry
- [ ] Hot-reload model artifacts at runtime (avoid API restart after retraining)
- [ ] Deploy FastAPI on cloud (AWS ECS / GCP Cloud Run)

---

## 📜 License

This project is open source under the [MIT License](LICENSE).

---

<p align="center">
  Built with ❤️ by <a href="https://github.com/HariHaran9597">Hari Haran</a>
</p>
