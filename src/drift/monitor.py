"""
Evidently AI drift detection (v0.7.x API).
Compares recent incoming customer records against training baseline.
"""
import sys
import os
import logging
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from src.database.db import get_recent_records

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REFERENCE_DATA_PATH = "data/telco_churn.csv"
DRIFT_FEATURES = [
    'tenure', 'MonthlyCharges', 'TotalCharges',
    'Contract', 'PaymentMethod', 'InternetService'
]
MIN_RECORDS_FOR_MONITORING = 50
DRIFT_SHARE_THRESHOLD = 0.5  # 50%+ columns drifted = dataset drift


def load_reference() -> pd.DataFrame:
    df = pd.read_csv(REFERENCE_DATA_PATH)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    return df[DRIFT_FEATURES]


def check_drift() -> bool:
    recent = get_recent_records(n=200)

    if len(recent) < MIN_RECORDS_FOR_MONITORING:
        logger.warning(
            f"Only {len(recent)} records in DB. "
            f"Need {MIN_RECORDS_FOR_MONITORING} to detect drift. Skipping."
        )
        return False

    current_df = pd.DataFrame(recent)
    current_df.rename(columns={
        'monthly_charges': 'MonthlyCharges',
        'total_charges': 'TotalCharges',
        'contract': 'Contract',
        'payment_method': 'PaymentMethod',
        'internet_service': 'InternetService'
    }, inplace=True)
    current_df = current_df[DRIFT_FEATURES]

    reference_df = load_reference()

    # Run Evidently drift report
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=reference_df, current_data=current_df)

    # Save HTML report
    os.makedirs("monitoring", exist_ok=True)
    snapshot.save_html("monitoring/drift_report.html")

    # Extract drift results from snapshot dict
    result = snapshot.dict()
    metrics = result.get('metrics', [])

    n_drifted = 0
    drift_share = 0.0

    for m in metrics:
        name = m.get('metric_name', '')
        if 'DriftedColumnsCount' in name:
            n_drifted = int(m['value'].get('count', 0))
            drift_share = m['value'].get('share', 0.0)
            break

    drift_detected = drift_share >= DRIFT_SHARE_THRESHOLD

    logger.info(
        f"Drift detected: {drift_detected} | "
        f"Drifted columns: {n_drifted}/{len(DRIFT_FEATURES)} "
        f"(share: {drift_share:.1%})"
    )
    return drift_detected


if __name__ == "__main__":
    drift = check_drift()
    sys.exit(1 if drift else 0)
