"""
Churn model training script.
Called by Prefect when drift is detected to retrain on latest data.
"""
import os
import pickle
import logging
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Make sure models dir exists
os.makedirs("src/models", exist_ok=True)

DATA_PATH = os.getenv("DATA_PATH", "data/telco_churn.csv")
MODEL_PATH = "src/models/churn_model.pkl"
PREPROCESSOR_PATH = "src/models/preprocessor.pkl"


def load_and_engineer(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
        
    df = pd.read_csv(path)

    # Clean data
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    df.drop(columns=['customerID'], errors='ignore', inplace=True)

    # Build Engineered features
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['total_services'] = df[service_cols].apply(
        lambda row: sum(v == 'Yes' for v in row), axis=1
    )

    df['tenure_bucket'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, 72, np.inf],
        labels=[0, 1, 2, 3, 4],
        include_lowest=True
    ).astype(int)

    df['charge_per_service'] = df['MonthlyCharges'] / (df['total_services'] + 1)
    df['customer_value'] = df['tenure'] * df['MonthlyCharges']
    df['has_premium'] = (df['total_services'] >= 3).astype(int)

    return df


def encode_features(df: pd.DataFrame):
    cat_cols = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    num_cols = [
        'tenure', 'MonthlyCharges', 'TotalCharges',
        'total_services', 'tenure_bucket', 'charge_per_service',
        'customer_value', 'has_premium', 'SeniorCitizen'
    ]

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    feature_cols = cat_cols + num_cols
    X = df[feature_cols]
    y = df['Churn']

    return X, y, encoders, scaler, feature_cols


def train():
    logger.info(f"Loading data from {DATA_PATH}...")
    df = load_and_engineer(DATA_PATH)
    X, y, encoders, scaler, feature_cols = encode_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Train XGBoost
    model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train_res, y_train_res)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    
    logger.info(f"Training Complete. AUC: {auc:.4f} | F1: {f1:.4f}")

    # Save artifacts
    preprocessor = {
        'encoders': encoders,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'num_cols': [
            'tenure', 'MonthlyCharges', 'TotalCharges',
            'total_services', 'tenure_bucket', 'charge_per_service',
            'customer_value', 'has_premium', 'SeniorCitizen'
        ]
    }

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(PREPROCESSOR_PATH, 'wb') as f:
        pickle.dump(preprocessor, f)
        
    logger.info(f"Saved artifacts to {MODEL_PATH} and {PREPROCESSOR_PATH}")

if __name__ == "__main__":
    train()
