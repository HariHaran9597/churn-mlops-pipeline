"""
MLOps Orchestration: Check drift → Retrain if needed.
Works both as a standalone script AND with Prefect server.
"""
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

USE_PREFECT = os.getenv("USE_PREFECT", "false").lower() == "true"

if USE_PREFECT:
    from prefect import flow, task
else:
    # Fallback: plain Python decorators that just run the function
    def task(**kwargs):
        def wrapper(fn):
            return fn
        return wrapper
    def flow(**kwargs):
        def wrapper(fn):
            return fn
        return wrapper


@task(name="check_drift", retries=2, retry_delay_seconds=10)
def check_drift_task() -> bool:
    from src.drift.monitor import check_drift
    return check_drift()


@task(name="retrain_model", retries=1)
def retrain_model_task():
    from src.training.train import train
    train()


@flow(name="churn-drift-correction-flow")
def drift_correction_flow():
    """
    Self-healing MLOps pipeline:
    1. Check if incoming data has drifted from training baseline
    2. If drift detected → retrain model automatically
    3. New model overwrites old → next API restart loads updated model
    """
    logger.info("=" * 50)
    logger.info("STARTING DRIFT CORRECTION FLOW")
    logger.info("=" * 50)

    drift_detected = check_drift_task()

    if drift_detected:
        logger.info("⚠️  DRIFT DETECTED — triggering automatic retraining...")
        retrain_model_task()
        logger.info("✅ Model retrained and saved successfully.")
        logger.info("🔄 Restart API server to load new model weights.")
    else:
        logger.info("✅ No drift detected. Model is healthy. No action needed.")

    return drift_detected


if __name__ == "__main__":
    drift_correction_flow()
