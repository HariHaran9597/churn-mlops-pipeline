import os
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://mlops:mlops@localhost:5432/mlops_db"
)

# Use pool_pre_ping for stability inside docker
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class CustomerRecord(Base):
    """Stores incoming customer prediction requests for drift monitoring."""
    __tablename__ = "customer_records"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Core numeric features (for drift detection)
    tenure = Column(Float, nullable=False)
    monthly_charges = Column(Float, nullable=False)
    total_charges = Column(Float, nullable=False)

    # Categorical features
    contract = Column(String(50))
    payment_method = Column(String(50))
    internet_service = Column(String(50))

    # Engineered features
    total_services = Column(Integer)
    customer_value = Column(Float)

    # Prediction output
    churn_prediction = Column(Integer)   # 0 = No Churn, 1 = Churn
    churn_probability = Column(Float)


def create_tables():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_recent_records(n: int = 200) -> list[dict]:
    """Fetch the most recent N customer records for drift detection."""
    with engine.connect() as conn:
        result = conn.execute(text(
            f"""SELECT tenure, monthly_charges, total_charges,
                       contract, payment_method, internet_service,
                       total_services, customer_value
               FROM customer_records
               ORDER BY timestamp DESC
               LIMIT {n}"""
        ))
        rows = result.fetchall()
        return [dict(row._mapping) for row in rows]
