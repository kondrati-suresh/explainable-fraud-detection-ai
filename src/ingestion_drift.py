"""
Phase 1: Multi-source Ingestion Simulation & Schema-Drift Detector
"""
import numpy as np
import pandas as pd
from src.config import SEED, EXPECTED_TRANSACTION_SCHEMA

class SchemaDriftDetector:
    def __init__(self, expected_schema: dict):
        self.expected_schema = expected_schema
        self.drift_log = []

    def inspect_and_validate(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        validated_df = df.copy()
        missing_cols = set(self.expected_schema.keys()) - set(df.columns)
        unexpected_cols = set(df.columns) - set(self.expected_schema.keys())

        if missing_cols:
            self.drift_log.append({"source": source_name, "type": "MISSING_COLUMNS", "details": list(missing_cols), "severity": "CRITICAL"})
            for col in missing_cols:
                default_val = False if "bool" in self.expected_schema.get(col, "str") else 0
                validated_df[col] = default_val

        if unexpected_cols:
            self.drift_log.append({"source": source_name, "type": "UNEXPECTED_COLUMNS_DETECTED", "details": list(unexpected_cols), "severity": "WARNING"})

        null_fractions = validated_df.isnull().mean()
        high_nulls = null_fractions[null_fractions > 0.20].to_dict()
        if high_nulls:
            self.drift_log.append({"source": source_name, "type": "ANOMALOUS_NULL_SPIKE", "details": high_nulls, "severity": "HIGH"})
            validated_df = validated_df.fillna(0)

        return validated_df

def generate_synthetic_transactions(n_records: int = 1000) -> pd.DataFrame:
    np.random.seed(SEED % (2**32))
    channels = ["CARD", "UPI", "LOAN"]
    age_groups = ["18-25", "26-45", "46-60", "60+"]
    regions = ["North", "South", "East", "West"]
    
    user_avg_amounts = np.random.uniform(500, 5000, n_records)
    fraud_flags = np.random.choice([False, True], size=n_records, p=[0.88, 0.12])
    
    amounts, new_devices, foreign, velocities, failed_logins, ip_mismatches = [], [], [], [], [], []
    for i, is_fraud in enumerate(fraud_flags):
        avg = user_avg_amounts[i]
        if is_fraud:
            amounts.append(round(avg * np.random.uniform(3.5, 10.0), 2))
            new_devices.append(np.random.choice([True, False], p=[0.75, 0.25]))
            foreign.append(np.random.choice([True, False], p=[0.60, 0.40]))
            velocities.append(int(np.random.poisson(lam=5) + 3))
            failed_logins.append(int(np.random.choice([0, 1, 3, 5], p=[0.1, 0.2, 0.4, 0.3])))
            ip_mismatches.append(np.random.choice([True, False], p=[0.70, 0.30]))
        else:
            amounts.append(round(avg * np.random.uniform(0.2, 2.5), 2))
            new_devices.append(np.random.choice([True, False], p=[0.10, 0.90]))
            foreign.append(np.random.choice([True, False], p=[0.05, 0.95]))
            velocities.append(int(np.random.poisson(lam=1)))
            failed_logins.append(int(np.random.choice([0, 1, 2], p=[0.85, 0.12, 0.03])))
            ip_mismatches.append(np.random.choice([True, False], p=[0.08, 0.92]))

    df = pd.DataFrame({
        "txn_id": [f"TXN_{100000 + i}" for i in range(n_records)],
        "user_id": [f"USR_{np.random.randint(1000, 9999)}" for _ in range(n_records)],
        "amount": amounts,
        "avg_amount": np.round(user_avg_amounts, 2),
        "channel": np.random.choice(channels, size=n_records, p=[0.45, 0.45, 0.10]),
        "is_new_device": new_devices,
        "is_foreign": foreign,
        "velocity_1h": velocities,
        "failed_logins": failed_logins,
        "ip_mismatch": ip_mismatches,
        "age_group": np.random.choice(age_groups, size=n_records, p=[0.25, 0.40, 0.20, 0.15]),
        "region": np.random.choice(regions, size=n_records),
        "is_actual_fraud": fraud_flags
    })
    return df
