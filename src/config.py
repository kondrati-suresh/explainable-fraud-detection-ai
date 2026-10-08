"""
Configuration & Anti-Copy Seed Initialization (Batch 15)
"""
SEED = 2113025148045
SLA_HIGH_SEVERITY_MINUTES = 30
SLA_MEDIUM_SEVERITY_MINUTES = 120
SLA_LOW_SEVERITY_MINUTES = 360
VALID_DOMAINS = ["CARD", "UPI", "LOAN"]
EXPECTED_TRANSACTION_SCHEMA = {
    "txn_id": "str", "user_id": "str", "amount": "float", "avg_amount": "float",
    "channel": "str", "is_new_device": "bool", "is_foreign": "bool",
    "velocity_1h": "int", "failed_logins": "int", "ip_mismatch": "bool",
    "age_group": "str", "region": "str", "is_actual_fraud": "bool"
}
