"""
Layer 2: Knowledge & Reasoning Layer (FOL & Naive Bayes)
"""
from typing import Dict, List, Tuple
import numpy as np

class FOLRuleEngine:
    def __init__(self):
        self.rules = [
            {"id": "RULE_R1_HIGH_SURGE", "condition": lambda r: r["amount"] > (5.0 * r["avg_amount"]) and r["is_new_device"], "severity": "HIGH", "code": "RC_SURGE_NEW_DEV", "description": "Transaction exceeds 5x user average from new device."},
            {"id": "RULE_R2_CROSS_BORDER_VELOCITY", "condition": lambda r: r["is_foreign"] and r["velocity_1h"] >= 4, "severity": "HIGH", "code": "RC_GEO_VELOCITY", "description": "High transaction velocity from foreign geolocation."},
            {"id": "RULE_R3_ACCOUNT_TAKEOVER", "condition": lambda r: r["ip_mismatch"] and r["failed_logins"] >= 3, "severity": "MEDIUM", "code": "RC_ATO_PATTERN", "description": "Multiple failed logins combined with IP mismatch."},
            {"id": "RULE_R4_COERCION_SPIKE", "condition": lambda r: r["amount"] > (3.0 * r["avg_amount"]) and r["failed_logins"] >= 2, "severity": "MEDIUM", "code": "RC_CREDENTIAL_SPIKE", "description": "Value spike following repeated login failures."}
        ]

    def forward_chain(self, txn: Dict) -> Tuple[bool, List[Dict], List[str]]:
        fired_rules = []
        inference_trace = [f"Evaluating transaction {txn['txn_id']} against {len(self.rules)} FOL axioms..."]
        for rule in self.rules:
            if rule["condition"](txn):
                fired_rules.append(rule)
                inference_trace.append(f"  [FIRED] {rule['id']} :: {rule['description']}")
            else:
                inference_trace.append(f"  [SKIPPED] {rule['id']} condition not met.")
        is_suspicious = len(fired_rules) > 0
        inference_trace.append(f"Decision: {'SUSPICIOUS' if is_suspicious else 'LEGITIMATE'} ({len(fired_rules)} rules activated).")
        return is_suspicious, fired_rules, inference_trace

class NaiveBayesRiskEngine:
    def __init__(self):
        self.p_fraud = 0.12
        self.p_non_fraud = 0.88
        self.likelihoods = {
            "is_new_device": {True: (0.75, 0.10), False: (0.25, 0.90)},
            "is_foreign": {True: (0.60, 0.05), False: (0.40, 0.95)},
            "ip_mismatch": {True: (0.70, 0.08), False: (0.30, 0.92)},
            "high_velocity": {True: (0.80, 0.15), False: (0.20, 0.85)},
            "excess_amount": {True: (0.85, 0.08), False: (0.15, 0.92)},
            "failed_login_spike": {True: (0.70, 0.05), False: (0.30, 0.95)}
        }

    def compute_risk_score(self, txn: Dict) -> float:
        feat_state = {
            "is_new_device": bool(txn["is_new_device"]), "is_foreign": bool(txn["is_foreign"]),
            "ip_mismatch": bool(txn["ip_mismatch"]), "high_velocity": txn["velocity_1h"] >= 3,
            "excess_amount": txn["amount"] > (3.0 * txn["avg_amount"]), "failed_login_spike": txn["failed_logins"] >= 2
        }
        log_p_e_given_fraud = np.log(self.p_fraud)
        log_p_e_given_legit = np.log(self.p_non_fraud)
        for feat, val in feat_state.items():
            p_f, p_l = self.likelihoods[feat][val]
            log_p_e_given_fraud += np.log(p_f)
            log_p_e_given_legit += np.log(p_l)
        max_log = max(log_p_e_given_fraud, log_p_e_given_legit)
        p_fraud_num = np.exp(log_p_e_given_fraud - max_log)
        p_legit_num = np.exp(log_p_e_given_legit - max_log)
        return float(p_fraud_num / (p_fraud_num + p_legit_num))
