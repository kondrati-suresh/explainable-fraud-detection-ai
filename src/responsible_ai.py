"""
Layer 3: Responsible AI & Explainable AI (XAI)
"""
from typing import Dict, List
import pandas as pd

class ResponsibleAIAuditor:
    @staticmethod
    def generate_xai_explanation(txn: Dict, fired_rules: List[Dict], bayes_score: float) -> str:
        status = "FLAGGED FOR REVIEW" if (fired_rules or bayes_score >= 0.50) else "APPROVED (LOW RISK)"
        explanation = [
            "=" * 65,
            f" [XAI FRAUD EXPLANATION AUDIT TRAIL] - ID: {txn['txn_id']}",
            f" Status: {status} | Combined Risk Score: {bayes_score:.4f}",
            f" Channel: {txn['channel']} | Amount: ₹{txn['amount']:,.2f} (Avg: ₹{txn['avg_amount']:,.2f})",
            f" Demographics: Region={txn['region']}, Age Group={txn['age_group']}",
            "-" * 65,
            " Active Reason Codes & Logic Axioms:"
        ]
        if fired_rules:
            for idx, rule in enumerate(fired_rules, 1):
                explanation.append(f"  {idx}. [{rule['code']}] {rule['id']} -> {rule['description']}")
        else:
            explanation.append("  - None (Zero deterministic high-risk rules triggered).")
        explanation.append(f" Bayesian Likelihood: {bayes_score * 100:.1f}%")
        explanation.append(f" Governance Action: {'Dispatch to Level-2 Analyst' if bayes_score >= 0.50 else 'Auto-Pass'}")
        explanation.append("=" * 65)
        return "\n".join(explanation)

    @staticmethod
    def audit_fairness_by_group(df: pd.DataFrame, pred_col: str, actual_col: str) -> pd.DataFrame:
        metrics = []
        for group in df["age_group"].unique():
            sub = df[df["age_group"] == group]
            fp = ((sub[pred_col] == True) & (sub[actual_col] == False)).sum()
            tn = ((sub[pred_col] == False) & (sub[actual_col] == False)).sum()
            tp = ((sub[pred_col] == True) & (sub[actual_col] == True)).sum()
            fn = ((sub[pred_col] == False) & (sub[actual_col] == True)).sum()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            metrics.append({"Demographic Slice": f"Age: {group}", "Count": len(sub), "FPR": f"{fpr * 100:.2f}%", "Recall": f"{recall * 100:.2f}%"})
        for reg in df["region"].unique():
            sub = df[df["region"] == reg]
            fp = ((sub[pred_col] == True) & (sub[actual_col] == False)).sum()
            tn = ((sub[pred_col] == False) & (sub[actual_col] == False)).sum()
            tp = ((sub[pred_col] == True) & (sub[actual_col] == True)).sum()
            fn = ((sub[pred_col] == False) & (sub[actual_col] == True)).sum()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            metrics.append({"Demographic Slice": f"Region: {reg}", "Count": len(sub), "FPR": f"{fpr * 100:.2f}%", "Recall": f"{recall * 100:.2f}%"})
        return pd.DataFrame(metrics)
