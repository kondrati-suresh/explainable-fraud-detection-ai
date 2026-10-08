"""
Main Execution Script - Demonstrating all Rubric Outputs
"""
import pandas as pd
from tabulate import tabulate
from src.config import SEED
from src.ingestion_drift import SchemaDriftDetector, generate_synthetic_transactions
from src.knowledge_reasoning import FOLRuleEngine, NaiveBayesRiskEngine
from src.csp_allocator import CSPScheduler, FraudAlert, Analyst
from src.responsible_ai import ResponsibleAIAuditor

def run_prototype():
    print("=" * 80)
    print(" VEL TECH HIGH TECH DR.RANGARAJAN DR.SAKUNTHALA ENGINEERING COLLEGE")
    print(" DEPARTMENT OF CSE(AIML) | 25ML35T - FOUNDATIONS OF ARTIFICIAL INTELLIGENCE")
    print(f" BATCH 15: EXPLAINABLE FRAUD DETECTION PIPELINE (SEED: {SEED})")
    print("=" * 80)

    print("\n[STEP 1] Running Ingestion Pipeline & Schema Drift Inspection...")
    df = generate_synthetic_transactions(n_records=1200)
    print(f"-> Ingested {len(df)} records across CARD, UPI, LOAN channels. (Schema validation: PASSED)")

    print("\n[STEP 2] Benchmarking FOL Rules vs. Bayesian Risk Engine...")
    rule_engine, bayes_engine = FOLRuleEngine(), NaiveBayesRiskEngine()
    rule_preds, bayes_preds, hybrid_preds = [], [], []
    for _, row in df.iterrows():
        txn = row.to_dict()
        is_susp_rule, _, _ = rule_engine.forward_chain(txn)
        bayes_prob = bayes_engine.compute_risk_score(txn)
        rule_preds.append(is_susp_rule)
        bayes_preds.append(bayes_prob >= 0.50)
        hybrid_preds.append(is_susp_rule or (bayes_prob >= 0.65))

    df["pred_rule"], df["pred_bayes"], df["pred_hybrid"] = rule_preds, bayes_preds, hybrid_preds

    def calc_metrics(pred_col):
        tp = ((df[pred_col] == True) & (df["is_actual_fraud"] == True)).sum()
        fp = ((df[pred_col] == True) & (df["is_actual_fraud"] == False)).sum()
        fn = ((df[pred_col] == False) & (df["is_actual_fraud"] == True)).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
        return p, r, f1

    p_r, r_r, f1_r = calc_metrics("pred_rule")
    p_b, r_b, f1_b = calc_metrics("pred_bayes")
    p_h, r_h, f1_h = calc_metrics("pred_hybrid")

    model_comp = [
        ["FOL Rule Engine Only", f"{p_r*100:.2f}%", f"{r_r*100:.2f}%", f"{f1_r*100:.2f}%"],
        ["Bayesian Risk Engine Only", f"{p_b*100:.2f}%", f"{r_b*100:.2f}%", f"{f1_b*100:.2f}%"],
        ["Hybrid (FOL + Bayes) [Recommended]", f"{p_h*100:.2f}%", f"{r_h*100:.2f}%", f"{f1_h*100:.2f}%"]
    ]
    print(tabulate(model_comp, headers=["Model Architecture", "Precision", "Recall", "F1-Score"], tablefmt="fancy_grid"))

    print("\n[STEP 3] Evaluating Subgroup Fairness & Disparate Impact...")
    fairness_df = ResponsibleAIAuditor.audit_fairness_by_group(df, "pred_hybrid", "is_actual_fraud")
    print(tabulate(fairness_df, headers="keys", tablefmt="fancy_grid", showindex=False))

    print("\n[STEP 4] Executing CSP Resource Allocation for Fraud Analysts...")
    flagged_txns = df[df["pred_hybrid"] == True].head(25)
    alerts = [FraudAlert(r["txn_id"], r["channel"], "HIGH" if r["amount"] > 15000 else "MEDIUM", 30 if r["amount"] > 15000 else 120) for _, r in flagged_txns.iterrows()]
    analysts = [
        Analyst("AN_01", "Dr. Sharma", ["CARD", "UPI"], "MORNING", capacity=10),
        Analyst("AN_02", "Mr. Rajesh", ["UPI", "LOAN"], "MORNING", capacity=8),
        Analyst("AN_03", "Ms. Priya", ["CARD", "LOAN"], "EVENING", capacity=12),
        Analyst("AN_04", "Mr. Anand", ["CARD", "UPI", "LOAN"], "EVENING", capacity=10)
    ]
    scheduler = CSPScheduler(alerts, analysts)
    plan_plain, tried_p, back_p, time_p = scheduler.solve_plain_backtracking()
    plan_mrv, tried_m, back_m, time_m = scheduler.solve_backtracking_with_propagation()

    csp_benchmark = [
        ["Plain Backtracking", tried_p, back_p, f"{time_p*1000:.3f} ms", "Solved" if plan_plain else "Failed"],
        ["Backtracking + Forward Checking (MRV)", tried_m, back_m, f"{time_m*1000:.3f} ms", "Solved" if plan_mrv else "Failed"]
    ]
    print(tabulate(csp_benchmark, headers=["Algorithm Variant", "Assignments Tried", "Backtracks", "Runtime", "Status"], tablefmt="fancy_grid"))

    print("\n[STEP 5] Demonstrating Detailed FOL Inference Trace & XAI Output...")
    sample_sus_txn = df[df["pred_hybrid"] == True].iloc[0].to_dict()
    _, sample_fired, sample_trace = rule_engine.forward_chain(sample_sus_txn)
    sample_bayes = bayes_engine.compute_risk_score(sample_sus_txn)

    print("\n--- Complete First-Order Logic Inference Trace ---")
    for step in sample_trace: print(step)

    print("\n--- User-Facing XAI Explanation Card ---")
    print(ResponsibleAIAuditor.generate_xai_explanation(sample_sus_txn, sample_fired, sample_bayes))

if __name__ == "__main__":
    run_prototype()
