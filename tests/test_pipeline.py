"""
Unit Tests
"""
import unittest
from src.ingestion_drift import SchemaDriftDetector, generate_synthetic_transactions
from src.knowledge_reasoning import FOLRuleEngine, NaiveBayesRiskEngine
from src.csp_allocator import CSPScheduler, FraudAlert, Analyst

class TestFraudPipeline(unittest.TestCase):
    def setUp(self):
        self.df = generate_synthetic_transactions(n_records=50)
        self.rule_engine = FOLRuleEngine()
        self.bayes_engine = NaiveBayesRiskEngine()

    def test_schema_drift(self):
        detector = SchemaDriftDetector(expected_schema={"txn_id": "str", "missing_col": "bool"})
        validated = detector.inspect_and_validate(self.df, "TestFeed")
        self.assertIn("missing_col", validated.columns)

    def test_fol_rules(self):
        sample = {"txn_id": "T1", "amount": 25000.0, "avg_amount": 2000.0, "is_new_device": True, "is_foreign": False, "velocity_1h": 1, "failed_logins": 0, "ip_mismatch": False}
        is_susp, fired, _ = self.rule_engine.forward_chain(sample)
        self.assertTrue(is_susp)

    def test_csp_allocation(self):
        alerts = [FraudAlert("A1", "CARD", "HIGH", 30)]
        analysts = [Analyst("AN_1", "Exp", ["CARD"], "MORNING", 5)]
        scheduler = CSPScheduler(alerts, analysts)
        plan, _, _, _ = scheduler.solve_backtracking_with_propagation()
        self.assertIsNotNone(plan)

if __name__ == "__main__":
    unittest.main()
