"""
Layer 1: Resource-Allocation CSP for Fraud Alert Assignment
"""
from typing import Dict, List, Optional, Tuple
import time

class FraudAlert:
    def __init__(self, alert_id: str, channel: str, severity: str, deadline_minutes: int):
        self.alert_id = alert_id
        self.channel = channel
        self.severity = severity
        self.deadline_minutes = deadline_minutes

class Analyst:
    def __init__(self, analyst_id: str, name: str, expertise: List[str], shift: str, capacity: int):
        self.analyst_id = analyst_id
        self.name = name
        self.expertise = expertise
        self.shift = shift
        self.capacity = capacity

class CSPScheduler:
    def __init__(self, alerts: List[FraudAlert], analysts: List[Analyst]):
        self.alerts = alerts
        self.analysts = analysts
        self.assignments_tried = 0
        self.backtracks = 0

    def is_consistent(self, alert: FraudAlert, analyst: Analyst, current_assignment: Dict[str, str]) -> bool:
        if alert.channel not in analyst.expertise:
            return False
        current_load = sum(1 for aid, an_id in current_assignment.items() if an_id == analyst.analyst_id)
        if current_load >= analyst.capacity:
            return False
        if alert.severity == "HIGH" and analyst.shift == "NIGHT" and alert.deadline_minutes <= 30:
            return False
        return True

    def solve_plain_backtracking(self) -> Tuple[Optional[Dict[str, str]], int, int, float]:
        self.assignments_tried, self.backtracks = 0, 0
        t_start = time.perf_counter()
        assignment = {}
        def backtrack(alert_idx: int) -> bool:
            if alert_idx == len(self.alerts): return True
            alert = self.alerts[alert_idx]
            for analyst in self.analysts:
                self.assignments_tried += 1
                if self.is_consistent(alert, analyst, assignment):
                    assignment[alert.alert_id] = analyst.analyst_id
                    if backtrack(alert_idx + 1): return True
                    del assignment[alert.alert_id]
                    self.backtracks += 1
            return False
        success = backtrack(0)
        return (assignment if success else None), self.assignments_tried, self.backtracks, time.perf_counter() - t_start

    def solve_backtracking_with_propagation(self) -> Tuple[Optional[Dict[str, str]], int, int, float]:
        self.assignments_tried, self.backtracks = 0, 0
        t_start = time.perf_counter()
        domains = {alert.alert_id: [a for a in self.analysts if alert.channel in a.expertise] for alert in self.alerts}
        assignment = {}

        def forward_check(unassigned: List[str]) -> bool:
            for aid in unassigned:
                al = next(a for a in self.alerts if a.alert_id == aid)
                if not any(self.is_consistent(al, an, assignment) for an in domains[aid]): return False
            return True

        def backtrack(unassigned: List[str]) -> bool:
            if not unassigned: return True
            var_id = min(unassigned, key=lambda aid: len(domains[aid]))
            var_alert = next(a for a in self.alerts if a.alert_id == var_id)
            remaining = [aid for aid in unassigned if aid != var_id]

            for analyst in domains[var_id]:
                self.assignments_tried += 1
                if self.is_consistent(var_alert, analyst, assignment):
                    assignment[var_id] = analyst.analyst_id
                    if forward_check(remaining):
                        if backtrack(remaining): return True
                    del assignment[var_id]
                    self.backtracks += 1
            return False

        success = backtrack([a.alert_id for a in self.alerts])
        return (assignment if success else None), self.assignments_tried, self.backtracks, time.perf_counter() - t_start
