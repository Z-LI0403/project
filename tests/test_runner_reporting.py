from __future__ import annotations

import sys
import unittest
from pathlib import Path


T0_DIR = Path(__file__).resolve().parents[1]
if str(T0_DIR) not in sys.path:
    sys.path.insert(0, str(T0_DIR))

from common_linear_gaussian import NumericalFailure, load_config, overall_status  # noqa: E402
from run_t0 import (  # noqa: E402
    _apply_case_count_gate,
    _module_execution_failure,
    _write_report,
)


class RunnerReportFixtureTests(unittest.TestCase):
    def test_coverage_gate_preserves_execution_error_priority(self) -> None:
        environmental = _module_execution_failure(
            "A", RuntimeError("unit environment"), numerical_failure_is_fail=True
        )
        _apply_case_count_gate("A", environmental, 13)
        self.assertEqual(environmental["verdict"], "INCONCLUSIVE")
        self.assertFalse(environmental["coverage_check"]["passed"])

        numerical = _module_execution_failure(
            "A", NumericalFailure("unit numerical"), numerical_failure_is_fail=True
        )
        _apply_case_count_gate("A", numerical, 13)
        self.assertEqual(numerical["verdict"], "FAIL")

        silently_incomplete = {"verdict": "PASS", "cases": {}}
        _apply_case_count_gate("A", silently_incomplete, 13)
        self.assertEqual(silently_incomplete["verdict"], "FAIL")

    def test_d_execution_failure_classification_is_fail_closed(self) -> None:
        numerical = _module_execution_failure(
            "D", NumericalFailure("unit binding computation"), numerical_failure_is_fail=False
        )
        environmental = _module_execution_failure(
            "D", RuntimeError("unit environment"), numerical_failure_is_fail=False
        )
        self.assertEqual(numerical["D_BRIDGE_INTEGRITY_VERDICT"], "E2_REVIEW_REQUIRED")
        self.assertEqual(environmental["D_BRIDGE_INTEGRITY_VERDICT"], "INCONCLUSIVE")
        self.assertEqual(numerical["D_APPROXIMATION_ADVISORY"], "NOT_EVALUABLE")

        complete = {
            "case_id": "D-BRIDGE",
            "D_BRIDGE_INTEGRITY_VERDICT": "PASS",
            "D_APPROXIMATION_ADVISORY": "WITHIN_ADVISORY_REFERENCE_THRESHOLDS",
        }
        _apply_case_count_gate("D", complete, 1)
        self.assertTrue(complete["coverage_check"]["passed"])

        wrong_id = {
            "case_id": "D-UNAPPROVED",
            "D_BRIDGE_INTEGRITY_VERDICT": "PASS",
            "D_APPROXIMATION_ADVISORY": "WITHIN_ADVISORY_REFERENCE_THRESHOLDS",
        }
        _apply_case_count_gate("D", wrong_id, 1)
        self.assertEqual(wrong_id["D_BRIDGE_INTEGRITY_VERDICT"], "E2_REVIEW_REQUIRED")
        self.assertEqual(wrong_id["D_APPROXIMATION_ADVISORY"], "NOT_EVALUABLE")

    def test_unknown_case_status_cannot_pass_overall_gate(self) -> None:
        self.assertEqual(overall_status(["PASS", "UNREGISTERED_STATUS"]), "FAIL")

    def test_partial_dependency_stop_report_is_mechanically_writable(self) -> None:
        target = T0_DIR / "tests" / "_report_writer_smoke.md"
        if target.exists():
            target.unlink()
        config = load_config(T0_DIR / "t0_config.json")
        modules = {
            "A": {
                "verdict": "PASS",
                "cases": {
                    "A-UNIT-NONFORMAL": {
                        "status": "PASS",
                        "dimensions": {"d": 3, "m": 1, "q": 1},
                    }
                },
                "mc_summaries": {},
            },
            "B": None,
            "C": None,
            "D": None,
        }
        metadata = {
            "plan_sha256": "unit-plan",
            "config_path": "experiments/theory_validation/t0/t0_config.json",
            "config_sha256": "unit-config",
            "formal_command_argv": ["NONFORMAL-REPORT-WRITER-UNIT"],
            "rng_stream_registry": {"root_seeds": config["seeds"]},
            "numpy_float64_epsilon": 2.220446049250313e-16,
            "environment": {
                "python": "unit",
                "python_implementation": "unit",
                "numpy": "unit",
                "platform": "unit",
                "processor": "unit",
                "dtype": "float64",
                "numpy_build_configuration": "unit",
            },
        }
        health = {
            "solver_routes": {},
            "module_verdicts": {"A": "PASS", "B": "NOT_RUN", "C": "NOT_RUN", "D": "NOT_RUN"},
            "actual_case_counts": {"A": 1, "B": 0, "C": 0, "D": 0},
            "coverage_counts_match": {
                "A": False,
                "B": "NOT_RUN_DUE_TO_STOP",
                "C": "NOT_RUN_DUE_TO_STOP",
                "D": "NOT_RUN_DUE_TO_STOP",
            },
            "execution_errors": {},
            "stopped_after": "A",
            "nan_inf_policy": "unit",
        }
        try:
            _write_report(
                target,
                config=config,
                source_lock={"files": {"unit": "digest"}},
                metadata=metadata,
                health=health,
                modules=modules,
                mathematical_verdict="INCONCLUSIVE",
                bridge_verdict="INCONCLUSIVE",
                advisory="NOT_EVALUABLE",
                combined="T0_INCONCLUSIVE",
                stopped_after="A",
            )
            text = target.read_text(encoding="utf-8")
            self.assertIn("A-UNIT-NONFORMAL", text)
            self.assertIn("NOT_RUN_DUE_TO_STOP", text)
            self.assertIn("T0_INCONCLUSIVE", text)
        finally:
            if target.exists():
                target.unlink()


if __name__ == "__main__":
    unittest.main()
