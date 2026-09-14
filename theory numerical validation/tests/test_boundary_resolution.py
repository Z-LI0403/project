from __future__ import annotations

import sys
import unittest
from pathlib import Path


T0_DIR = Path(__file__).resolve().parents[1]
if str(T0_DIR) not in sys.path:
    sys.path.insert(0, str(T0_DIR))

from common_linear_gaussian import APPROVED_CONFIG_PATH, load_config  # noqa: E402
from validate_boundaries import (  # noqa: E402
    _direct_risk_accuracy_classification,
    _noise_binding_checks,
    _noise_case,
    _singular_case,
)


class BoundaryResolutionRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(APPROVED_CONFIG_PATH)

    def test_noise_distinct_retains_original_point_and_appends_production_rows(self) -> None:
        result = _noise_case(False, self.config)
        rows = {row["k"]: row for row in result["rows"]}
        self.assertEqual(list(rows), list(range(1, 9)))
        self.assertEqual(rows[6]["a"], 1e-6)
        self.assertGreater(rows[6]["absolute_error"], 1e-12)
        self.assertLess(rows[6]["absolute_error"], 5e-12)
        for exponent in (7, 8):
            self.assertEqual(rows[exponent]["a"], 10.0 ** (-exponent))
            self.assertEqual(rows[exponent]["b"], 1.0)
            self.assertEqual(rows[exponent]["gap_route"], "SUBSPACE_PARENT_GAP")
            self.assertTrue(rows[exponent]["finite"])
            self.assertTrue(rows[exponent]["nonnegative"])
        self.assertTrue(result["binding_final_check"])
        self.assertEqual(result["status"], "PASS")

    def test_noise_distinct_wrong_exact_limit_fails(self) -> None:
        result = _noise_case(False, self.config)
        wrong = _noise_binding_checks(
            rows=result["rows"],
            computed_limit=result["computed_subspace_limit"],
            expected_limit=0.0,
            matched=False,
            config=self.config,
        )
        self.assertFalse(wrong["computed_limit_matches_exact_limit"])
        self.assertFalse(all(wrong.values()))

    def test_singular_accuracy_gate_classifies_k3_k4_k5_and_preserves_k6(self) -> None:
        result = _singular_case(self.config)
        rows = {row["k"]: row for row in result["rows"]}

        self.assertEqual(
            rows[3]["direct_route"], "DIRECT_RISK_ACCURACY_ELIGIBLE_BINDING"
        )
        self.assertTrue(rows[3]["direct_evaluated"])
        self.assertTrue(rows[3]["direct_comparison_binding"])
        self.assertTrue(rows[3]["direct_check"])
        self.assertLessEqual(
            rows[3]["accuracy_budget"], rows[3]["risk_comparison_tolerance"]
        )

        for exponent in (4, 5):
            self.assertEqual(
                rows[exponent]["direct_route"],
                "FINITE_PRECISION_DIRECT_RISK_DIAGNOSTIC",
            )
            self.assertTrue(rows[exponent]["direct_evaluated"])
            self.assertFalse(rows[exponent]["direct_comparison_binding"])
            self.assertGreater(
                rows[exponent]["accuracy_budget"],
                rows[exponent]["risk_comparison_tolerance"],
            )

        self.assertEqual(rows[6]["direct_route"], "ILL_CONDITIONED_FULL_RANK")
        self.assertFalse(rows[6]["solver_admissible"])
        self.assertFalse(rows[6]["direct_evaluated"])
        self.assertTrue(all(result["binding_checks"].values()))
        self.assertEqual(result["status"], "PASS")

    def test_well_conditioned_wrong_direct_gap_remains_binding_failure(self) -> None:
        result = _direct_risk_accuracy_classification(
            condition_number=200.0,
            direct_gap=0.75,
            exact_gap=0.5,
            tolerances=self.config["tolerances"],
        )
        self.assertTrue(result["accuracy_eligible"])
        self.assertFalse(result["direct_agreement"])
        self.assertFalse(result["binding_passed"])


if __name__ == "__main__":
    unittest.main()
