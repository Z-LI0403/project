from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


T0_DIR = Path(__file__).resolve().parents[1]
if str(T0_DIR) not in sys.path:
    sys.path.insert(0, str(T0_DIR))

from common_linear_gaussian import (  # noqa: E402
    deterministic_orthogonal,
    load_config,
)
from validate_theorem1 import (  # noqa: E402
    _evaluate_a_inv_02,
    build_generic_case,
)


class Theorem1InvarianceRegressionTests(unittest.TestCase):
    def test_a_inv_02_records_raw_gate_and_passes_canonical_binding_routes(self) -> None:
        config = load_config(T0_DIR / "t0_config.json")
        root_seed = int(config["seeds"]["a_synthetic"])
        spec = next(
            item
            for item in config["a"]["generic_cases"]
            if item["id"] == "A-GEN-02"
        )
        base = build_generic_case(spec, root_seed)
        coordinate = (
            deterministic_orthogonal(
                root_seed, "A-INV-02", "coordinate-left", 4
            )[0]
            @ np.diag((1e-2, 1e-1, 1e1, 1e2))
            @ deterministic_orthogonal(
                root_seed, "A-INV-02", "coordinate-right", 4
            )[0].T
        )
        result = _evaluate_a_inv_02(base, coordinate, config)
        raw = result["raw_coordinate_diagnostic"]
        canonical = result["canonical_qr"]

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(
            raw["label"], "EXPECTED_RAW_COORDINATE_ILL_CONDITIONING"
        )
        self.assertFalse(raw["theorem_failure"])
        self.assertEqual(raw["expected_mathematical_rank"], 4)
        self.assertTrue(raw["mathematical_full_column_rank_by_invertible_construction"])
        self.assertEqual(raw["raw_measurement_rank"]["rank"], 4)
        self.assertTrue(raw["raw_measurement_rank"]["full_column_rank"])
        self.assertEqual(raw["raw_normal_matrix_rank"]["rank"], 3)
        self.assertFalse(raw["raw_normal_matrix_rank_gate_passed"])
        self.assertFalse(raw["raw_normal_matrix_condition_gate_passed"])
        self.assertFalse(result["raw_normal_system_cholesky_attempted"])

        self.assertEqual(canonical["Q_rank"]["rank"], 4)
        self.assertEqual(canonical["T_rank"]["rank"], 4)
        self.assertTrue(canonical["Q_rank"]["full_column_rank"])
        self.assertTrue(canonical["T_rank"]["full_column_rank"])
        self.assertLessEqual(
            canonical["normalized_reconstruction_residual"],
            float(config["tolerances"]["normalized_residual"]),
        )
        self.assertLessEqual(
            canonical["raw_column_space_residual"],
            float(config["tolerances"]["normalized_residual"]),
        )
        self.assertTrue(all(result["transformed_pairwise_checks"].values()))
        self.assertTrue(all(result["base_to_transformed_gap_checks"].values()))
        self.assertTrue(all(result["direct_risk_comparisons"].values()))
        self.assertTrue(all(result["transformed_risk_invariants"].values()))
        self.assertTrue(all(result["binding_checks"].values()))


if __name__ == "__main__":
    unittest.main()
