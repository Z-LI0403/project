from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


T0_DIR = Path(__file__).resolve().parents[1]
if str(T0_DIR) not in sys.path:
    sys.path.insert(0, str(T0_DIR))

from common_linear_gaussian import (  # noqa: E402
    NumericalFailure,
    SourceDriftError,
    cholesky_solve,
    direct_gaussian_conditioning,
    jsonable,
    load_config,
    psd_square_root,
    rank_diagnostics,
    stable_seed,
    subspace_parent_gap,
    theorem1_formula,
    whitened_residual_gap,
)


class CommonLinearGaussianTests(unittest.TestCase):
    def test_stable_seed_is_deterministic_and_component_specific(self) -> None:
        first = stable_seed(1234, "UNIT-CASE", "stream-a")
        second = stable_seed(1234, "UNIT-CASE", "stream-a")
        different = stable_seed(1234, "UNIT-CASE", "stream-b")
        self.assertEqual(first, second)
        self.assertNotEqual(first, different)
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 2**128)

    def test_three_independent_routes_on_nonformal_three_dimensional_fixture(self) -> None:
        sigma0 = np.asarray(
            ((1.1, 0.2, 0.05), (0.2, 0.9, -0.1), (0.05, -0.1, 0.7)),
            dtype=np.float64,
        )
        target = np.asarray(((1.0,), (0.0,), (0.0,)), dtype=np.float64)
        measurement = np.asarray(((1.0,), (1.0,), (0.5,)), dtype=np.float64)
        measurement /= np.linalg.norm(measurement)
        formula = theorem1_formula(sigma0, 0.61, 0.73, target, measurement)["gap"]
        direct = direct_gaussian_conditioning(sigma0, 0.61, 0.73, target, measurement)["gap"]
        whitened = whitened_residual_gap(sigma0, 0.61, 0.73, target, measurement)["gap"]
        self.assertAlmostEqual(formula, direct, places=12)
        self.assertAlmostEqual(formula, whitened, places=12)
        self.assertGreaterEqual(formula, 0.0)

    def test_distinct_parent_formula_on_nonformal_fixture(self) -> None:
        identity = np.eye(3, dtype=np.float64)
        retained = identity[:, :1]
        complement = identity[:, 1:]
        target = (identity[:, :1] + identity[:, 1:2]) / math.sqrt(2.0)
        sigma0 = np.asarray(((1.0, 0.3, 0.0), (0.3, 0.8, 0.1), (0.0, 0.1, 0.6)))
        parent = subspace_parent_gap(sigma0, 0.43, 0.82, retained, complement, target)
        formula = theorem1_formula(sigma0, 0.43, 0.82, target, retained)
        self.assertAlmostEqual(parent["gap"], formula["gap"], places=12)

    def test_frozen_condition_gate_fails_before_cholesky_route(self) -> None:
        system = np.diag(np.asarray((1.0, 5e-13), dtype=np.float64))
        with self.assertRaisesRegex(
            NumericalFailure, "case_id=UNIT-RANK-GATE, system=unit/ill-conditioned"
        ):
            cholesky_solve(
                system,
                np.ones((2, 1), dtype=np.float64),
                case_id="UNIT-RANK-GATE",
                system_name="unit/ill-conditioned",
            )

    def test_singular_clean_covariance_uses_eigh_and_has_spd_interior(self) -> None:
        retained = np.zeros((8, 4), dtype=np.float64)
        complement = np.zeros((8, 4), dtype=np.float64)
        scale = 1.0 / math.sqrt(2.0)
        for index in range(4):
            retained[2 * index : 2 * index + 2, index] = (scale, scale)
            complement[2 * index : 2 * index + 2, index] = (scale, -scale)
        basis = np.column_stack((retained, complement))
        retained_block = np.diag((0.0, 0.5, 1.0, 2.0))
        discarded_block = np.diag((0.25, 0.75, 1.5, 3.0))
        sigma0 = basis @ np.block(
            [
                [retained_block, np.zeros((4, 4), dtype=np.float64)],
                [np.zeros((4, 4), dtype=np.float64), discarded_block],
            ]
        ) @ basis.T

        clean_eigenvalues = np.linalg.eigvalsh((sigma0 + sigma0.T) / 2.0)
        self.assertEqual(rank_diagnostics(sigma0).rank, 7)
        self.assertAlmostEqual(float(clean_eigenvalues[0]), 0.0, places=15)
        np.testing.assert_allclose(
            retained.T @ sigma0 @ retained,
            retained_block,
            atol=1e-15,
            rtol=1e-15,
        )

        square_root_diagnostics: dict[str, object] = {}
        sigma0_root = psd_square_root(
            sigma0,
            diagnostics=square_root_diagnostics,
            case_id="UNIT-SINGULAR-CLEAN",
            system_name="clean Sigma_0",
        )
        np.testing.assert_allclose(
            sigma0_root @ sigma0_root.T, sigma0, atol=1e-14, rtol=1e-14
        )
        self.assertLessEqual(int(square_root_diagnostics["negative_clip_count"]), 1)
        self.assertGreaterEqual(
            float(square_root_diagnostics["lambda_min_before_clip"]), -3e-12
        )

        a = b = 1.0 / math.sqrt(2.0)
        sigma_t = a * a * sigma0 + b * b * np.eye(8, dtype=np.float64)
        measurement_gram = retained.T @ sigma_t @ retained
        self.assertTrue(rank_diagnostics(sigma_t).full_column_rank)
        self.assertTrue(rank_diagnostics(measurement_gram).full_column_rank)
        np.testing.assert_allclose(
            np.linalg.eigvalsh(sigma_t),
            np.asarray((0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5, 2.0)),
            atol=1e-14,
            rtol=1e-14,
        )
        np.testing.assert_allclose(
            np.linalg.eigvalsh(measurement_gram),
            np.asarray((0.5, 0.75, 1.0, 1.5)),
            atol=1e-14,
            rtol=1e-14,
        )
        cholesky_solve(
            sigma_t,
            retained,
            case_id="UNIT-SINGULAR-CLEAN",
            system_name="interior Sigma_t",
        )
        cholesky_solve(
            measurement_gram,
            retained.T @ retained,
            case_id="UNIT-SINGULAR-CLEAN",
            system_name="interior M^T Sigma_t M",
        )
        formula = theorem1_formula(
            sigma0, a, b, retained, retained, case_id="UNIT-SINGULAR-CLEAN"
        )
        direct = direct_gaussian_conditioning(
            sigma0, a, b, retained, retained, case_id="UNIT-SINGULAR-CLEAN"
        )
        whitened = whitened_residual_gap(
            sigma0, a, b, retained, retained, case_id="UNIT-SINGULAR-CLEAN"
        )
        self.assertAlmostEqual(formula["gap"], 0.0, places=14)
        self.assertAlmostEqual(formula["gap"], direct["gap"], places=14)
        self.assertAlmostEqual(formula["gap"], whitened["gap"], places=14)

    def test_noncanonical_config_path_is_rejected(self) -> None:
        with self.assertRaises(SourceDriftError):
            load_config(T0_DIR / "not-the-approved-config.json")

    def test_cholesky_route_handles_multiple_right_hand_sides(self) -> None:
        system = np.asarray(((3.0, 0.4, -0.2), (0.4, 2.0, 0.3), (-0.2, 0.3, 1.5)))
        right = np.asarray(((1.0, -1.0), (2.0, 0.5), (-0.25, 3.0)))
        solution = cholesky_solve(system, right)
        np.testing.assert_allclose(system @ solution, right, atol=1e-12, rtol=1e-12)

    def test_nonfinite_numpy_values_are_serialized_fail_closed(self) -> None:
        converted = jsonable(
            {
                "array": np.asarray((np.nan, np.inf, -np.inf), dtype=np.float64),
                "scalar": np.float64(np.nan),
            }
        )
        self.assertEqual(converted["array"], ["nan", "inf", "-inf"])
        self.assertEqual(converted["scalar"], "nan")


if __name__ == "__main__":
    unittest.main()
