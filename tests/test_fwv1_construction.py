from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


T0_DIR = Path(__file__).resolve().parents[1]
if str(T0_DIR) not in sys.path:
    sys.path.insert(0, str(T0_DIR))

from fwv1_reference import (  # noqa: E402
    OutOfDomainRankError,
    a_rho,
    ambient_measurement_risk,
    build_fwv1,
    four_mode_risk,
    golden_section_minimize,
    l_tau_matrix,
    measurement,
    rational_risk,
    rho_star,
)
from common_linear_gaussian import (  # noqa: E402
    APPROVED_CONFIG_PATH,
    cholesky_solve,
    load_config,
    rank_diagnostics,
)
from validate_fwv1 import (  # noqa: E402
    _finalize_dirac_policy,
    _optimizer_candidate_checks,
    _static_policy,
)


class FWV1ConstructionTests(unittest.TestCase):
    @staticmethod
    def _optimizer_diagnostic(tau: float):
        config = load_config(APPROVED_CONFIG_PATH)
        dense_rho = np.linspace(-4.0, 4.0, 401, dtype=np.float64)
        dense_rho = dense_rho[dense_rho != np.float64(-0.5)]
        result = golden_section_minimize(
            lambda rho: np.float64(rational_risk(rho, tau)),
            interval_tolerance=float(config["tolerances"]["optimizer_interval"]),
            maximum_iterations=int(
                config["tolerances"]["optimizer_max_iterations"]
            ),
        )
        dense_minimum = float(
            np.min(
                np.asarray(
                    [rational_risk(rho, tau) for rho in dense_rho],
                    dtype=np.float64,
                )
            )
        )
        diagnostic = _optimizer_candidate_checks(
            rho_hat=result.x,
            tau=tau,
            solver_objective=result.objective,
            dense_minimum=dense_minimum,
            tolerances=config["tolerances"],
        )
        return config, dense_rho, result, diagnostic

    def test_construction_and_three_risk_routes_at_nonformal_point(self) -> None:
        model = build_fwv1()
        self.assertEqual(model.target.shape, (64, 16))
        np.testing.assert_allclose(model.target.T @ model.target, np.eye(16), atol=1e-13, rtol=1e-13)
        self.assertGreater(np.linalg.eigvalsh(model.sigma0)[0], 0.0)
        rho, tau = 0.137, 0.371
        ambient = ambient_measurement_risk(model, rho, tau)["risk"]
        reduced = four_mode_risk(rho, tau)
        closed = float(rational_risk(rho, tau))
        self.assertAlmostEqual(ambient, reduced, places=11)
        self.assertAlmostEqual(ambient, closed, places=11)
        self.assertEqual(rank_diagnostics(measurement(model, rho)).rank, 16)

    def test_frozen_golden_section_at_nonformal_tau(self) -> None:
        tau = 0.371
        result = golden_section_minimize(lambda rho: np.float64(rational_risk(rho, tau)))
        self.assertLessEqual(abs(result.x - float(rho_star(tau))), 2e-8)
        self.assertLessEqual(result.interval[1] - result.interval[0], 1e-12)

    def test_factorization_and_discarded_witness_at_nonformal_point(self) -> None:
        model = build_fwv1()
        tau = 0.371
        rho = float(rho_star(tau))
        conditional = cholesky_solve(np.eye(64) + tau * model.sigma0, model.target)
        factorized = measurement(model, rho) @ np.kron(l_tau_matrix(tau), np.eye(4))
        np.testing.assert_allclose(conditional, factorized, atol=1e-10, rtol=1e-9)

        witness_rho = 0.137
        identity = np.eye(8)
        p0 = (identity[:, 0] + identity[:, 1]) / np.sqrt(2.0)
        q0 = (identity[:, 0] - identity[:, 1]) / np.sqrt(2.0)
        discarded = np.kron(q0, p0)
        transformed = model.target.T @ a_rho(model, witness_rho) @ discarded
        self.assertAlmostEqual(float(transformed @ transformed), witness_rho**2 / 2.0, places=12)

    def test_rank_domain_api_returns_structured_rejection_without_risk(self) -> None:
        model = build_fwv1()
        with self.assertRaises(OutOfDomainRankError) as caught:
            ambient_measurement_risk(model, -0.5, 0.371)
        self.assertEqual(caught.exception.structured_result, "OUT_OF_DOMAIN_RANK_12")
        self.assertEqual(caught.exception.diagnostics.rank, 12)

    def test_tau_one_parameter_reference_miss_is_nonbinding(self) -> None:
        _, _, result, diagnostic = self._optimizer_diagnostic(1.0)
        self.assertEqual(result.objective, 11.0)
        self.assertFalse(diagnostic["parameter_recovery_reference"]["passed"])
        self.assertGreater(
            diagnostic["parameter_recovery_reference"]["absolute_error"], 2e-8
        )
        self.assertTrue(diagnostic["binding_passed"])
        self.assertTrue(all(diagnostic["binding_checks"].values()))

    def test_high_snr_parameter_plateau_is_reported_nonbinding(self) -> None:
        _, _, _, diagnostic = self._optimizer_diagnostic(1e6)
        self.assertFalse(diagnostic["parameter_recovery_reference"]["passed"])
        self.assertGreater(
            diagnostic["parameter_recovery_reference"]["absolute_error"], 1e-5
        )
        self.assertTrue(diagnostic["binding_passed"])
        self.assertLessEqual(
            diagnostic["exact_regret_at_candidate"],
            diagnostic["risk_resolution_tolerance"],
        )

    def test_bad_optimizer_candidate_fails_objective_and_regret_gates(self) -> None:
        config = load_config(APPROVED_CONFIG_PATH)
        tau = 1.0
        bad_rho = 0.0
        bad_risk = float(rational_risk(bad_rho, tau))
        diagnostic = _optimizer_candidate_checks(
            rho_hat=bad_rho,
            tau=tau,
            solver_objective=bad_risk,
            dense_minimum=float(rational_risk(0.25, tau)),
            tolerances=config["tolerances"],
        )
        self.assertFalse(diagnostic["binding_passed"])
        self.assertFalse(diagnostic["binding_checks"]["objective_vs_exact_value"])
        self.assertFalse(
            diagnostic["binding_checks"]["exact_regret_within_risk_resolution"]
        )

    def test_dirac_policy_equality_uses_risk_resolution_not_parameter_gate(self) -> None:
        config, dense_rho, _, _ = self._optimizer_diagnostic(1.0)
        row = _static_policy(
            "dirac-unit-regression",
            [1.0],
            [1.0],
            {"dense_rho": dense_rho},
            config,
        )
        row = _finalize_dirac_policy(row, config["tolerances"])
        self.assertEqual(row["status"], "PASS")
        self.assertFalse(row["parameter_recovery_reference"]["passed"])
        self.assertTrue(all(row["binding_checks"].values()))
        self.assertEqual(row["risk_difference_advantage"], 0.0)
        self.assertLessEqual(
            abs(row["exact_regret_advantage"]), row["risk_resolution_tolerance"]
        )


if __name__ == "__main__":
    unittest.main()
