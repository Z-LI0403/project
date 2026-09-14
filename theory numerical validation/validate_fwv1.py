"""T0-B: exact and Monte Carlo regression for the frozen FW-v1 theory."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from common_linear_gaussian import (
    NumericalFailure,
    SourceDriftError,
    base_parser,
    cholesky_solve,
    environment_metadata,
    load_config,
    new_run_directory,
    overall_status,
    parse_numeric_token,
    rank_diagnostics,
    scalar_close,
    verify_source_lock,
    write_json,
)
from fwv1_reference import (
    FWV1,
    OutOfDomainRankError,
    a_rho,
    ambient_full_risk,
    ambient_measurement_risk,
    build_fwv1,
    d_polynomial,
    exact_regret,
    four_mode_risk,
    fwv1_monte_carlo,
    golden_section_minimize,
    identity_gap,
    incremental_optimal_objective,
    incremental_weighted_objective,
    l_tau_eigenvalues,
    l_tau_matrix,
    measurement,
    optimal_value,
    q_polynomial,
    rational_risk,
    rho_star,
    witness_relative_gain,
)


def _token_values(tokens: Sequence[str]) -> np.ndarray:
    return np.asarray([parse_numeric_token(token) for token in tokens], dtype=np.float64)


def _grids(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    spec = config["b"]
    core_tau = _token_values(spec["core_tau_tokens"])
    start, stop, count = spec["dense_tau_logspace"]
    dense_tau = np.unique(
        np.concatenate(
            (core_tau, np.logspace(start, stop, int(count), dtype=np.float64))
        )
    )
    core_rho = _token_values(spec["core_rho_tokens"])
    r_start, r_stop, r_count = spec["dense_rho"]
    dense_rho = np.linspace(r_start, r_stop, int(r_count), dtype=np.float64)
    dense_rho = dense_rho[dense_rho != np.float64(-0.5)]
    return {
        "core_tau": core_tau,
        "dense_tau": dense_tau,
        "core_rho": core_rho,
        "dense_rho": dense_rho,
    }


def _unique_risk_points(grids: Mapping[str, np.ndarray]) -> list[tuple[float, float]]:
    points: dict[tuple[str, str], tuple[float, float]] = {}
    for tau in grids["dense_tau"]:
        for rho in np.concatenate((grids["core_rho"], np.atleast_1d(rho_star(tau)))):
            if rho == -0.5:
                continue
            key = (format(float(tau), ".17g"), format(float(rho), ".17g"))
            points[key] = (float(rho), float(tau))
    for tau in grids["core_tau"]:
        for rho in grids["dense_rho"]:
            key = (format(float(tau), ".17g"), format(float(rho), ".17g"))
            points[key] = (float(rho), float(tau))
    return [points[key] for key in sorted(points)]


def _risk_grid_case(
    model: FWV1, grids: Mapping[str, np.ndarray], tolerances: Mapping[str, float]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    max_ambient_four = 0.0
    max_ambient_rational = 0.0
    minimum_q = math.inf
    for rho, tau in _unique_risk_points(grids):
        ambient = ambient_measurement_risk(model, rho, tau)
        reduced = four_mode_risk(rho, tau)
        closed = float(rational_risk(rho, tau))
        q_value = float(q_polynomial(rho, tau))
        error_four = abs(ambient["risk"] - reduced)
        error_closed = abs(ambient["risk"] - closed)
        max_ambient_four = max(max_ambient_four, error_four)
        max_ambient_rational = max(max_ambient_rational, error_closed)
        minimum_q = min(minimum_q, q_value)
        passed = (
            q_value > 0.0
            and scalar_close(
                ambient["risk"],
                reduced,
                float(tolerances["scalar_atol"]),
                float(tolerances["scalar_rtol"]),
            )
            and scalar_close(
                ambient["risk"],
                closed,
                float(tolerances["scalar_atol"]),
                float(tolerances["scalar_rtol"]),
            )
        )
        if not passed:
            failures.append(f"rho={rho:.17g},tau={tau:.17g}")
        rows.append(
            {
                "rho": rho,
                "tau": tau,
                "ambient": ambient["risk"],
                "four_mode": reduced,
                "rational": closed,
                "q": q_value,
                "ambient_four_error": error_four,
                "ambient_rational_error": error_closed,
                "measurement_rank": ambient["measurement_rank"]["rank"],
                "gram_condition": ambient["gram_condition"],
                "passed": passed,
            }
        )
    case = {
        "status": "PASS" if not failures else "FAIL",
        "points": len(rows),
        "minimum_q": minimum_q,
        "max_ambient_four_error": max_ambient_four,
        "max_ambient_rational_error": max_ambient_rational,
        "failed_points": failures,
    }
    return case, rows


def _identity_case(
    model: FWV1, tau_values: np.ndarray, tolerances: Mapping[str, float]
) -> dict[str, Any]:
    failures: list[float] = []
    rows = []
    for tau in tau_values:
        reference = float(identity_gap(tau))
        matrix_gap = ambient_measurement_risk(model, 0.0, float(tau))["risk"] - ambient_full_risk(
            model, float(tau)
        )
        passed = reference > 0.0 and scalar_close(
            reference,
            matrix_gap,
            float(tolerances["scalar_atol"]),
            float(tolerances["scalar_rtol"]),
        )
        if not passed:
            failures.append(float(tau))
        rows.append(
            {"tau": float(tau), "reference_gap": reference, "matrix_gap": matrix_gap, "passed": passed}
        )
    center_risk = float(rational_risk(0.0, 1.0))
    center_gap = float(identity_gap(1.0))
    center = {
        "risk_67_over_6": scalar_close(
            center_risk,
            67.0 / 6.0,
            float(tolerances["center_atol"]),
            float(tolerances["center_rtol"]),
        ),
        "gap_1_over_6": scalar_close(
            center_gap,
            1.0 / 6.0,
            float(tolerances["center_atol"]),
            float(tolerances["center_rtol"]),
        ),
        "risk": center_risk,
        "gap": center_gap,
    }
    return {
        "status": "PASS" if not failures and all(value for key, value in center.items() if key.endswith("6")) else "FAIL",
        "rows": rows,
        "center": center,
        "failed_tau": failures,
    }


def _witness_case(tau_values: np.ndarray, tolerances: Mapping[str, float]) -> dict[str, Any]:
    rows = []
    failures: list[float] = []
    zero = float(tolerances["theoretical_zero"])
    for tau in tau_values:
        gap = float(exact_regret(0.25, tau))
        risk_difference = float(rational_risk(0.0, tau) - rational_risk(0.25, tau))
        relative_reference = float(witness_relative_gain(tau))
        is_registered_tau_one = float(tau) == 1.0
        zero_iff_tau_one = (
            abs(gap) <= zero if is_registered_tau_one else gap > 0.0
        )
        passed = zero_iff_tau_one and scalar_close(
            risk_difference,
            relative_reference,
            float(tolerances["scalar_atol"]),
            float(tolerances["scalar_rtol"]),
        ) and gap >= float(tolerances["nonnegative_floor"])
        if not passed:
            failures.append(float(tau))
        rows.append(
            {
                "tau": float(tau),
                "witness_gap": gap,
                "identity_minus_witness": risk_difference,
                "relative_reference": relative_reference,
                "is_registered_tau_one": is_registered_tau_one,
                "zero_iff_tau_one": zero_iff_tau_one,
                "passed": passed,
            }
        )
    special = {
        "gap_zero_at_tau_1": abs(float(exact_regret(0.25, 1.0)))
        <= float(tolerances["theoretical_zero"]),
        "crossing_at_7_over_47": abs(float(witness_relative_gain(7.0 / 47.0)))
        <= float(tolerances["center_atol"]),
        "negative_below_crossing": float(witness_relative_gain(0.1)) < 0.0,
        "positive_above_crossing": float(witness_relative_gain(1.0)) > 0.0,
    }
    return {
        "status": "PASS" if not failures and all(special.values()) else "FAIL",
        "rows": rows,
        "special_checks": special,
        "failed_tau": failures,
    }


def _risk_resolution_tolerance(
    candidate_risk: float,
    exact_value: float,
    tolerances: Mapping[str, float],
) -> float:
    """Frozen section-10 scalar-risk comparison resolution."""
    return float(tolerances["scalar_atol"]) + float(tolerances["scalar_rtol"]) * max(
        abs(float(candidate_risk)), abs(float(exact_value))
    )


def _optimizer_candidate_checks(
    *,
    rho_hat: float,
    tau: float,
    solver_objective: float,
    dense_minimum: float,
    tolerances: Mapping[str, float],
) -> dict[str, Any]:
    """Post-solver diagnostics; the golden-section objective remains raw risk."""
    closed_rho = float(rho_star(tau))
    closed_value = float(optimal_value(tau))
    parameter_error = abs(float(rho_hat) - closed_rho)
    parameter_reference_passed = parameter_error <= float(
        tolerances["optimizer_parameter"]
    )
    regret = float(exact_regret(rho_hat, tau))
    risk_resolution = _risk_resolution_tolerance(
        solver_objective, closed_value, tolerances
    )
    binding_checks = {
        "solver_objective_finite": bool(np.isfinite(solver_objective)),
        "objective_vs_exact_value": scalar_close(
            solver_objective,
            closed_value,
            float(tolerances["scalar_atol"]),
            float(tolerances["scalar_rtol"]),
        ),
        "dense_crosscheck": solver_objective <= dense_minimum + 1e-10,
        "exact_regret_nonnegative": regret >= -1e-10,
        "exact_regret_within_risk_resolution": regret <= risk_resolution,
    }
    return {
        "rho_star": closed_rho,
        "V": closed_value,
        "parameter_recovery_reference": {
            "label": "PARAMETER_RECOVERY_REFERENCE_CHECK",
            "binding": False,
            "absolute_error": parameter_error,
            "threshold": float(tolerances["optimizer_parameter"]),
            "passed": parameter_reference_passed,
        },
        "exact_regret_at_candidate": regret,
        "risk_resolution_tolerance": risk_resolution,
        "binding_checks": binding_checks,
        "binding_passed": all(binding_checks.values()),
    }


def _optimizer_case(grids: Mapping[str, np.ndarray], config: Mapping[str, Any]) -> dict[str, Any]:
    tolerances = config["tolerances"]
    rows = []
    binding_failures: list[float] = []
    parameter_reference_misses: list[float] = []
    parameter_profile: list[dict[str, Any]] = []
    dense_cross = grids["dense_rho"]
    for tau in grids["dense_tau"]:
        objective = lambda rho, t=float(tau): np.float64(rational_risk(rho, t))
        try:
            result = golden_section_minimize(
                objective,
                interval_tolerance=float(tolerances["optimizer_interval"]),
                maximum_iterations=int(tolerances["optimizer_max_iterations"]),
            )
            dense_values = np.asarray(
                [rational_risk(rho, tau) for rho in dense_cross], dtype=np.float64
            )
            if not np.all(np.isfinite(dense_values)):
                raise NumericalFailure("B-OPT-01 dense cross-check is nonfinite")
        except NumericalFailure as error:
            binding_failures.append(float(tau))
            rows.append(
                {
                    "tau": float(tau),
                    "solver_status": "FAIL",
                    "error": str(error),
                    "binding_checks": {"solver_completed": False},
                }
            )
            continue
        dense_minimum = float(np.min(dense_values))
        diagnostics = _optimizer_candidate_checks(
            rho_hat=result.x,
            tau=float(tau),
            solver_objective=result.objective,
            dense_minimum=dense_minimum,
            tolerances=tolerances,
        )
        binding_checks = {
            "solver_completed": True,
            **diagnostics["binding_checks"],
        }
        parameter_reference = diagnostics["parameter_recovery_reference"]
        parameter_profile.append(
            {
                "tau": float(tau),
                "absolute_error": parameter_reference["absolute_error"],
                "passed": parameter_reference["passed"],
            }
        )
        if not parameter_reference["passed"]:
            parameter_reference_misses.append(float(tau))
        if not all(binding_checks.values()):
            binding_failures.append(float(tau))
        rows.append(
            {
                "tau": float(tau),
                "solver_status": "PASS",
                "solver": result.to_dict(),
                "rho_star": diagnostics["rho_star"],
                "V": diagnostics["V"],
                "dense_minimum": dense_minimum,
                "parameter_recovery_reference": parameter_reference,
                "exact_regret_at_candidate": diagnostics["exact_regret_at_candidate"],
                "risk_resolution_tolerance": diagnostics["risk_resolution_tolerance"],
                "binding_checks": binding_checks,
            }
        )
    parameter_errors = [row["absolute_error"] for row in parameter_profile]
    return {
        "status": "PASS" if not binding_failures else "FAIL",
        "rows": rows,
        "failed_tau": binding_failures,
        "binding_failed_tau": binding_failures,
        "parameter_recovery_reference_summary": {
            "label": "PARAMETER_RECOVERY_REFERENCE_CHECK",
            "binding": False,
            "threshold": float(tolerances["optimizer_parameter"]),
            "evaluated_count": len(parameter_profile),
            "pass_count": len(parameter_profile) - len(parameter_reference_misses),
            "miss_count": len(parameter_reference_misses),
            "maximum_absolute_error": max(parameter_errors) if parameter_errors else None,
            "median_absolute_error": float(np.median(parameter_errors))
            if parameter_errors
            else None,
        },
        "parameter_recovery_reference_miss_tau": parameter_reference_misses,
        "parameter_recovery_profile": parameter_profile,
    }


def _regret_case(risk_rows: Sequence[Mapping[str, Any]], tolerances: Mapping[str, float]) -> dict[str, Any]:
    maximum_error = 0.0
    minimum_regret = math.inf
    failures: list[dict[str, float]] = []
    sampled_optimizer_count = 0
    sampled_nonoptimizer_count = 0
    for row in risk_rows:
        rho, tau = float(row["rho"]), float(row["tau"])
        difference = float(row["rational"] - optimal_value(tau))
        reference = float(exact_regret(rho, tau))
        optimizer = float(rho_star(tau))
        is_sampled_optimizer = rho == optimizer
        sampled_optimizer_count += int(is_sampled_optimizer)
        sampled_nonoptimizer_count += int(not is_sampled_optimizer)
        zero_iff_sampled_optimizer = (
            abs(reference) <= float(tolerances["theoretical_zero"])
            if is_sampled_optimizer
            else reference > 0.0
        )
        maximum_error = max(maximum_error, abs(difference - reference))
        minimum_regret = min(minimum_regret, reference)
        if not scalar_close(
            difference,
            reference,
            float(tolerances["scalar_atol"]),
            float(tolerances["scalar_rtol"]),
        ) or reference < float(tolerances["nonnegative_floor"]) or not zero_iff_sampled_optimizer:
            failures.append({"rho": rho, "tau": tau})
    return {
        "status": "PASS" if not failures else "FAIL",
        "max_identity_error": maximum_error,
        "minimum_regret": minimum_regret,
        "sampled_optimizer_count": sampled_optimizer_count,
        "sampled_nonoptimizer_count": sampled_nonoptimizer_count,
        "zero_iff_rule": (
            "exact-regret <= theoretical_zero at explicitly sampled rho_star; "
            "exact-regret strictly positive at every other registered point"
        ),
        "failed_points": failures,
    }


def _full_case(model: FWV1, tau_values: np.ndarray, tolerances: Mapping[str, float]) -> dict[str, Any]:
    rows = []
    failures = []
    for tau in tau_values:
        full = ambient_full_risk(model, float(tau))
        value = float(optimal_value(tau))
        family = ambient_measurement_risk(model, float(rho_star(tau)), float(tau))["risk"]
        checks = {
            "full_vs_V": scalar_close(full, value, float(tolerances["scalar_atol"]), float(tolerances["scalar_rtol"])),
            "family_vs_V": scalar_close(family, value, float(tolerances["scalar_atol"]), float(tolerances["scalar_rtol"])),
        }
        if not all(checks.values()):
            failures.append(float(tau))
        rows.append({"tau": float(tau), "full": full, "V": value, "family": family, "checks": checks})
    center = scalar_close(
        float(optimal_value(1.0)),
        11.0,
        float(tolerances["center_atol"]),
        float(tolerances["center_rtol"]),
    )
    return {"status": "PASS" if not failures and center else "FAIL", "rows": rows, "center_V_11": center}


def _factorization_case(model: FWV1, config: Mapping[str, Any]) -> dict[str, Any]:
    tolerances = config["tolerances"]
    rows = []
    failures = []
    for token in config["b"]["factorization_tau_tokens"]:
        tau = float(parse_numeric_token(token))
        sigma_tau = np.eye(64) + tau * model.sigma0
        conditional = cholesky_solve(sigma_tau, model.target)
        rho = float(rho_star(tau))
        optimal_measurement = measurement(model, rho)
        diagnostic = rank_diagnostics(optimal_measurement, float(tolerances["rank_relative"]))
        if not diagnostic.full_column_rank:
            raise NumericalFailure(f"B-FAC optimal measurement rank failure at tau={token}")
        basis, _ = np.linalg.qr(optimal_measurement, mode="reduced")
        column_residual = np.linalg.norm(conditional - basis @ (basis.T @ conditional), ord="fro") / max(
            1.0, np.linalg.norm(conditional, ord="fro")
        )
        l_matrix = l_tau_matrix(tau)
        factor_residual = np.linalg.norm(
            conditional - optimal_measurement @ np.kron(l_matrix, np.eye(4)), ord="fro"
        ) / max(1.0, np.linalg.norm(conditional, ord="fro"))
        analytic_eigenvalues = np.sort(np.asarray(l_tau_eigenvalues(tau)))
        numeric_eigenvalues = np.linalg.eigvalsh(l_matrix)
        eigen_check = np.allclose(
            numeric_eigenvalues,
            analytic_eigenvalues,
            atol=float(tolerances["matrix_atol"]),
            rtol=float(tolerances["matrix_rtol"]),
        )
        identity_measurement = measurement(model, 0.0)
        identity_diagnostic = rank_diagnostics(
            identity_measurement, float(tolerances["rank_relative"])
        )
        if not identity_diagnostic.full_column_rank:
            raise NumericalFailure(
                f"B-FAC identity control rank failure at tau={token}"
            )
        identity_basis, _ = np.linalg.qr(identity_measurement, mode="reduced")
        identity_residual = np.linalg.norm(
            conditional - identity_basis @ (identity_basis.T @ conditional), ord="fro"
        ) / max(1.0, np.linalg.norm(conditional, ord="fro"))
        checks = {
            "column_residual": column_residual <= float(tolerances["normalized_residual"]),
            "factor_residual": factor_residual <= float(tolerances["normalized_residual"]),
            "eigenvalues": bool(eigen_check),
            "identity_nonzero": identity_residual > float(tolerances["theoretical_zero"]),
        }
        if not all(checks.values()):
            failures.append(token)
        rows.append(
            {
                "tau_token": token,
                "tau": tau,
                "rho_star": rho,
                "column_residual": column_residual,
                "factorization_residual": factor_residual,
                "analytic_eigenvalues": analytic_eigenvalues,
                "numeric_eigenvalues": numeric_eigenvalues,
                "identity_column_residual": identity_residual,
                "identity_measurement_rank": identity_diagnostic.to_dict(),
                "checks": checks,
            }
        )
    return {"status": "PASS" if not failures else "FAIL", "rows": rows, "failed_tau_tokens": failures}


def _subspace_metrics(model: FWV1, tau1: float, tau2: float) -> dict[str, Any]:
    q1, _ = np.linalg.qr(measurement(model, float(rho_star(tau1))), mode="reduced")
    q2, _ = np.linalg.qr(measurement(model, float(rho_star(tau2))), mode="reduced")
    singular = np.clip(np.linalg.svd(q1.T @ q2, compute_uv=False), 0.0, 1.0)
    angles = np.arccos(singular)
    p1, p2 = q1 @ q1.T, q2 @ q2.T
    projector_distance = np.linalg.norm(p1 - p2, ord="fro") / math.sqrt(32.0)
    forward = np.linalg.norm(q2 - p1 @ q2, ord="fro") / 4.0
    reverse = np.linalg.norm(q1 - p2 @ q1, ord="fro") / 4.0
    return {
        "tau_1": tau1,
        "tau_2": tau2,
        "principal_angles": angles,
        "maximum_principal_angle": float(np.max(angles)),
        "projector_distance": float(projector_distance),
        "forward_residual": float(forward),
        "reverse_residual": float(reverse),
    }


def _snr_case(model: FWV1, grids: Mapping[str, np.ndarray], config: Mapping[str, Any]) -> dict[str, Any]:
    threshold = float(config["tolerances"]["subspace_nonzero"])
    rho_core = rho_star(grids["core_tau"])
    rho_dense = rho_star(grids["dense_tau"])
    formula_checks = {
        "core_strict": bool(np.all(np.diff(rho_core) > 0.0)),
        "dense_strict": bool(np.all(np.diff(rho_dense) > 0.0)),
        "domain": bool(np.all((rho_dense > 0.0) & (rho_dense < 8.0 / 25.0))),
    }
    pair_rows = []
    pair_checks = []
    for first, second in config["b"]["subspace_pairs"]:
        row = _subspace_metrics(model, float(parse_numeric_token(first)), float(parse_numeric_token(second)))
        passed = (
            row["maximum_principal_angle"] > threshold
            and row["projector_distance"] > threshold
            and row["forward_residual"] > threshold
            and row["reverse_residual"] > threshold
        )
        row["passed"] = passed
        pair_rows.append(row)
        pair_checks.append(passed)
    descriptive = [
        _subspace_metrics(model, float(first), float(second))
        for first, second in zip(grids["dense_tau"][:-1], grids["dense_tau"][1:], strict=True)
    ]
    return {
        "status": "PASS" if all(formula_checks.values()) and all(pair_checks) else "FAIL",
        "formula_checks": formula_checks,
        "binding_pairs": pair_rows,
        "dense_adjacent_descriptive_profile": descriptive,
    }


def _rank_cases(model: FWV1, config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    tolerances = config["tolerances"]
    exact = rank_diagnostics(measurement(model, -0.5), float(tolerances["rank_relative"]))
    api_result = "NO_EXCEPTION"
    api_diagnostics: Mapping[str, Any] | None = None
    ordinary_risk_evaluated = False
    try:
        ordinary = ambient_measurement_risk(model, -0.5, 1.0)
        ordinary_risk_evaluated = "risk" in ordinary
    except OutOfDomainRankError as error:
        api_result = error.structured_result
        api_diagnostics = error.diagnostics.to_dict()
    rank01 = {
        "status": (
            "EXPECTED_OUT_OF_DOMAIN"
            if exact.rank == 12
            and api_result == "OUT_OF_DOMAIN_RANK_12"
            and not ordinary_risk_evaluated
            else "FAIL"
        ),
        "structured_result": api_result,
        "rank": exact.to_dict(),
        "api_rank_diagnostics": api_diagnostics,
        "full_rank_api_exercised": True,
        "ordinary_risk_evaluated": ordinary_risk_evaluated,
        "reduced_basis_risk_evaluated": False,
    }
    rows = []
    failure = False
    for offset in (1e-2, 1e-4, 1e-6):
        for sign in (-1.0, 1.0):
            rho = -0.5 + sign * offset
            matrix = measurement(model, rho)
            diag = rank_diagnostics(matrix, float(tolerances["rank_relative"]))
            gram = matrix.T @ (np.eye(64) + model.sigma0) @ matrix
            gram_condition = rank_diagnostics(gram).condition_number
            classification = (
                "ILL_CONDITIONED_FULL_RANK"
                if gram_condition > float(tolerances["condition_gate"])
                else "FULL_RANK"
            )
            if not diag.full_column_rank:
                failure = True
            rows.append(
                {
                    "rho": rho,
                    "offset": offset,
                    "side": "left" if sign < 0 else "right",
                    "rank": diag.to_dict(),
                    "normal_matrix_condition": gram_condition,
                    "classification": classification,
                }
            )
    rank02 = {"status": "FAIL" if failure else "PASS", "rows": rows}
    return rank01, rank02


def _schedule_tau(config: Mapping[str, Any]) -> np.ndarray:
    d = config["d"]
    indices = np.arange(int(d["timesteps"]), dtype=np.float64)
    beta = np.float64(d["beta_start"]) + (
        np.float64(d["beta_end"]) - np.float64(d["beta_start"])
    ) * indices / np.float64(int(d["timesteps"]) - 1)
    alpha = np.empty_like(beta)
    accumulator = np.float64(1.0)
    for index, value in enumerate(beta):
        accumulator = np.float64(accumulator * np.float64(1.0 - value))
        alpha[index] = accumulator
    return alpha / (1.0 - alpha)


def _static_policy(
    name: str,
    tau_values: Sequence[float],
    weights: Sequence[float],
    grids: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    tolerances = config["tolerances"]
    objective = lambda rho: incremental_weighted_objective(rho, tau_values, weights)
    try:
        solution = golden_section_minimize(
            objective,
            interval_tolerance=float(tolerances["optimizer_interval"]),
            maximum_iterations=int(tolerances["optimizer_max_iterations"]),
        )
    except NumericalFailure as error:
        return {
            "name": name,
            "tau": list(tau_values),
            "weights": list(weights),
            "optimizer_status": "FAIL",
            "optimizer_error": str(error),
        }
    policy = float(incremental_optimal_objective(tau_values, weights))
    risk_difference_advantage = solution.objective - policy
    exact_regret_advantage = np.float64(0.0)
    for tau, weight in zip(tau_values, weights, strict=True):
        exact_regret_advantage = np.float64(
            exact_regret_advantage
            + np.float64(weight) * np.float64(exact_regret(solution.x, tau))
        )
    dense_values = np.asarray(
        [objective(rho) for rho in grids["dense_rho"]], dtype=np.float64
    )
    if not np.all(np.isfinite(dense_values)):
        return {
            "name": name,
            "tau": list(tau_values),
            "weights": list(weights),
            "optimizer_status": "FAIL",
            "optimizer_error": "dense cross-check contains nonfinite objective",
        }
    dense_minimum = float(np.min(dense_values))
    risk_resolution = _risk_resolution_tolerance(
        solution.objective, policy, tolerances
    )
    return {
        "name": name,
        "tau": list(tau_values),
        "weights": list(weights),
        "optimizer_status": "PASS",
        "static_solver": solution.to_dict(),
        "policy_risk": policy,
        "static_risk_vs_policy_check": scalar_close(
            solution.objective,
            policy,
            float(tolerances["scalar_atol"]),
            float(tolerances["scalar_rtol"]),
        ),
        "risk_resolution_tolerance": risk_resolution,
        "advantage": float(exact_regret_advantage),
        "risk_difference_advantage": risk_difference_advantage,
        "exact_regret_advantage": float(exact_regret_advantage),
        "advantage_identity_check": scalar_close(
            risk_difference_advantage,
            float(exact_regret_advantage),
            float(tolerances["scalar_atol"]),
            float(tolerances["scalar_rtol"]),
        ),
        "dense_minimum": dense_minimum,
        "dense_crosscheck": solution.objective <= dense_minimum + 1e-10,
    }


def _finalize_dirac_policy(
    dirac: dict[str, Any], tolerances: Mapping[str, float]
) -> dict[str, Any]:
    """Apply the v1.2 Dirac-law binding equality and reference diagnostic."""
    zero = float(tolerances["theoretical_zero"])
    if dirac["optimizer_status"] == "PASS":
        parameter_error = abs(dirac["static_solver"]["x"] - 0.25)
        dirac["parameter_recovery_reference"] = {
            "label": "PARAMETER_RECOVERY_REFERENCE_CHECK",
            "binding": False,
            "absolute_error": parameter_error,
            "threshold": float(tolerances["optimizer_parameter"]),
            "passed": parameter_error <= float(tolerances["optimizer_parameter"]),
        }
        dirac["binding_checks"] = {
            "static_solver_completed_and_finite": bool(
                np.isfinite(dirac["static_solver"]["objective"])
            ),
            "static_risk_equals_exact_policy_risk": bool(
                dirac["static_risk_vs_policy_check"]
            ),
            "risk_difference_advantage_theoretical_zero": abs(
                dirac["risk_difference_advantage"]
            )
            <= zero,
            "exact_regret_advantage_within_risk_resolution": abs(
                dirac["exact_regret_advantage"]
            )
            <= dirac["risk_resolution_tolerance"],
            "dense_crosscheck": bool(dirac["dense_crosscheck"]),
        }
    else:
        dirac["parameter_recovery_reference"] = {
            "label": "PARAMETER_RECOVERY_REFERENCE_CHECK",
            "binding": False,
            "absolute_error": None,
            "threshold": float(tolerances["optimizer_parameter"]),
            "passed": False,
        }
        dirac["binding_checks"] = {"static_solver_completed_and_finite": False}
    dirac["status"] = "PASS" if all(dirac["binding_checks"].values()) else "FAIL"
    return dirac


def _policy_cases(grids: Mapping[str, np.ndarray], config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    cases["B-POL-01"] = _static_policy("dirac-1", [1.0], [1.0], grids, config)
    cases["B-POL-02"] = _static_policy("two-point", [0.1, 1.0], [0.5, 0.5], grids, config)
    spread = [1e-3, 1e-1, 1.0, 10.0, 1e3]
    cases["B-POL-03"] = _static_policy("five-point", spread, [0.2] * 5, grids, config)
    schedule = _schedule_tau(config)
    cases["B-POL-04"] = _static_policy(
        "e2-uniform", schedule.tolist(), [1.0 / 1000.0] * 1000, grids, config
    )
    near = []
    for epsilon in (1e-1, 1e-2, 1e-3):
        row = _static_policy(
            f"near-dirac-{epsilon:.0e}",
            [1.0 - epsilon, 1.0 + epsilon],
            [0.5, 0.5],
            grids,
            config,
        )
        row["epsilon"] = epsilon
        near.append(row)
    cases["B-POL-05"] = {"laws": near}

    zero = float(config["tolerances"]["theoretical_zero"])
    cases["B-POL-01"] = _finalize_dirac_policy(
        cases["B-POL-01"], config["tolerances"]
    )
    for case_id in ("B-POL-02", "B-POL-03", "B-POL-04"):
        cases[case_id]["status"] = (
            "PASS"
            if cases[case_id]["optimizer_status"] == "PASS"
            and cases[case_id]["advantage"] > zero
            and cases[case_id]["advantage_identity_check"]
            and cases[case_id]["dense_crosscheck"]
            else "FAIL"
        )
    optimizer_ok = all(row["optimizer_status"] == "PASS" for row in near)
    advantages = [row.get("advantage") for row in near]
    near_ok = optimizer_ok and (
        advantages[0] > advantages[1] > advantages[2] > 0.0
        and all(
            row["dense_crosscheck"] and row["advantage_identity_check"] for row in near
        )
    )
    cases["B-POL-05"]["ordered_advantages"] = advantages
    cases["B-POL-05"]["status"] = "PASS" if near_ok else "FAIL"
    return cases


def _post_cases(model: FWV1, grids: Mapping[str, np.ndarray], config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    zero = float(config["tolerances"]["theoretical_zero"])
    projector = model.target @ model.target.T
    e0, e1 = np.eye(8)[:, 0], np.eye(8)[:, 1]
    p0 = (e0 + e1) / math.sqrt(2.0)
    q0 = (e0 - e1) / math.sqrt(2.0)
    discarded = np.kron(q0, p0)
    rows = []
    failures = []
    for rho in grids["core_rho"]:
        if rho == -0.5:
            continue
        transform = model.target.T @ a_rho(model, float(rho))
        operator_residual = np.linalg.norm(transform @ (np.eye(64) - projector), ord="fro")
        explicit_squared = float(np.linalg.norm(transform @ discarded) ** 2)
        reference = float(rho * rho / 2.0)
        checks = {
            "explicit": scalar_close(
                explicit_squared,
                reference,
                float(config["tolerances"]["scalar_atol"]),
                float(config["tolerances"]["scalar_rtol"]),
            ),
            "zero_iff": operator_residual <= zero if rho == 0.0 else operator_residual > zero,
        }
        if not all(checks.values()):
            failures.append(float(rho))
        rows.append(
            {
                "rho": float(rho),
                "operator_residual": float(operator_residual),
                "explicit_squared": explicit_squared,
                "reference_rho_squared_over_2": reference,
                "checks": checks,
            }
        )
    post01 = {"status": "PASS" if not failures else "FAIL", "rows": rows, "failed_rho": failures}

    separation_rows = []
    separation_failures = []
    for tau in grids["dense_tau"]:
        full = ambient_full_risk(model, float(tau))
        pre = float(optimal_value(tau))
        post = float(rational_risk(0.0, tau))
        gap = float(identity_gap(tau))
        observed_gap = post - pre
        sign_resolved = gap > zero
        checks = {
            "pre_full": scalar_close(full, pre, float(config["tolerances"]["scalar_atol"]), float(config["tolerances"]["scalar_rtol"])),
            "exact_reference_positive": gap > 0.0,
            "subtraction_sign_if_numerically_resolved": observed_gap > 0.0 if sign_resolved else True,
            "gap": scalar_close(observed_gap, gap, float(config["tolerances"]["scalar_atol"]), float(config["tolerances"]["scalar_rtol"])),
        }
        if not all(checks.values()):
            separation_failures.append(float(tau))
        separation_rows.append(
            {
                "tau": float(tau),
                "full": full,
                "pre": pre,
                "post": post,
                "observed_risk_subtraction": observed_gap,
                "identity_gap": gap,
                "strictness_source": "positive exact identity-gap formula",
                "subtraction_sign_status": (
                    "RESOLVED_POSITIVE"
                    if sign_resolved and observed_gap > 0.0
                    else "NUMERICALLY_UNRESOLVED_POSITIVE_REFERENCE"
                    if not sign_resolved
                    else "SIGN_MISMATCH"
                ),
                "checks": checks,
            }
        )
    post02 = {
        "status": "PASS" if not separation_failures else "FAIL",
        "rows": separation_rows,
        "failed_tau": separation_failures,
        "extra_linear_post_candidates_evaluated": False,
    }
    return post01, post02


def _monte_carlo(model: FWV1, config: Mapping[str, Any]) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    seen: dict[tuple[str, str], str] = {}
    root_seed = int(config["seeds"]["b_monte_carlo"])
    for tau_token in config["b"]["mc_tau_tokens"]:
        tau = float(parse_numeric_token(tau_token))
        policies = (
            ("identity", 0.0),
            ("optimal", float(rho_star(tau))),
            ("quarter", 0.25),
        )
        value_to_case: dict[str, str] = {}
        for tag, rho in policies:
            rho_key = format(rho, ".17g")
            case_id = f"B-MC/{tau_token}/{tag}"
            if rho_key in value_to_case:
                outputs[case_id] = {
                    "status": "PASS",
                    "deduplicated": True,
                    "alias_of": value_to_case[rho_key],
                    "tau_token": tau_token,
                    "policy_tag": tag,
                    "rho": rho,
                }
                continue
            value_to_case[rho_key] = case_id
            outputs[case_id] = fwv1_monte_carlo(
                model,
                tau=tau,
                tau_token=tau_token,
                rho=rho,
                policy_tag=tag,
                root_seed=root_seed,
                mc_config=config["monte_carlo"],
            )
    return outputs


def run_fwv1(config: Mapping[str, Any]) -> dict[str, Any]:
    model = build_fwv1()
    grids = _grids(config)
    tolerances = config["tolerances"]
    construction = {
        "target_orthonormal": bool(np.allclose(model.target.T @ model.target, np.eye(16), atol=1e-10, rtol=1e-9)),
        "sigma0_spd": bool(np.linalg.eigvalsh(model.sigma0)[0] > 0.0),
        "dimensions": {"target": list(model.target.shape), "sigma0": list(model.sigma0.shape)},
    }
    risk_case, risk_rows = _risk_grid_case(model, grids, tolerances)
    cases: dict[str, Any] = {
        "B-RSK-01": risk_case,
        "B-ID-01": _identity_case(model, grids["dense_tau"], tolerances),
        "B-WIT-01": _witness_case(grids["dense_tau"], tolerances),
        "B-OPT-01": _optimizer_case(grids, config),
        "B-REG-01": _regret_case(risk_rows, tolerances),
        "B-FULL-01": _full_case(model, grids["dense_tau"], tolerances),
        "B-FAC-01": _factorization_case(model, config),
        "B-SNR-01": _snr_case(model, grids, config),
    }
    cases["B-RNK-01"], cases["B-RNK-02"] = _rank_cases(model, config)
    cases.update(_policy_cases(grids, config))
    cases["B-POST-01"], cases["B-POST-02"] = _post_cases(model, grids, config)
    mc = _monte_carlo(model, config)
    mc_status = overall_status(item["status"] for item in mc.values())
    verdict = overall_status([item["status"] for item in cases.values()] + [mc_status])
    if not all((construction["target_orthonormal"], construction["sigma0_spd"])):
        verdict = "FAIL"
    return {
        "module": "B",
        "theory_scope": "Theorem 2 and Corollaries 2.1-2.2 in FW-v1",
        "evidence_type": "deterministic matrix/rational regression plus finite-sample Monte Carlo corroboration",
        "claim_status_changed": False,
        "verdict": verdict,
        "construction": construction,
        "grid_metadata": {
            "core_tau": grids["core_tau"],
            "dense_tau_count": int(grids["dense_tau"].size),
            "core_rho": grids["core_rho"],
            "dense_rho_count": int(grids["dense_rho"].size),
        },
        "cases": cases,
        "risk_grid_rows": risk_rows,
        "mc_summaries": mc,
    }


def smoke_check(config: Mapping[str, Any]) -> dict[str, Any]:
    model = build_fwv1()
    rho, tau = 0.13, 0.37
    ambient = ambient_measurement_risk(model, rho, tau)["risk"]
    reduced = four_mode_risk(rho, tau)
    closed = float(rational_risk(rho, tau))
    optimizer = golden_section_minimize(lambda value: np.float64(rational_risk(value, tau)))
    optimizer_diagnostics = _optimizer_candidate_checks(
        rho_hat=optimizer.x,
        tau=tau,
        solver_objective=optimizer.objective,
        dense_minimum=optimizer.objective,
        tolerances=config["tolerances"],
    )
    checks = {
        "ambient_vs_reduced": scalar_close(ambient, reduced, 1e-10, 1e-9),
        "ambient_vs_closed": scalar_close(ambient, closed, 1e-10, 1e-9),
        "optimizer_candidate_quality": optimizer_diagnostics["binding_passed"],
        "nonformal_measurement_full_rank": rank_diagnostics(measurement(model, rho)).rank == 16,
    }
    return {
        "mode": "SMOKE_ONLY_NON_FORMAL_POINT",
        "formal_case_ids_used": [],
        "rho": rho,
        "tau": tau,
        "parameter_recovery_reference": optimizer_diagnostics[
            "parameter_recovery_reference"
        ],
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = base_parser(__doc__ or "T0 FW-v1 validator")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.smoke:
            result = smoke_check(config)
            print(result)
            if not result["passed"]:
                raise SystemExit(1)
            return
        source_lock = verify_source_lock(config)
    except SourceDriftError as error:
        print(f"SOURCE_DRIFT_REVIEW_REQUIRED: {error}")
        raise SystemExit(2) from error
    run_dir = new_run_directory(config, "b")
    result = run_fwv1(config)
    write_json(run_dir / "RUN_METADATA.json", {"module": "B", "environment": environment_metadata(), "source_lock": source_lock})
    write_json(run_dir / "B_FWV1_RESULTS.json", result)
    write_json(run_dir / "MC_SUMMARIES.json", result["mc_summaries"])
    print(run_dir)
    if result["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
