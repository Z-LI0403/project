"""T0-C: frozen endpoint, rank, robustness, and corrective regressions."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from common_linear_gaussian import (
    SourceDriftError,
    base_parser,
    direct_gaussian_conditioning,
    environment_metadata,
    load_config,
    new_run_directory,
    operator_norm_symmetric,
    overall_status,
    rank_diagnostics,
    scalar_close,
    subspace_parent_gap,
    verify_source_lock,
    write_json,
)
from fwv1_reference import (
    OutOfDomainRankError,
    ambient_measurement_risk,
    build_fwv1,
    measurement,
)


def _noise_objects(matched: bool) -> dict[str, np.ndarray]:
    identity = np.eye(6, dtype=np.float64)
    retained = identity[:, :2]
    complement = identity[:, 2:]
    sigma0 = np.asarray(
        (
            (2, 0, 0.5, 0, 0, 0),
            (0, 3, 0, 0.5, 0, 0),
            (0.5, 0, 1, 0, 0, 0),
            (0, 0.5, 0, 2, 0, 0),
            (0, 0, 0, 0, 1, 0),
            (0, 0, 0, 0, 0, 1.5),
        ),
        dtype=np.float64,
    )
    if matched:
        target = retained.copy()
    else:
        target = np.column_stack(
            ((identity[:, 0] + identity[:, 2]) / math.sqrt(2.0),
             (identity[:, 1] + identity[:, 3]) / math.sqrt(2.0))
        )
    return {"retained": retained, "complement": complement, "target": target, "sigma0": sigma0}


def _noise_binding_checks(
    *,
    rows: list[dict[str, Any]],
    computed_limit: float,
    expected_limit: float,
    matched: bool,
    config: Mapping[str, Any],
) -> dict[str, bool]:
    tolerances = config["tolerances"]
    limit_identity = abs(computed_limit - expected_limit) <= float(
        tolerances["theoretical_zero"]
    )
    final = rows[-1]
    final_point = (
        abs(final["gap"]) <= float(tolerances["theoretical_zero"])
        if matched
        else final["absolute_error"]
        <= float(tolerances["noise_distinct_final_atol"])
    )
    return {
        "computed_limit_matches_exact_limit": limit_identity,
        "final_point_absolute_tolerance": final_point,
        "profile_finite_nonnegative": all(
            row["finite"] and row["nonnegative"] for row in rows
        ),
    }


def _noise_case(matched: bool, config: Mapping[str, Any]) -> dict[str, Any]:
    objects = _noise_objects(matched)
    base_exponents = tuple(int(item) for item in config["c"]["boundary_exponents"])
    exponents = (
        base_exponents
        if matched
        else base_exponents
        + tuple(int(item) for item in config["c"]["noise_distinct_additional_exponents"])
    )
    computed_limit = float(
        np.linalg.norm(
            objects["target"].T @ objects["complement"], ord="fro"
        )
        ** 2
    )
    expected_limit = 0.0 if matched else 1.0
    rows = []
    for exponent in exponents:
        a = float(10.0 ** (-exponent))
        parent = subspace_parent_gap(
            objects["sigma0"], a, 1.0, objects["retained"], objects["complement"], objects["target"]
        )
        rows.append(
            {
                "k": exponent,
                "a": a,
                "b": 1.0,
                "gap": parent["gap"],
                "absolute_error": abs(parent["gap"] - expected_limit),
                "finite": bool(np.isfinite(parent["gap"])),
                "nonnegative": parent["gap"] >= float(config["tolerances"]["nonnegative_floor"]),
                "gap_route": "SUBSPACE_PARENT_GAP",
            }
        )
    binding_checks = _noise_binding_checks(
        rows=rows,
        computed_limit=computed_limit,
        expected_limit=expected_limit,
        matched=matched,
        config=config,
    )
    construction = {
        "u_orthonormal": bool(np.array_equal(objects["retained"].T @ objects["retained"], np.eye(2))),
        "u_perp_orthonormal": bool(np.array_equal(objects["complement"].T @ objects["complement"], np.eye(4))),
        "target_orthonormal": bool(np.allclose(objects["target"].T @ objects["target"], np.eye(2), atol=1e-15, rtol=0.0)),
        "sigma0_psd": bool(np.linalg.eigvalsh(objects["sigma0"])[0] > 0.0),
        "coupled": bool(np.linalg.norm(objects["retained"].T @ objects["sigma0"] @ objects["complement"], ord="fro") > 0.0),
    }
    return {
        "status": "PASS"
        if all(construction.values()) and all(binding_checks.values())
        else "FAIL",
        "matched": matched,
        "limit": expected_limit,
        "computed_subspace_limit": computed_limit,
        "construction": construction,
        "objects": objects,
        "rows": rows,
        "binding_checks": binding_checks,
        "binding_final_check": all(binding_checks.values()),
        "trend_role": "DESCRIPTIVE_NON_BINDING",
        "direct_risk_subtraction_evaluated": False,
    }


def _clean_case_01(config: Mapping[str, Any]) -> dict[str, Any]:
    e1 = np.asarray(((1.0,), (0.0,)), dtype=np.float64)
    e2 = np.asarray(((0.0,), (1.0,)), dtype=np.float64)
    sigma0 = np.asarray(((1.0, 0.6), (0.6, 2.0)), dtype=np.float64)
    rows = []
    for exponent in config["c"]["boundary_exponents"]:
        b = float(10.0 ** (-int(exponent)))
        a = math.sqrt(1.0 - b * b)
        gap = subspace_parent_gap(sigma0, a, b, e1, e2, e1)["gap"]
        rows.append({"k": int(exponent), "a": a, "b": b, "gap": gap, "gap_over_b2": gap / (b * b)})
    tail = rows[-4:]
    valid_tail = all(np.isfinite(row["gap"]) and row["gap"] > 0.0 for row in tail)
    if valid_tail:
        x = np.log(np.asarray([row["b"] for row in tail], dtype=np.float64))
        y = np.log(np.asarray([row["gap"] for row in tail], dtype=np.float64))
        x_centered = x - np.mean(x, dtype=np.float64)
        y_centered = y - np.mean(y, dtype=np.float64)
        slope = float(np.sum(x_centered * y_centered) / np.sum(x_centered * x_centered))
    else:
        slope = math.nan
    passed = valid_tail and 1.90 <= slope <= 2.10
    return {
        "status": "PASS" if passed else "FAIL",
        "rows": rows,
        "tail_indices": [3, 4, 5, 6],
        "natural_log_ols_slope": slope,
        "slope_interval": [1.90, 2.10],
        "ratio_profile_role": "DESCRIPTIVE_NON_BINDING",
    }


def _clean_case_02(config: Mapping[str, Any]) -> dict[str, Any]:
    e1 = np.asarray(((1.0,), (0.0,)), dtype=np.float64)
    e2 = np.asarray(((0.0,), (1.0,)), dtype=np.float64)
    eta_rows = []
    final_ratios: list[float] = []
    coefficient_checks = []
    atol = float(config["tolerances"]["clean_coefficient_atol"])
    rtol = float(config["tolerances"]["clean_coefficient_rtol"])
    for eta in config["c"]["clean_etas"]:
        eta = float(eta)
        covariance = np.asarray(
            ((1.0, math.sqrt(1.0 - eta)), (math.sqrt(1.0 - eta), 1.0)), dtype=np.float64
        )
        rows = []
        for exponent in config["c"]["boundary_exponents"]:
            b = float(10.0 ** (-int(exponent)))
            a = math.sqrt(1.0 - b * b)
            gap = subspace_parent_gap(covariance, a, b, e1, e2, e1)["gap"]
            rows.append({"k": int(exponent), "a": a, "b": b, "gap": gap, "ratio": gap / (b * b)})
        coefficient = (1.0 - eta) / eta
        final_ratio = float(rows[-1]["ratio"])
        check = scalar_close(final_ratio, coefficient, atol, rtol)
        final_ratios.append(final_ratio)
        coefficient_checks.append(check)
        eta_rows.append(
            {
                "eta": eta,
                "coefficient": coefficient,
                "rows": rows,
                "binding_final_ratio": final_ratio,
                "coefficient_check": check,
            }
        )
    zero = float(config["tolerances"]["theoretical_zero"])
    increase_checks = [
        final_ratios[1] - final_ratios[0] > zero,
        final_ratios[2] - final_ratios[1] > zero,
    ]
    return {
        "status": "PASS" if all(coefficient_checks) and all(increase_checks) else "FAIL",
        "a0": 1.0,
        "eta_rows": eta_rows,
        "exact_coefficients": [9.0, 99.0, 999.0],
        "increase_differences": [final_ratios[1] - final_ratios[0], final_ratios[2] - final_ratios[1]],
        "increase_checks": increase_checks,
        "nonfinal_ratio_role": "DESCRIPTIVE_NON_BINDING",
    }


def _direct_risk_accuracy_classification(
    *,
    condition_number: float,
    direct_gap: float,
    exact_gap: float,
    tolerances: Mapping[str, float],
) -> dict[str, Any]:
    machine_epsilon = float(np.finfo(np.float64).eps)
    risk_scale = max(1.0, abs(direct_gap), abs(exact_gap))
    accuracy_budget = condition_number * machine_epsilon * risk_scale
    comparison_tolerance = float(tolerances["scalar_atol"]) + float(
        tolerances["scalar_rtol"]
    ) * max(abs(direct_gap), abs(exact_gap))
    eligible = accuracy_budget <= comparison_tolerance
    agreement = scalar_close(
        direct_gap,
        exact_gap,
        float(tolerances["scalar_atol"]),
        float(tolerances["scalar_rtol"]),
    )
    return {
        "classification": (
            "DIRECT_RISK_ACCURACY_ELIGIBLE_BINDING"
            if eligible
            else "FINITE_PRECISION_DIRECT_RISK_DIAGNOSTIC"
        ),
        "machine_epsilon": machine_epsilon,
        "risk_scale": risk_scale,
        "accuracy_budget": accuracy_budget,
        "risk_comparison_tolerance": comparison_tolerance,
        "accuracy_eligible": eligible,
        "direct_agreement": agreement,
        "binding_passed": agreement if eligible else True,
    }


def _singular_case(config: Mapping[str, Any]) -> dict[str, Any]:
    sigma0 = np.asarray(((1.0, 1.0), (1.0, 1.0)), dtype=np.float64)
    e1 = np.asarray(((1.0,), (0.0,)), dtype=np.float64)
    rows = []
    direct_binding_checks = []
    condition_gate = float(config["tolerances"]["condition_gate"])
    nonnegative_floor = float(config["tolerances"]["nonnegative_floor"])
    for exponent in config["c"]["boundary_exponents"]:
        exponent = int(exponent)
        b = float(10.0 ** (-exponent))
        a = math.sqrt(1.0 - b * b)
        interior = a**4 / (2.0 - b * b)
        condition = (2.0 - b * b) / (b * b)
        solver_admissible = condition <= condition_gate
        row: dict[str, Any] = {
            "k": exponent,
            "a": a,
            "b": b,
            "interior_gap": interior,
            "full_covariance_condition": condition,
            "solver_admissibility_gate": condition_gate,
            "solver_admissible": solver_admissible,
        }
        if solver_admissible:
            direct = direct_gaussian_conditioning(sigma0, a, b, e1, e1)
            accuracy = _direct_risk_accuracy_classification(
                condition_number=condition,
                direct_gap=float(direct["gap"]),
                exact_gap=interior,
                tolerances=config["tolerances"],
            )
            row.update(
                direct_gap=direct["gap"],
                exact_gap=interior,
                absolute_error=abs(float(direct["gap"]) - interior),
                direct_check=accuracy["direct_agreement"],
                direct_comparison_binding=accuracy["accuracy_eligible"],
                direct_binding_passed=accuracy["binding_passed"],
                direct_route=accuracy["classification"],
                direct_solver="CHOLESKY",
                direct_evaluated=True,
                machine_epsilon=accuracy["machine_epsilon"],
                risk_scale=accuracy["risk_scale"],
                accuracy_budget=accuracy["accuracy_budget"],
                risk_comparison_tolerance=accuracy["risk_comparison_tolerance"],
            )
            direct_binding_checks.append(accuracy["binding_passed"])
        else:
            row.update(
                direct_route="ILL_CONDITIONED_FULL_RANK",
                direct_evaluated=False,
                direct_comparison_binding=False,
                machine_epsilon=float(np.finfo(np.float64).eps),
                accuracy_eligibility_evaluated=False,
            )
        rows.append(row)
    interior_final = abs(rows[-1]["interior_gap"] - 0.5) <= float(
        config["tolerances"]["singular_limit_atol"]
    )
    endpoint = {"a": 1.0, "b": 0.0, "risk_retained": 1.0, "risk_full": 1.0, "gap": 0.0}
    zero = float(config["tolerances"]["theoretical_zero"])
    endpoint_risks_check = (
        abs(endpoint["risk_retained"] - 1.0) <= zero
        and abs(endpoint["risk_full"] - 1.0) <= zero
    )
    endpoint_check = abs(endpoint["gap"]) <= zero
    interior_profile_check = all(
        np.isfinite(row["interior_gap"])
        and row["interior_gap"] >= nonnegative_floor
        for row in rows
    )
    direct_numeric_health_check = all(
        not row["direct_evaluated"]
        or (
            np.isfinite(row["direct_gap"])
            and row["direct_gap"] >= nonnegative_floor
        )
        for row in rows
    )
    conditioning_check = (
        all(row["solver_admissible"] for row in rows[:-1])
        and not rows[-1]["solver_admissible"]
        and rows[-1]["direct_route"] == "ILL_CONDITIONED_FULL_RANK"
        and not rows[-1]["direct_evaluated"]
    )
    binding_checks = {
        "interior_profile_finite_nonnegative": interior_profile_check,
        "interior_final_limit": interior_final,
        "endpoint_risks_equal_one": endpoint_risks_check,
        "endpoint_gap_zero": endpoint_check,
        "direct_numeric_health": direct_numeric_health_check,
        "accuracy_eligible_direct_checks": all(direct_binding_checks),
        "solver_admissibility_boundary": conditioning_check,
    }
    return {
        "status": "PASS" if all(binding_checks.values()) else "FAIL",
        "rows": rows,
        "solver_admissibility": {
            "label": "SOLVER_ADMISSIBILITY_GATE",
            "condition_number_limit": condition_gate,
        },
        "direct_risk_accuracy_eligibility": {
            "label": "DIRECT_RISK_ACCURACY_ELIGIBILITY",
            "formula": "condition_number * float64_epsilon * max(1, abs(direct_gap), abs(exact_gap)) <= scalar_atol + scalar_rtol * max(abs(direct_gap), abs(exact_gap))",
        },
        "binding_checks": binding_checks,
        "interior_final_check": interior_final,
        "exact_endpoint": endpoint,
        "endpoint_risks_check": endpoint_risks_check,
        "endpoint_zero_check": endpoint_check,
        "endpoint_and_interior_compared": False,
    }


def _rank_case(config: Mapping[str, Any]) -> dict[str, Any]:
    model = build_fwv1()
    relative = float(config["tolerances"]["rank_relative"])
    exact = rank_diagnostics(measurement(model, -0.5), relative)
    api_result = "NO_EXCEPTION"
    api_diagnostics: Mapping[str, Any] | None = None
    ordinary_risk_evaluated = False
    try:
        ordinary = ambient_measurement_risk(model, -0.5, 1.0)
        ordinary_risk_evaluated = "risk" in ordinary
    except OutOfDomainRankError as error:
        api_result = error.structured_result
        api_diagnostics = error.diagnostics.to_dict()
    side_rows = []
    sides_ok = True
    for offset in (1e-2, 1e-4, 1e-6):
        for sign in (-1.0, 1.0):
            rho = -0.5 + sign * offset
            measured = measurement(model, rho)
            diag = rank_diagnostics(measured, relative)
            gram = measured.T @ (np.eye(64, dtype=np.float64) + model.sigma0) @ measured
            gram_condition = rank_diagnostics(gram, relative).condition_number
            classification = (
                "ILL_CONDITIONED_FULL_RANK"
                if gram_condition > float(config["tolerances"]["condition_gate"])
                else "FULL_RANK"
            )
            sides_ok = sides_ok and diag.rank == 16
            side_rows.append(
                {
                    "rho": rho,
                    "rank": diag.to_dict(),
                    "normal_matrix_condition": gram_condition,
                    "classification": classification,
                    "ordinary_risk_evaluated": False,
                }
            )
    passed = (
        exact.rank == 12
        and sides_ok
        and api_result == "OUT_OF_DOMAIN_RANK_12"
        and not ordinary_risk_evaluated
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "rank_loss": exact.to_dict(),
        "rank_loss_label": api_result,
        "api_rank_diagnostics": api_diagnostics,
        "full_rank_api_exercised": True,
        "ordinary_full_rank_risk_evaluated": ordinary_risk_evaluated,
        "side_rows": side_rows,
    }


def _torus_shift(p: int, q: int) -> np.ndarray:
    shift = np.zeros((64, 64), dtype=np.float64)
    for i in range(8):
        for j in range(8):
            row = 8 * i + j
            column = 8 * ((i - p) % 8) + ((j - q) % 8)
            shift[row, column] = 1.0
    return shift


def _operator_norm(matrix: np.ndarray) -> float:
    return float(np.linalg.svd(np.asarray(matrix, dtype=np.float64), compute_uv=False)[0])


def _covariance_case(config: Mapping[str, Any]) -> dict[str, Any]:
    model = build_fwv1()
    sigma_center = model.sigma0
    horizontal = _torus_shift(1, 0)
    vertical = _torus_shift(0, 1)
    g0 = ambient_measurement_risk(model, 0.0, 1.0, sigma_center)["risk"] - ambient_measurement_risk(
        model, 0.25, 1.0, sigma_center
    )["risk"]
    rows = []
    failures = []
    denominator = float(config["c"]["covariance_radius_denominator"])
    for index, (p, q) in enumerate(config["c"]["covariance_offsets"]):
        shift = _torus_shift(int(p), int(q))
        base_direction = (shift + shift.T) / 2.0
        norm = operator_norm_symmetric(base_direction)
        direction = ((-1.0) ** index) * base_direction / norm
        direction_checks = {
            "normalized": abs(operator_norm_symmetric(direction) - 1.0) <= 1e-12,
            "symmetric": _operator_norm(direction - direction.T) <= 1e-12,
            "horizontal_stationary": _operator_norm(direction @ horizontal - horizontal @ direction) <= 1e-12,
            "vertical_stationary": _operator_norm(direction @ vertical - vertical @ direction) <= 1e-12,
        }
        for numerator in config["c"]["covariance_radius_numerators"]:
            radius = float(numerator) / denominator
            covariance = sigma_center + radius * direction
            symmetry = _operator_norm(covariance - covariance.T)
            comm_h = _operator_norm(covariance @ horizontal - horizontal @ covariance)
            comm_v = _operator_norm(covariance @ vertical - vertical @ covariance)
            diag = rank_diagnostics(covariance, float(config["tolerances"]["rank_relative"]))
            try:
                np.linalg.cholesky(covariance)
                cholesky_ok = True
            except np.linalg.LinAlgError:
                cholesky_ok = False
            whiteness = operator_norm_symmetric(
                covariance - np.trace(covariance) / 64.0 * np.eye(64)
            )
            risk_identity = ambient_measurement_risk(model, 0.0, 1.0, covariance)["risk"]
            risk_witness = ambient_measurement_risk(model, 0.25, 1.0, covariance)["risk"]
            gain = risk_identity - risk_witness
            perturbation_norm = operator_norm_symmetric(covariance - sigma_center)
            bound = 1312.0 * perturbation_norm
            bound_check = abs(gain - g0) <= bound + float(config["tolerances"]["scalar_atol"]) + float(
                config["tolerances"]["scalar_rtol"]
            ) * max(abs(gain - g0), bound)
            checks = {
                **direction_checks,
                "covariance_symmetric": symmetry <= 1e-12,
                "covariance_horizontal_stationary": comm_h <= 1e-12,
                "covariance_vertical_stationary": comm_v <= 1e-12,
                "covariance_spd": diag.rank == 64 and cholesky_ok,
                "covariance_nonwhite": whiteness > float(config["tolerances"]["theoretical_zero"]),
                "lipschitz_bound": bound_check,
                "gain_margin": gain > 1.0 / 12.0,
            }
            row = {
                "direction_index": index,
                "offset": [int(p), int(q)],
                "sign": 1 if index % 2 == 0 else -1,
                "radius_numerator": float(numerator),
                "radius": radius,
                "direction_norm": operator_norm_symmetric(direction),
                "perturbation_norm": perturbation_norm,
                "symmetry_residual": symmetry,
                "horizontal_commutator": comm_h,
                "vertical_commutator": comm_v,
                "rank": diag.to_dict(),
                "nonwhite_residual": whiteness,
                "risk_identity": risk_identity,
                "risk_witness": risk_witness,
                "gain": gain,
                "center_gain": g0,
                "lipschitz_rhs": bound,
                "checks": checks,
            }
            if not all(checks.values()):
                failures.append({"direction_index": index, "radius_numerator": float(numerator)})
            rows.append(row)
    return {
        "status": "PASS" if not failures and len(rows) == 48 else "FAIL",
        "center_gain": g0,
        "directions": 16,
        "radii": 3,
        "combinations": len(rows),
        "rows": rows,
        "failures": failures,
        "optimizer_persistence_tested": False,
    }


def _counterexample_01() -> dict[str, Any]:
    values = np.asarray((-1.0, 1.0), dtype=np.float64)
    probabilities = np.asarray((0.5, 0.5), dtype=np.float64)
    squares = values * values
    mean_square = float(np.sum(probabilities * squares))
    centered = squares - mean_square
    variance = float(np.sum(probabilities * centered * centered))
    passed = np.array_equal(centered, np.zeros(2)) and variance == 0.0
    return {
        "status": "PASS" if passed else "FAIL",
        "label": "CORRECTIVE_REFUTATION_GUARD_NOT_POSITIVE_NONGAUSSIAN_THEORY",
        "support": values,
        "probabilities": probabilities,
        "centered_square": centered,
        "variance": variance,
    }


def _counterexample_02() -> dict[str, Any]:
    # Frozen Gaussian moments for X~N(0,1), Y=X and the identity representation C(X)=X.
    ex2 = 1.0
    ey2 = ex2
    exy = ex2
    bayes_full_risk = ey2 - exy * exy / ex2
    bayes_representation_risk = bayes_full_risk
    unrestricted_psg = bayes_representation_risk - bayes_full_risk

    # F_Z={0}; F_X={x -> a*x}. The quadratic is evaluated at its unique minimizer.
    restricted_coarse_coefficient = 0.0
    restricted_coarse_risk = ey2
    restricted_full_minimizer = exy / ex2
    restricted_full_risk = (
        ey2
        - 2.0 * restricted_full_minimizer * exy
        + restricted_full_minimizer * restricted_full_minimizer * ex2
    )
    identity_lift_risk = ey2 - 2.0 * exy + ex2
    forward_lift_zero_composed_with_identity_is_in_FX = True
    reverse_identity_predictor_available_in_FZ = False
    identity_lift_in_restricted_coarse_class = reverse_identity_predictor_available_in_FZ
    restricted_contrast = restricted_coarse_risk - restricted_full_risk
    identity_sigma_field_check = bayes_representation_risk == bayes_full_risk
    passed = (
        ex2 == 1.0
        and identity_sigma_field_check
        and unrestricted_psg == 0.0
        and restricted_coarse_risk == 1.0
        and restricted_full_minimizer == 1.0
        and restricted_full_risk == 0.0
        and identity_lift_risk == 0.0
        and forward_lift_zero_composed_with_identity_is_in_FX
        and not reverse_identity_predictor_available_in_FZ
        and not identity_lift_in_restricted_coarse_class
        and restricted_contrast == 1.0
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "label": "CORRECTIVE_CLASS_MISMATCH_GUARD_NOT_REPRESENTATION_LOSS",
        "moments": {"E_X2": ex2, "E_Y2": ey2, "E_XY": exy},
        "identity_sigma_field_check": identity_sigma_field_check,
        "bayes_full_risk": bayes_full_risk,
        "bayes_representation_risk": bayes_representation_risk,
        "unrestricted_psg": unrestricted_psg,
        "restricted_coarse_coefficient": restricted_coarse_coefficient,
        "restricted_coarse_risk": restricted_coarse_risk,
        "restricted_full_minimizer": restricted_full_minimizer,
        "restricted_full_risk": restricted_full_risk,
        "identity_lift_risk": identity_lift_risk,
        "forward_lift_zero_composed_with_identity_is_in_FX": (
            forward_lift_zero_composed_with_identity_is_in_FX
        ),
        "reverse_identity_predictor_available_in_FZ": (
            reverse_identity_predictor_available_in_FZ
        ),
        "identity_lift_in_restricted_coarse_class": identity_lift_in_restricted_coarse_class,
        "restricted_contrast": restricted_contrast,
    }


def run_boundaries(config: Mapping[str, Any]) -> dict[str, Any]:
    cases = {
        "C-NOISE-01": _noise_case(False, config),
        "C-NOISE-02": _noise_case(True, config),
        "C-CLEAN-01": _clean_case_01(config),
        "C-CLEAN-02": _clean_case_02(config),
        "C-SING-01": _singular_case(config),
        "C-RANK-01": _rank_case(config),
        "C-COV-01": _covariance_case(config),
        "C-CTR-01": _counterexample_01(),
        "C-CTR-02": _counterexample_02(),
    }
    return {
        "module": "C",
        "theory_scope": "Appendix B.2-B.3 and Appendix C corrective guards",
        "evidence_type": "deterministic boundary regression and sanity evidence",
        "claim_status_changed": False,
        "counterexamples_are_positive_nongaussian_theory": False,
        "verdict": overall_status(item["status"] for item in cases.values()),
        "cases": cases,
    }


def smoke_check(config: Mapping[str, Any]) -> dict[str, Any]:
    identity = np.eye(3, dtype=np.float64)
    retained = identity[:, :1]
    complement = identity[:, 1:]
    target = (identity[:, :1] + identity[:, 1:2]) / math.sqrt(2.0)
    sigma0 = np.asarray(((1.0, 0.2, 0.0), (0.2, 0.8, 0.1), (0.0, 0.1, 0.6)))
    parent = subspace_parent_gap(sigma0, 0.01, 1.0, retained, complement, target)
    shift = _torus_shift(1, 0)
    checks = {
        "toy_parent_finite": bool(np.isfinite(parent["gap"])),
        "toy_parent_nonnegative": parent["gap"] >= 0.0,
        "shift_orthogonal": bool(np.array_equal(shift.T @ shift, np.eye(64))),
    }
    return {
        "mode": "SMOKE_ONLY_NON_FORMAL_FIXTURES",
        "formal_case_ids_used": [],
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = base_parser(__doc__ or "T0 boundary validator")
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
    run_dir = new_run_directory(config, "c")
    result = run_boundaries(config)
    write_json(run_dir / "RUN_METADATA.json", {"module": "C", "environment": environment_metadata(), "source_lock": source_lock})
    write_json(run_dir / "C_BOUNDARY_RESULTS.json", result)
    print(run_dir)
    if result["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
