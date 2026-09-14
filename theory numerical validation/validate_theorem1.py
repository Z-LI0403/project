"""T0-A: numerical regression for Theorem 1 and its stated appendices."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from common_linear_gaussian import (
    NumericalFailure,
    PROJECT_ROOT,
    SourceDriftError,
    base_parser,
    deterministic_orthogonal,
    direct_gaussian_conditioning,
    environment_metadata,
    gaussian_monte_carlo,
    load_config,
    matrix_close,
    new_run_directory,
    operator_norm_symmetric,
    overall_status,
    rank_diagnostics,
    scalar_close,
    subspace_parent_gap,
    theorem1_formula,
    verify_source_lock,
    whitened_residual_gap,
    write_json,
)


def _spectrum(spec: Mapping[str, Any]) -> np.ndarray:
    if "lambda" in spec:
        return np.asarray(spec["lambda"], dtype=np.float64)
    if "lambda_logspace" in spec:
        start, stop, count = spec["lambda_logspace"]
        return np.logspace(start, stop, int(count), dtype=np.float64)
    if "lambda_prefix_logspace" in spec:
        start, stop, count = spec["lambda_prefix_logspace"]
        return np.concatenate(
            (
                np.logspace(start, stop, int(count), dtype=np.float64),
                np.zeros(int(spec["lambda_trailing_zeros"]), dtype=np.float64),
            )
        )
    raise NumericalFailure(f"missing frozen spectrum for {spec['id']}")


def _measurement_spectrum(spec: Mapping[str, Any]) -> np.ndarray:
    if "measurement_singular_values" in spec:
        return np.asarray(spec["measurement_singular_values"], dtype=np.float64)
    start, stop, count = spec["measurement_logspace"]
    return np.logspace(start, stop, int(count), dtype=np.float64)


def build_generic_case(spec: Mapping[str, Any], root_seed: int) -> dict[str, Any]:
    case_id = str(spec["id"])
    d, m, q = int(spec["d"]), int(spec["m"]), int(spec["q"])
    q_sigma, seed_sigma = deterministic_orthogonal(root_seed, case_id, "sigma-basis", d)
    q_target, seed_target = deterministic_orthogonal(root_seed, case_id, "target-basis", d)
    q_left, seed_left = deterministic_orthogonal(root_seed, case_id, "measurement-left", d)
    q_right, seed_right = deterministic_orthogonal(root_seed, case_id, "measurement-right", q)
    values = _spectrum(spec)
    singular = _measurement_spectrum(spec)
    if values.shape != (d,) or singular.shape != (q,):
        raise NumericalFailure(f"frozen spectrum shape mismatch for {case_id}")
    sigma0 = q_sigma @ np.diag(values) @ q_sigma.T
    target = q_target[:, :m]
    measurement = q_left[:, :q] @ np.diag(singular) @ q_right.T
    return {
        "id": case_id,
        "d": d,
        "m": m,
        "q": q,
        "sigma0": sigma0,
        "target": target,
        "measurement": measurement,
        "a": float(spec["a"]),
        "b": float(spec["b"]),
        "seeds": {
            "sigma-basis": seed_sigma,
            "target-basis": seed_target,
            "measurement-left": seed_left,
            "measurement-right": seed_right,
        },
        "clean_eigenvalues": values,
        "measurement_singular_values": singular,
        "mc": bool(spec["mc"]),
    }


def _direct_risk_invariants(
    direct: Mapping[str, Any], m: int, tolerances: Mapping[str, float]
) -> dict[str, bool]:
    atol = float(tolerances["scalar_atol"])
    floor = float(tolerances["nonnegative_floor"])
    values = (
        float(direct["risk_full"]),
        float(direct["risk_measurement"]),
        float(direct["gap"]),
    )
    return {
        "finite": bool(np.all(np.isfinite(values))),
        "risk_full_in_0_m": -atol <= values[0] <= m + atol,
        "risk_measurement_in_0_m": -atol <= values[1] <= m + atol,
        "gap_nonnegative_to_floor": values[2] >= floor,
    }


def _three_routes(case: Mapping[str, Any], tolerances: Mapping[str, float]) -> dict[str, Any]:
    formula = theorem1_formula(
        case["sigma0"],
        case["a"],
        case["b"],
        case["target"],
        case["measurement"],
        case_id=str(case["id"]),
    )
    direct = direct_gaussian_conditioning(
        case["sigma0"],
        case["a"],
        case["b"],
        case["target"],
        case["measurement"],
        case_id=str(case["id"]),
    )
    whitened = whitened_residual_gap(
        case["sigma0"],
        case["a"],
        case["b"],
        case["target"],
        case["measurement"],
        rank_relative=float(tolerances["rank_relative"]),
        case_id=str(case["id"]),
    )
    atol, rtol = float(tolerances["scalar_atol"]), float(tolerances["scalar_rtol"])
    pairwise = {
        "formula_vs_direct": scalar_close(formula["gap"], direct["gap"], atol, rtol),
        "formula_vs_whitened": scalar_close(formula["gap"], whitened["gap"], atol, rtol),
        "direct_vs_whitened": scalar_close(direct["gap"], whitened["gap"], atol, rtol),
    }
    risk_invariants = _direct_risk_invariants(direct, int(case["m"]), tolerances)
    risk_bounds = all(risk_invariants.values())
    return {
        "formula": formula,
        "direct": direct,
        "whitened": whitened,
        "route_checks": pairwise,
        "risk_invariants": risk_invariants,
        "risk_bounds": risk_bounds,
        "passed": all(pairwise.values()) and risk_bounds and whitened["gap"] >= 0.0,
    }


def _evaluate_a_inv_02(
    base: Mapping[str, Any],
    coordinate: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the human-approved A-INV-02 canonical-coordinate amendment."""
    case_id = "A-INV-02"
    tolerances = config["tolerances"]
    rank_relative = float(tolerances["rank_relative"])
    condition_gate = float(tolerances["condition_gate"])
    normalized_tolerance = float(tolerances["normalized_residual"])
    atol = float(tolerances["scalar_atol"])
    rtol = float(tolerances["scalar_rtol"])
    d, m, q = int(base["d"]), int(base["m"]), int(base["q"])

    raw_measurement = np.asarray(base["measurement"] @ coordinate, dtype=np.float64)
    sigma_t = (
        np.float64(base["a"]) ** 2 * np.asarray(base["sigma0"], dtype=np.float64)
        + np.float64(base["b"]) ** 2 * np.eye(d, dtype=np.float64)
    )
    raw_gram = raw_measurement.T @ sigma_t @ raw_measurement
    raw_measurement_rank = rank_diagnostics(raw_measurement, rank_relative)
    raw_gram_rank = rank_diagnostics(raw_gram, rank_relative)
    raw_rank_gate_passed = raw_gram_rank.full_column_rank
    raw_condition_gate_passed = raw_gram_rank.condition_number <= condition_gate
    raw_classification = (
        "EXPECTED_RAW_COORDINATE_ILL_CONDITIONING"
        if raw_measurement_rank.full_column_rank
        and not (raw_rank_gate_passed and raw_condition_gate_passed)
        else "UNEXPECTED_RAW_COORDINATE_CLASSIFICATION"
    )

    canonical, triangular = np.linalg.qr(raw_measurement, mode="reduced")
    canonical = np.asarray(canonical, dtype=np.float64)
    triangular = np.asarray(triangular, dtype=np.float64)
    canonical_rank = rank_diagnostics(canonical, rank_relative)
    triangular_rank = rank_diagnostics(triangular, rank_relative)
    reconstruction_residual = float(
        np.linalg.norm(raw_measurement - canonical @ triangular, ord="fro")
        / max(1.0, float(np.linalg.norm(raw_measurement, ord="fro")))
    )
    canonical_projector = canonical @ canonical.T
    raw_column_residual = float(
        np.linalg.norm(
            raw_measurement - canonical_projector @ raw_measurement, ord="fro"
        )
        / max(1.0, float(np.linalg.norm(raw_measurement, ord="fro")))
    )
    base_coordinates, _ = np.linalg.qr(
        np.asarray(base["measurement"], dtype=np.float64), mode="reduced"
    )
    base_projector = base_coordinates @ base_coordinates.T
    projector_residual = float(
        np.linalg.norm(base_projector - canonical_projector, ord="fro")
        / max(1.0, float(np.linalg.norm(base_projector, ord="fro")))
    )
    base_to_canonical_residual = float(
        np.linalg.norm(
            canonical - base_projector @ canonical, ord="fro"
        )
        / max(1.0, float(np.linalg.norm(canonical, ord="fro")))
    )
    canonical_to_base_residual = float(
        np.linalg.norm(
            base_coordinates - canonical_projector @ base_coordinates, ord="fro"
        )
        / max(1.0, float(np.linalg.norm(base_coordinates, ord="fro")))
    )

    reference_routes = _three_routes(base, tolerances)
    canonical_formula = theorem1_formula(
        base["sigma0"],
        base["a"],
        base["b"],
        base["target"],
        canonical,
        case_id=case_id,
    )
    canonical_direct = direct_gaussian_conditioning(
        base["sigma0"],
        base["a"],
        base["b"],
        base["target"],
        canonical,
        case_id=case_id,
    )
    raw_whitened = whitened_residual_gap(
        base["sigma0"],
        base["a"],
        base["b"],
        base["target"],
        raw_measurement,
        rank_relative=rank_relative,
        case_id=case_id,
    )
    transformed_pairwise = {
        "formula_vs_direct": scalar_close(
            canonical_formula["gap"], canonical_direct["gap"], atol, rtol
        ),
        "formula_vs_whitened": scalar_close(
            canonical_formula["gap"], raw_whitened["gap"], atol, rtol
        ),
        "direct_vs_whitened": scalar_close(
            canonical_direct["gap"], raw_whitened["gap"], atol, rtol
        ),
    }
    base_to_transformed_gaps = {
        "formula": scalar_close(
            reference_routes["formula"]["gap"], canonical_formula["gap"], atol, rtol
        ),
        "direct": scalar_close(
            reference_routes["direct"]["gap"], canonical_direct["gap"], atol, rtol
        ),
        "whitened": scalar_close(
            reference_routes["whitened"]["gap"], raw_whitened["gap"], atol, rtol
        ),
    }
    direct_risk_comparisons = {
        key: scalar_close(
            reference_routes["direct"][key], canonical_direct[key], atol, rtol
        )
        for key in ("risk_full", "risk_measurement", "gap")
    }
    transformed_risk_invariants = _direct_risk_invariants(
        canonical_direct, m, tolerances
    )
    coordinate_condition = rank_diagnostics(coordinate, rank_relative).condition_number
    canonical_gram = canonical.T @ sigma_t @ canonical
    canonical_gram_rank = rank_diagnostics(canonical_gram, rank_relative)

    binding_checks = {
        "reference_routes_passed": bool(reference_routes["passed"]),
        "raw_measurement_full_column_rank": raw_measurement_rank.full_column_rank,
        "raw_expected_classification": raw_classification
        == "EXPECTED_RAW_COORDINATE_ILL_CONDITIONING",
        "coordinate_condition_within_frozen_bound": coordinate_condition
        <= 1e4 * (1.0 + rank_relative),
        "canonical_Q_full_column_rank": canonical_rank.full_column_rank
        and canonical.shape == (d, q),
        "triangular_T_full_rank": triangular_rank.full_column_rank
        and triangular.shape == (q, q),
        "measurement_dimension_preserved": raw_measurement.shape == (d, q)
        and canonical.shape[1] == raw_measurement.shape[1],
        "qr_reconstruction": reconstruction_residual <= normalized_tolerance,
        "raw_column_space_residual": raw_column_residual <= normalized_tolerance,
        "base_transformed_projector": projector_residual <= normalized_tolerance,
        "base_to_canonical_column_residual": base_to_canonical_residual
        <= normalized_tolerance,
        "canonical_to_base_column_residual": canonical_to_base_residual
        <= normalized_tolerance,
        "canonical_gram_full_rank": canonical_gram_rank.full_column_rank,
        "canonical_gram_condition_gate": canonical_gram_rank.condition_number
        <= condition_gate,
        "transformed_three_route_pairwise": all(transformed_pairwise.values()),
        "base_to_transformed_gaps": all(base_to_transformed_gaps.values()),
        "direct_risk_invariance": all(direct_risk_comparisons.values()),
        "transformed_direct_risk_invariants": all(
            transformed_risk_invariants.values()
        ),
        "raw_whitened_finite_nonnegative": bool(
            np.isfinite(raw_whitened["gap"]) and raw_whitened["gap"] >= 0.0
        ),
    }
    passed = all(binding_checks.values())
    return {
        "status": "PASS" if passed else "FAIL",
        "dimensions": {"d": d, "m": m, "q": q},
        "protocol_route": "A-INV-02-ONLY-CANONICAL-QR-v1.1",
        "post_result_amendment": True,
        "amendment_path": (
            "experiments/theory_validation/t0/T0_PROTOCOL_AMENDMENT_A_INV_02.md"
        ),
        "raw_normal_system_cholesky_attempted": False,
        "raw_coordinate_diagnostic": {
            "label": raw_classification,
            "theorem_failure": False,
            "expected_mathematical_rank": q,
            "mathematical_full_column_rank_by_invertible_construction": True,
            "raw_measurement_rank": raw_measurement_rank.to_dict(),
            "raw_normal_matrix_rank": raw_gram_rank.to_dict(),
            "raw_normal_matrix_rank_gate_passed": raw_rank_gate_passed,
            "raw_normal_matrix_condition_gate_passed": raw_condition_gate_passed,
            "raw_normal_matrix_antisymmetric_frobenius": float(
                np.linalg.norm(raw_gram - raw_gram.T, ord="fro")
            ),
            "raw_normal_matrix_eigenvalues": np.linalg.eigvalsh(
                (raw_gram + raw_gram.T) / 2.0
            ),
        },
        "canonical_qr": {
            "Q_rank": canonical_rank.to_dict(),
            "T_rank": triangular_rank.to_dict(),
            "normalized_reconstruction_residual": reconstruction_residual,
            "raw_column_space_residual": raw_column_residual,
            "base_transformed_projector_residual": projector_residual,
            "base_to_canonical_column_residual": base_to_canonical_residual,
            "canonical_to_base_column_residual": canonical_to_base_residual,
            "canonical_gram_rank": canonical_gram_rank.to_dict(),
            "canonical_gram_antisymmetric_frobenius": float(
                np.linalg.norm(canonical_gram - canonical_gram.T, ord="fro")
            ),
            "canonical_gram_eigenvalues": np.linalg.eigvalsh(
                (canonical_gram + canonical_gram.T) / 2.0
            ),
        },
        "coordinate_condition": coordinate_condition,
        "reference_routes": reference_routes,
        "canonical_formula": canonical_formula,
        "canonical_direct": canonical_direct,
        "raw_whitened": raw_whitened,
        "transformed_pairwise_checks": transformed_pairwise,
        "base_to_transformed_gap_checks": base_to_transformed_gaps,
        "direct_risk_comparisons": direct_risk_comparisons,
        "transformed_risk_invariants": transformed_risk_invariants,
        "binding_checks": binding_checks,
    }
def _mc_for_case(
    case: Mapping[str, Any], config: Mapping[str, Any], *, smoke: bool = False
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if smoke:
        kwargs.update(sample_checkpoints=(1024,), batch_size=256)
    return gaussian_monte_carlo(
        case["sigma0"],
        case["a"],
        case["b"],
        case["target"],
        case["measurement"],
        root_seed=int(config["seeds"]["a_monte_carlo"]),
        case_id=str(case["id"]),
        mc_config=config["monte_carlo"],
        **kwargs,
    )


def _pair_basis_8() -> tuple[np.ndarray, np.ndarray]:
    retained = np.zeros((8, 4), dtype=np.float64)
    complement = np.zeros((8, 4), dtype=np.float64)
    scale = 1.0 / math.sqrt(2.0)
    for j in range(4):
        retained[2 * j : 2 * j + 2, j] = (scale, scale)
        complement[2 * j : 2 * j + 2, j] = (scale, -scale)
    return retained, complement


def _haar_covariance(
    retained: np.ndarray,
    complement: np.ndarray,
    retained_block: np.ndarray,
    discarded_block: np.ndarray,
    cross: np.ndarray,
) -> np.ndarray:
    basis = np.column_stack((retained, complement))
    block = np.block([[retained_block, cross], [cross.T, discarded_block]])
    return basis @ block @ basis.T


def _helmert_case() -> dict[str, Any]:
    columns_u: list[np.ndarray] = []
    columns_perp: list[np.ndarray] = []
    local = (
        np.asarray((1, 1, 1, 1), dtype=np.float64) / 2.0,
        np.asarray((1, -1, 0, 0), dtype=np.float64) / math.sqrt(2.0),
        np.asarray((1, 1, -2, 0), dtype=np.float64) / math.sqrt(6.0),
        np.asarray((1, 1, 1, -3), dtype=np.float64) / math.sqrt(12.0),
    )
    for block in range(4):
        embedded = []
        for vector in local:
            value = np.zeros(16, dtype=np.float64)
            value[4 * block : 4 * block + 4] = vector
            embedded.append(value)
        columns_u.append(embedded[0])
        columns_perp.extend(embedded[1:])
    retained = np.column_stack(columns_u)
    complement = np.column_stack(columns_perp)
    basis = np.column_stack((retained, complement))
    sigma0 = basis @ np.diag(np.linspace(0.5, 2.0, 16, dtype=np.float64)) @ basis.T
    return {
        "id": "A-HMR-02",
        "d": 16,
        "m": 4,
        "q": 4,
        "sigma0": sigma0,
        "target": retained,
        "measurement": retained,
        "retained": retained,
        "complement": complement,
        "a": 0.8,
        "b": 0.6,
        "mc": False,
    }


def _evaluate_special_case(
    case: Mapping[str, Any], config: Mapping[str, Any], *, include_parent: bool, include_mc: bool
) -> dict[str, Any]:
    tolerances = config["tolerances"]
    formula = theorem1_formula(
        case["sigma0"],
        case["a"],
        case["b"],
        case["target"],
        case["measurement"],
        case_id=str(case["id"]),
    )
    direct = direct_gaussian_conditioning(
        case["sigma0"],
        case["a"],
        case["b"],
        case["target"],
        case["measurement"],
        case_id=str(case["id"]),
    )
    measurement_rank = rank_diagnostics(
        case["measurement"], float(tolerances["rank_relative"])
    )
    risk_invariants = _direct_risk_invariants(direct, int(case["m"]), tolerances)
    checks = {
        "formula_vs_direct": scalar_close(
            formula["gap"],
            direct["gap"],
            float(tolerances["scalar_atol"]),
            float(tolerances["scalar_rtol"]),
        ),
        "measurement_full_column_rank": measurement_rank.full_column_rank,
        **risk_invariants,
    }
    result: dict[str, Any] = {
        "dimensions": {
            "d": int(case["d"]),
            "m": int(case["m"]),
            "q": int(case["q"]),
        },
        "formula": formula,
        "direct": direct,
        "measurement_rank": measurement_rank.to_dict(),
        "risk_invariants": risk_invariants,
        "checks": checks,
    }
    if include_parent:
        parent = subspace_parent_gap(
            case["sigma0"],
            case["a"],
            case["b"],
            case["retained"],
            case["complement"],
            case["target"],
            case_id=str(case["id"]),
        )
        checks["formula_vs_parent"] = scalar_close(
            formula["gap"],
            parent["gap"],
            float(tolerances["scalar_atol"]),
            float(tolerances["scalar_rtol"]),
        )
        result["parent"] = parent
    if include_mc:
        result["mc"] = _mc_for_case(case, config)
    deterministic_ok = all(checks.values())
    if not deterministic_ok:
        result["status"] = "FAIL"
    elif include_mc and result["mc"]["status"] == "FAIL":
        result["status"] = "FAIL"
    elif include_mc and result["mc"]["status"] == "INCONCLUSIVE_PRECISION":
        result["status"] = "INCONCLUSIVE_PRECISION"
    else:
        result["status"] = "PASS"
    return result


def run_theorem1(config: Mapping[str, Any]) -> dict[str, Any]:
    tolerances = config["tolerances"]
    root_seed = int(config["seeds"]["a_synthetic"])
    cases: dict[str, Any] = {}
    built_generic: dict[str, dict[str, Any]] = {}
    mc_summaries: dict[str, Any] = {}

    for spec in config["a"]["generic_cases"]:
        case = build_generic_case(spec, root_seed)
        built_generic[case["id"]] = case
        routes = _three_routes(case, tolerances)
        measurement_rank = rank_diagnostics(
            case["measurement"], float(tolerances["rank_relative"])
        )
        clean_values = np.linalg.eigvalsh((case["sigma0"] + case["sigma0"].T) / 2.0)
        construction_ok = (
            measurement_rank.full_column_rank
            and float(clean_values[0])
            >= -float(tolerances["rank_relative"]) * max(1.0, float(clean_values[-1]))
        )
        mc = _mc_for_case(case, config)
        mc_summaries[case["id"]] = mc
        zero_ok = True
        if case["id"] == "A-GEN-04":
            zero_ok = abs(float(routes["formula"]["gap"])) <= float(
                tolerances["theoretical_zero"]
            )
        status = "PASS"
        if not construction_ok or not routes["passed"] or not zero_ok or mc["status"] == "FAIL":
            status = "FAIL"
        elif mc["status"] == "INCONCLUSIVE_PRECISION":
            status = "INCONCLUSIVE_PRECISION"
        cases[case["id"]] = {
            "status": status,
            "dimensions": {"d": case["d"], "m": case["m"], "q": case["q"]},
            "a": case["a"],
            "b": case["b"],
            "derived_seeds": case["seeds"],
            "clean_eigenvalues": clean_values,
            "measurement_rank": measurement_rank.to_dict(),
            "routes": routes,
            "zero_check": zero_ok,
            "mc": mc,
        }

    base = built_generic["A-GEN-02"]
    for case_id, coordinate in (
        (
            "A-INV-01",
            deterministic_orthogonal(root_seed, "A-INV-01", "coordinate-right", 4)[0],
        ),
        (
            "A-INV-02",
            deterministic_orthogonal(root_seed, "A-INV-02", "coordinate-left", 4)[0]
            @ np.diag((1e-2, 1e-1, 1e1, 1e2))
            @ deterministic_orthogonal(root_seed, "A-INV-02", "coordinate-right", 4)[0].T,
        ),
    ):
        if case_id == "A-INV-02":
            cases[case_id] = _evaluate_a_inv_02(base, coordinate, config)
            continue
        transformed = dict(base)
        transformed["id"] = case_id
        transformed["measurement"] = base["measurement"] @ coordinate
        reference_routes = _three_routes(base, tolerances)
        transformed_routes = _three_routes(transformed, tolerances)
        comparisons = {
            name: scalar_close(
                reference_routes[name]["gap"],
                transformed_routes[name]["gap"],
                float(tolerances["scalar_atol"]),
                float(tolerances["scalar_rtol"]),
            )
            for name in ("formula", "direct", "whitened")
        }
        direct_risk_comparisons = {
            key: scalar_close(
                reference_routes["direct"][key],
                transformed_routes["direct"][key],
                float(tolerances["scalar_atol"]),
                float(tolerances["scalar_rtol"]),
            )
            for key in ("risk_full", "risk_measurement", "gap")
        }
        condition = rank_diagnostics(coordinate).condition_number
        condition_ok = condition <= 1e4 * (1.0 + float(tolerances["rank_relative"]))
        binding_checks = {
            "reference_routes_passed": bool(reference_routes["passed"]),
            "transformed_routes_passed": bool(transformed_routes["passed"]),
            "same_route_gap_invariance": all(comparisons.values()),
            "direct_risk_invariance": all(direct_risk_comparisons.values()),
            "coordinate_condition": condition_ok,
        }
        cases[case_id] = {
            "status": "PASS" if all(binding_checks.values()) else "FAIL",
            "dimensions": {"d": base["d"], "m": base["m"], "q": base["q"]},
            "coordinate_condition": condition,
            "condition_check": condition_ok,
            "comparisons": comparisons,
            "direct_risk_comparisons": direct_risk_comparisons,
            "binding_checks": binding_checks,
            "reference_gaps": {name: reference_routes[name]["gap"] for name in comparisons},
            "transformed_gaps": {name: transformed_routes[name]["gap"] for name in comparisons},
            "reference_routes": reference_routes,
            "transformed_routes": transformed_routes,
        }

    retained8, complement8 = _pair_basis_8()
    zero_case = {
        "id": "A-HMR-00",
        "d": 8,
        "m": 4,
        "q": 4,
        "sigma0": _haar_covariance(
            retained8,
            complement8,
            np.diag((0.0, 0.5, 1.0, 2.0)),
            np.diag((0.25, 0.75, 1.5, 3.0)),
            np.zeros((4, 4), dtype=np.float64),
        ),
        "target": retained8,
        "measurement": retained8,
        "retained": retained8,
        "complement": complement8,
        "a": 1.0 / math.sqrt(2.0),
        "b": 1.0 / math.sqrt(2.0),
        "mc": True,
    }
    cross = 0.25 * np.diag((1.0, -1.0, 1.0, -1.0))
    positive_case = dict(zero_case)
    positive_case.update(
        id="A-HMR-01",
        sigma0=_haar_covariance(
            retained8,
            complement8,
            2.0 * np.eye(4),
            1.5 * np.eye(4),
            cross,
        ),
    )
    for case, expected_zero in ((zero_case, True), (positive_case, False)):
        result = _evaluate_special_case(case, config, include_parent=True, include_mc=True)
        clean_cross_norm = operator_norm_symmetric(
            np.block(
                [
                    [np.zeros((4, 4)), result["parent"]["clean_cross"]],
                    [result["parent"]["clean_cross"].T, np.zeros((4, 4))],
                ]
            )
        )
        if expected_zero:
            criterion = (
                clean_cross_norm <= float(tolerances["theoretical_zero"])
                and abs(result["parent"]["gap"]) <= float(tolerances["theoretical_zero"])
            )
        else:
            criterion = (
                clean_cross_norm > float(tolerances["theoretical_zero"])
                and result["parent"]["gap"] > float(tolerances["theoretical_zero"])
            )
        if not criterion:
            result["status"] = "FAIL"
        result["haar_criterion"] = {
            "clean_cross_operator_norm": clean_cross_norm,
            "expected_zero": expected_zero,
            "passed": criterion,
        }
        cases[case["id"]] = result
        mc_summaries[case["id"]] = result["mc"]

    helmert = _helmert_case()
    cases["A-HMR-02"] = _evaluate_special_case(
        helmert, config, include_parent=False, include_mc=False
    )

    scaling_fields: dict[str, Any] = {}
    for field, case, divisor in (
        ("pair_average", positive_case, math.sqrt(2.0)),
        ("four_pixel_average", helmert, 2.0),
    ):
        normalized_measurement = case["measurement"]
        average_measurement = case["measurement"] / divisor
        normalized = direct_gaussian_conditioning(
            case["sigma0"],
            case["a"],
            case["b"],
            case["target"],
            normalized_measurement,
            case_id="A-HMR-03",
        )
        average = direct_gaussian_conditioning(
            case["sigma0"],
            case["a"],
            case["b"],
            case["target"],
            average_measurement,
            case_id="A-HMR-03",
        )
        checks = {
            key: scalar_close(
                normalized[key],
                average[key],
                float(tolerances["scalar_atol"]),
                float(tolerances["scalar_rtol"]),
            )
            for key in ("risk_full", "risk_measurement", "gap")
        }
        normalized_invariants = _direct_risk_invariants(
            normalized, int(case["m"]), tolerances
        )
        average_invariants = _direct_risk_invariants(
            average, int(case["m"]), tolerances
        )
        checks.update(
            {
                f"normalized_{key}": value
                for key, value in normalized_invariants.items()
            }
        )
        checks.update(
            {
                f"average_{key}": value
                for key, value in average_invariants.items()
            }
        )
        normalized_rank = rank_diagnostics(
            normalized_measurement, float(tolerances["rank_relative"])
        )
        average_rank = rank_diagnostics(
            average_measurement, float(tolerances["rank_relative"])
        )
        checks["normalized_measurement_full_column_rank"] = (
            normalized_rank.full_column_rank
        )
        checks["average_measurement_full_column_rank"] = average_rank.full_column_rank
        scaling_fields[field] = {
            "divisor": divisor,
            "normalized": normalized,
            "ordinary_average": average,
            "normalized_measurement_rank": normalized_rank.to_dict(),
            "average_measurement_rank": average_rank.to_dict(),
            "checks": checks,
        }
    cases["A-HMR-03"] = {
        "status": "PASS"
        if all(all(item["checks"].values()) for item in scaling_fields.values())
        else "FAIL",
        "dimensions": {"pair_average_d": 8, "four_pixel_average_d": 16},
        "mandatory_scaling_fields": scaling_fields,
    }

    e1 = np.asarray(((1.0,), (0.0,)), dtype=np.float64)
    e2 = np.asarray(((0.0,), (1.0,)), dtype=np.float64)
    distinct_cases = (
        {
            "id": "A-DST-01",
            "sigma0": np.eye(2, dtype=np.float64),
            "target": e2,
            "measurement": e1,
            "retained": e1,
            "complement": e2,
            "a": 1.0,
            "b": 1.0,
            "d": 2,
            "m": 1,
            "q": 1,
        },
        {
            "id": "A-DST-02",
            "sigma0": np.asarray(((1.0, 0.5), (0.5, 1.0)), dtype=np.float64),
            "target": np.asarray(((1.0,), (0.25,)), dtype=np.float64) / math.sqrt(1.0625),
            "measurement": e1,
            "retained": e1,
            "complement": e2,
            "a": 1.0,
            "b": 1.0,
            "d": 2,
            "m": 1,
            "q": 1,
        },
    )
    for case in distinct_cases:
        formula = theorem1_formula(
            case["sigma0"],
            case["a"],
            case["b"],
            case["target"],
            case["measurement"],
            case_id=str(case["id"]),
        )
        direct = direct_gaussian_conditioning(
            case["sigma0"],
            case["a"],
            case["b"],
            case["target"],
            case["measurement"],
            case_id=str(case["id"]),
        )
        risk_invariants = _direct_risk_invariants(
            direct, int(case["m"]), tolerances
        )
        measurement_rank = rank_diagnostics(
            case["measurement"], float(tolerances["rank_relative"])
        )
        parent = subspace_parent_gap(
            case["sigma0"],
            case["a"],
            case["b"],
            case["retained"],
            case["complement"],
            case["target"],
            case_id=str(case["id"]),
        )
        agreement = scalar_close(
            formula["gap"],
            parent["gap"],
            float(tolerances["scalar_atol"]),
            float(tolerances["scalar_rtol"]),
        )
        mc = _mc_for_case(case, config)
        mc_summaries[case["id"]] = mc
        if case["id"] == "A-DST-01":
            boundary = (
                np.linalg.norm(parent["clean_cross"], ord="fro")
                <= float(tolerances["matrix_atol"])
                and parent["gap"] > float(tolerances["theoretical_zero"])
            )
        else:
            boundary = (
                np.linalg.norm(parent["clean_cross"], ord="fro")
                > float(tolerances["matrix_atol"])
                and np.linalg.norm(parent["conditional_cross"], ord="fro")
                <= float(tolerances["matrix_atol"])
                and abs(parent["gap"]) <= float(tolerances["theoretical_zero"])
            )
        status = "PASS"
        if (
            not agreement
            or not boundary
            or not measurement_rank.full_column_rank
            or not all(risk_invariants.values())
            or mc["status"] == "FAIL"
        ):
            status = "FAIL"
        elif mc["status"] == "INCONCLUSIVE_PRECISION":
            status = "INCONCLUSIVE_PRECISION"
        cases[case["id"]] = {
            "status": status,
            "dimensions": {"d": case["d"], "m": case["m"], "q": case["q"]},
            "formula": formula,
            "direct": direct,
            "measurement_rank": measurement_rank.to_dict(),
            "risk_invariants": risk_invariants,
            "parent": parent,
            "agreement": agreement,
            "boundary_guard": boundary,
            "mc": mc,
        }

    verdict = overall_status(item["status"] for item in cases.values())
    return {
        "module": "A",
        "theory_scope": "Theorem 1(b), Corollary 1.1, Appendix A.1, Appendix B.1",
        "evidence_type": "deterministic numerical regression plus finite-sample Monte Carlo corroboration",
        "claim_status_changed": False,
        "verdict": verdict,
        "cases": cases,
        "mc_summaries": mc_summaries,
    }


def smoke_check(config: Mapping[str, Any]) -> dict[str, Any]:
    sigma0 = np.asarray(
        ((1.2, 0.15, 0.0), (0.15, 0.8, 0.1), (0.0, 0.1, 0.5)),
        dtype=np.float64,
    )
    target = np.asarray(((1.0,), (0.0,), (0.0,)), dtype=np.float64)
    measurement = np.asarray(((1.0,), (1.0,), (0.0,)), dtype=np.float64) / math.sqrt(2.0)
    case = {
        "id": "SMOKE-A-TOY-3D",
        "sigma0": sigma0,
        "target": target,
        "measurement": measurement,
        "a": 0.7,
        "b": 0.5,
        "m": 1,
    }
    routes = _three_routes(case, config["tolerances"])
    mc = _mc_for_case(case, config, smoke=True)
    final_summaries = mc["final"]["summaries"]
    mc_pipeline_healthy = (
        mc["status"] != "FAIL"
        and all(int(summary["count"]) == 1024 for summary in final_summaries.values())
        and all(
            np.isfinite(float(summary[field]))
            for summary in final_summaries.values()
            for field in ("mean", "unbiased_variance", "standard_error")
        )
    )
    return {
        "mode": "SMOKE_ONLY_NON_FORMAL_FIXTURE",
        "formal_case_ids_used": [],
        "routes_passed": routes["passed"],
        "mc_status": mc["status"],
        "mc_precision_verdict_role": "NOT_APPLICABLE_TO_1024_SAMPLE_SMOKE",
        "mc_pipeline_healthy": mc_pipeline_healthy,
        "passed": routes["passed"] and mc_pipeline_healthy,
    }


def main() -> None:
    parser = base_parser(__doc__ or "T0 Theorem 1 validator")
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
    run_dir = new_run_directory(config, "a")
    result = run_theorem1(config)
    write_json(run_dir / "RUN_METADATA.json", {
        "module": "A",
        "environment": environment_metadata(),
        "source_lock": source_lock,
    })
    write_json(run_dir / "A_THEOREM1_RESULTS.json", result)
    write_json(run_dir / "MC_SUMMARIES.json", result["mc_summaries"])
    print(run_dir)
    if result["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
