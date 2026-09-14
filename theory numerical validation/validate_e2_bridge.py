"""T0-D: frozen DDPM/SNR/basis integrity and five-knot FW-v1 advisory."""

from __future__ import annotations

import csv
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from common_linear_gaussian import (
    NumericalFailure,
    PROJECT_ROOT,
    SourceDriftError,
    base_parser,
    environment_metadata,
    load_config,
    new_run_directory,
    scalar_close,
    verify_source_lock,
    write_json,
)
from fwv1_reference import (
    exact_regret,
    golden_section_minimize,
    identity_gap,
    incremental_optimal_objective,
    incremental_weighted_objective,
    optimal_value,
    rational_risk,
    rho_star,
)


def build_schedule_float64(
    timesteps: int, beta_start: float, beta_end: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(timesteps, dtype=np.float64)
    beta = np.float64(beta_start) + (
        np.float64(beta_end) - np.float64(beta_start)
    ) * indices / np.float64(timesteps - 1)
    alpha_bar = np.empty(timesteps, dtype=np.float64)
    accumulator = np.float64(1.0)
    for index, item in enumerate(beta):
        accumulator = np.float64(accumulator * np.float64(1.0 - item))
        alpha_bar[index] = accumulator
    tau = alpha_bar / (1.0 - alpha_bar)
    u = np.clip(np.log(tau), -12.0, 12.0) / 12.0
    return beta, alpha_bar, tau, u


def independent_hat_weights(values: np.ndarray, knots: np.ndarray) -> np.ndarray:
    u = np.asarray(values, dtype=np.float64).reshape(-1)
    q = np.asarray(knots, dtype=np.float64).reshape(-1)
    if q.shape != (5,) or not np.all(np.diff(q) > 0.0):
        raise NumericalFailure("D requires five strictly increasing knots")
    result = np.empty((u.size, 5), dtype=np.float64)
    result[:, 0] = np.maximum(0.0, (q[1] - u) / (q[1] - q[0]))
    for m in (1, 2, 3):
        left = (u - q[m - 1]) / (q[m] - q[m - 1])
        right = (q[m + 1] - u) / (q[m + 1] - q[m])
        result[:, m] = np.maximum(0.0, np.minimum(left, right))
    result[:, 4] = np.maximum(0.0, (u - q[3]) / (q[4] - q[3]))
    return result


def _load_e2_interfaces() -> Any:
    source = PROJECT_ROOT / "experiments/controlled_protocol/e2/e2_impl.py"
    e1_dir = PROJECT_ROOT / "experiments/controlled_protocol/e1"
    e2_dir = source.parent
    for directory in (e1_dir, e2_dir):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    specification = importlib.util.spec_from_file_location("t0_frozen_e2_impl", source)
    if specification is None or specification.loader is None:
        raise NumericalFailure("unable to load frozen E2 implementation")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    try:
        specification.loader.exec_module(module)
    except Exception:
        sys.modules.pop(specification.name, None)
        raise
    return module


def _load_e2_config(relative_path: str) -> dict[str, Any]:
    with (PROJECT_ROOT / Path(relative_path)).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _close_array(
    observed: np.ndarray, expected: np.ndarray, atol: float, rtol: float
) -> bool:
    return bool(np.allclose(observed, expected, atol=atol, rtol=rtol))


def independent_hat_weights_float32(
    values: np.ndarray, knots: np.ndarray
) -> np.ndarray:
    """Independent binary32 implementation of the frozen five-hat semantics."""
    u = np.asarray(values, dtype=np.float32).reshape(-1)
    q = np.asarray(knots, dtype=np.float32).reshape(-1)
    if q.shape != (5,) or not np.all(np.diff(q) > np.float32(0.0)):
        raise NumericalFailure("D float32 reference requires five increasing knots")

    zero = np.float32(0.0)
    result = np.empty((u.size, 5), dtype=np.float32)

    numerator = np.subtract(q[1], u, dtype=np.float32)
    denominator = np.subtract(q[1], q[0], dtype=np.float32)
    endpoint = np.divide(numerator, denominator, dtype=np.float32)
    result[:, 0] = np.maximum(endpoint, zero)

    for m in (1, 2, 3):
        left_numerator = np.subtract(u, q[m - 1], dtype=np.float32)
        left_denominator = np.subtract(q[m], q[m - 1], dtype=np.float32)
        left = np.divide(left_numerator, left_denominator, dtype=np.float32)
        right_numerator = np.subtract(q[m + 1], u, dtype=np.float32)
        right_denominator = np.subtract(q[m + 1], q[m], dtype=np.float32)
        right = np.divide(right_numerator, right_denominator, dtype=np.float32)
        result[:, m] = np.maximum(np.minimum(left, right), zero)

    numerator = np.subtract(u, q[3], dtype=np.float32)
    denominator = np.subtract(q[4], q[3], dtype=np.float32)
    endpoint = np.divide(numerator, denominator, dtype=np.float32)
    result[:, 4] = np.maximum(endpoint, zero)
    return result


def independent_kernel_interpolation_float32(
    hats: np.ndarray, base: np.ndarray, residuals: np.ndarray
) -> np.ndarray:
    """Accumulate the base-plus-four-residual kernel without E2/Torch helpers."""
    weights = np.asarray(hats, dtype=np.float32)
    base32 = np.asarray(base, dtype=np.float32)
    residual32 = np.asarray(residuals, dtype=np.float32)
    if weights.ndim != 2 or weights.shape[1] != 5:
        raise NumericalFailure("D float32 kernel reference requires T-by-5 hats")
    if base32.shape != (3, 3, 3, 3) or residual32.shape != (4, 3, 3, 3, 3):
        raise NumericalFailure("D float32 kernel reference received wrong shapes")

    result = np.broadcast_to(base32, (weights.shape[0],) + base32.shape).copy()
    product = np.empty_like(result)
    for m in range(4):
        np.multiply(
            weights[:, m + 1, None, None, None, None],
            residual32[m][None, ...],
            out=product,
        )
        np.add(result, product, out=result)
    return result


def _float32_semantic_comparison(
    actual: np.ndarray,
    reference: np.ndarray,
    *,
    operation_count: int,
    path_factor: int,
) -> dict[str, Any]:
    """Compare two binary32 paths under a frozen Higham-style gamma budget."""
    observed = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(reference, dtype=np.float32)
    if observed.shape != expected.shape:
        return {
            "passed": False,
            "shape_match": False,
            "actual_shape": tuple(observed.shape),
            "reference_shape": tuple(expected.shape),
        }

    eps32 = float(np.finfo(np.float32).eps)
    unit_roundoff = eps32 / 2.0
    n = int(operation_count)
    paths = int(path_factor)
    if n <= 0 or paths <= 0 or n * unit_roundoff >= 1.0:
        raise NumericalFailure("invalid D float32 operation-count budget")
    gamma_n = float(n * unit_roundoff / (1.0 - n * unit_roundoff))
    finite = bool(np.all(np.isfinite(observed)) and np.all(np.isfinite(expected)))
    actual_max = float(np.max(np.abs(observed.astype(np.float64)))) if observed.size else 0.0
    reference_max = float(np.max(np.abs(expected.astype(np.float64)))) if expected.size else 0.0
    scale = max(1.0, actual_max, reference_max)
    absolute_budget = float(paths * gamma_n * scale)
    absolute_error = float(
        np.max(np.abs(observed.astype(np.float64) - expected.astype(np.float64)))
    ) if observed.size else 0.0
    scale_ulp = float(np.spacing(np.float32(scale)))
    budget_in_scale_ulps = (
        float(absolute_budget / scale_ulp) if scale_ulp > 0.0 else None
    )
    error_budget_ratio = (
        float(absolute_error / absolute_budget) if absolute_budget > 0.0 else None
    )
    return {
        "passed": bool(finite and absolute_error <= absolute_budget),
        "shape_match": True,
        "all_finite": finite,
        "actual_dtype": str(observed.dtype),
        "reference_dtype": str(expected.dtype),
        "float32_epsilon": eps32,
        "float32_unit_roundoff": unit_roundoff,
        "operation_count": n,
        "comparison_path_factor": paths,
        "gamma_n": gamma_n,
        "scale": scale,
        "scale_ulp": scale_ulp,
        "absolute_error_budget": absolute_budget,
        "budget_in_scale_ulps": budget_in_scale_ulps,
        "absolute_max_error": absolute_error,
        "error_budget_ratio": error_budget_ratio,
    }


def _runtime_vs_float64_target_comparison(
    actual_float32: np.ndarray,
    reference_float32: np.ndarray,
    ideal_float64: np.ndarray,
    semantic_comparison: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen triangle-inequality runtime-to-ideal rounding budget."""
    actual = np.asarray(actual_float32, dtype=np.float32)
    reference = np.asarray(reference_float32, dtype=np.float32)
    ideal = np.asarray(ideal_float64, dtype=np.float64)
    shape_match = actual.shape == reference.shape == ideal.shape
    if not shape_match:
        return {
            "passed": False,
            "shape_match": False,
            "actual_shape": tuple(actual.shape),
            "reference_shape": tuple(reference.shape),
            "ideal_shape": tuple(ideal.shape),
        }

    actual64 = actual.astype(np.float64)
    reference64 = reference.astype(np.float64)
    finite = bool(
        np.all(np.isfinite(actual64))
        and np.all(np.isfinite(reference64))
        and np.all(np.isfinite(ideal))
    )
    absolute_error = float(np.max(np.abs(actual64 - ideal))) if actual.size else 0.0
    reference_displacement = (
        float(np.max(np.abs(reference64 - ideal))) if reference.size else 0.0
    )
    semantic_budget = float(semantic_comparison["absolute_error_budget"])
    total_budget = float(reference_displacement + semantic_budget)
    scale = max(1.0, float(np.max(np.abs(ideal))) if ideal.size else 0.0)
    scale_aware_error = float(absolute_error / scale)
    scale_ulp = float(np.spacing(np.float32(scale)))
    budget_in_scale_ulps = float(total_budget / scale_ulp) if scale_ulp > 0.0 else None
    error_budget_ratio = float(absolute_error / total_budget) if total_budget > 0.0 else None
    return {
        "passed": bool(finite and absolute_error <= total_budget),
        "shape_match": True,
        "all_finite": finite,
        "actual_dtype": str(actual.dtype),
        "reference_dtype": str(reference.dtype),
        "ideal_dtype": str(ideal.dtype),
        "float32_epsilon": float(np.finfo(np.float32).eps),
        "float32_unit_roundoff": float(np.finfo(np.float32).eps / 2.0),
        "scale": scale,
        "scale_ulp": scale_ulp,
        "absolute_max_error": absolute_error,
        "scale_aware_error": scale_aware_error,
        "independent_reference_rounding_displacement": reference_displacement,
        "semantic_execution_budget": semantic_budget,
        "derived_rounding_budget": total_budget,
        "budget_in_scale_ulps": budget_in_scale_ulps,
        "error_budget_ratio": error_budget_ratio,
    }


def _nodal_kernels(nodes: np.ndarray) -> np.ndarray:
    kernels = np.zeros((5, 3, 3, 3, 3), dtype=np.float64)
    for node, rho in enumerate(nodes):
        for channel in range(3):
            kernels[node, channel, channel, 1, 1] = 1.0
            kernels[node, channel, channel, 1, 0] = rho
            kernels[node, channel, channel, 1, 2] = rho
    return kernels


def _incremental_mean(values: Sequence[float]) -> float:
    total = np.float64(0.0)
    weight = np.float64(1.0 / len(values))
    for value in values:
        total = np.float64(total + weight * np.float64(value))
    return float(total)


def _bridge_profile(
    tau: np.ndarray, u: np.ndarray, hats: np.ndarray, nodes: np.ndarray, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    exact_rho = rho_star(tau)
    approximate_rho = hats @ nodes
    rho_error = np.abs(approximate_rho - exact_rho)
    regret = exact_regret(approximate_rho, tau)
    repairable = identity_gap(tau)
    per_coordinate = regret / 16.0
    ratio_floor = float(config["d"]["ratio_denominator_floor"])
    ratio = np.full(1000, np.nan, dtype=np.float64)
    defined = repairable > ratio_floor
    ratio[defined] = regret[defined] / repairable[defined]
    rows: list[dict[str, Any]] = []
    for timestep in range(1000):
        rows.append(
            {
                "timestep": timestep,
                "tau": float(tau[timestep]),
                "u": float(u[timestep]),
                "rho_star": float(exact_rho[timestep]),
                "rho_hat_5": float(approximate_rho[timestep]),
                "rho_absolute_error": float(rho_error[timestep]),
                "regret": float(regret[timestep]),
                "per_coordinate_regret": float(per_coordinate[timestep]),
                "identity_repairable_gap": float(repairable[timestep]),
                "regret_over_identity_gap": float(ratio[timestep]) if defined[timestep] else None,
                **{f"hat_{index}": float(hats[timestep, index]) for index in range(5)},
            }
        )

    total_regret = np.float64(0.0)
    total_repairable = np.float64(0.0)
    for timestep in range(1000):
        total_regret = np.float64(total_regret + regret[timestep])
        total_repairable = np.float64(total_repairable + repairable[timestep])
    capture = float(1.0 - total_regret / total_repairable)

    bins = []
    width = int(config["d"]["bin_width"])
    for bin_id in range(int(config["d"]["bins"])):
        start, stop = bin_id * width, (bin_id + 1) * width
        bin_regret = np.float64(0.0)
        bin_repairable = np.float64(0.0)
        for timestep in range(start, stop):
            bin_regret = np.float64(bin_regret + regret[timestep])
            bin_repairable = np.float64(bin_repairable + repairable[timestep])
        bin_capture = (
            float(1.0 - bin_regret / bin_repairable)
            if bin_repairable > float(config["d"]["bin_capture_denominator_floor"])
            else None
        )
        bins.append(
            {
                "bin": bin_id,
                "start": start,
                "stop_exclusive": stop,
                "rho_error_max": float(np.max(rho_error[start:stop])),
                "rho_error_rmse": float(math.sqrt(np.mean(rho_error[start:stop] ** 2))),
                "regret_sum": float(bin_regret),
                "identity_gap_sum": float(bin_repairable),
                "capture": bin_capture,
            }
        )

    defined_ratio = ratio[defined]
    quantiles = {
        str(quantile): float(np.quantile(defined_ratio, quantile))
        for quantile in (0.0, 0.25, 0.5, 0.75, 1.0)
    }
    summary = {
        "rho_error_max": float(np.max(rho_error)),
        "rho_error_rmse": float(math.sqrt(np.mean(rho_error * rho_error))),
        "regret_max": float(np.max(regret)),
        "regret_mean": float(np.mean(regret)),
        "per_coordinate_regret_max": float(np.max(per_coordinate)),
        "per_coordinate_regret_mean": float(np.mean(per_coordinate)),
        "overall_capture": capture,
        "bins": bins,
        "regret_over_identity_gap_quantiles": quantiles,
        "undefined_ratio_count": int(np.count_nonzero(~defined)),
        "exact_policy_integrated_risk": _incremental_mean(optimal_value(tau).tolist()),
        "five_knot_integrated_risk": _incremental_mean(rational_risk(approximate_rho, tau).tolist()),
    }
    return rows, summary


def run_bridge(config: Mapping[str, Any]) -> dict[str, Any]:
    d = config["d"]
    tolerances = config["tolerances"]
    e2_config = _load_e2_config(str(d["e2_config_path"]))
    module = _load_e2_interfaces()
    beta, alpha_bar, tau, u = build_schedule_float64(
        int(d["timesteps"]), float(d["beta_start"]), float(d["beta_end"])
    )
    reference_beta, reference_alpha, reference_u = module.build_schedule(e2_config)
    reference_tau = reference_alpha / (1.0 - reference_alpha)
    schedule_atol = float(tolerances["schedule_atol"])
    schedule_rtol = float(tolerances["schedule_rtol"])
    schedule_checks = {
        "config_timesteps": int(e2_config["diffusion"]["timesteps"]) == int(d["timesteps"]),
        "config_beta_start": float(e2_config["diffusion"]["beta_start"]) == float(d["beta_start"]),
        "config_beta_end": float(e2_config["diffusion"]["beta_end"]) == float(d["beta_end"]),
        "beta_values": _close_array(beta, reference_beta, schedule_atol, schedule_rtol),
        "alpha_bar_values": _close_array(alpha_bar, reference_alpha, schedule_atol, schedule_rtol),
        "tau_values": _close_array(tau, reference_tau, schedule_atol, schedule_rtol),
        "u_values": _close_array(u, reference_u, schedule_atol, schedule_rtol),
        "beta_strict_increase": bool(np.all(np.diff(beta) > 0.0)),
        "alpha_bar_strict_decrease": bool(np.all(np.diff(alpha_bar) < 0.0)),
        "tau_strict_decrease": bool(np.all(np.diff(tau) < 0.0)),
        "u_strict_decrease": bool(np.all(np.diff(u) < 0.0)),
    }

    knot_times = np.asarray(d["knot_timesteps"], dtype=np.int64)
    config_knot_times = np.asarray(e2_config["diffusion"]["knot_timesteps"], dtype=np.int64)
    knots = u[knot_times]
    reference_knots = module.schedule_knots(e2_config)
    expected_alpha = np.asarray(d["expected_alpha_bar_knots"], dtype=np.float64)
    expected_u = np.asarray(d["expected_u_knots"], dtype=np.float64)
    hats = independent_hat_weights(u, knots)
    reference_hats = module.numpy_hat_weights(reference_u, reference_knots)
    knot_hats = independent_hat_weights(knots, knots)
    adapter = module.TimeAdapter(reference_knots).cpu().eval()
    runtime_u = module.torch.from_numpy(reference_u).to(dtype=module.torch.float32)
    runtime_resolution = d["float32_runtime_resolution"]
    runtime_operation_count = int(runtime_resolution["operation_count"])
    runtime_path_factor = int(runtime_resolution["comparison_path_factor"])
    with module.torch.no_grad():
        runtime_hats_float32 = (
            module.hat_weights(runtime_u, adapter.knots)
            .detach()
            .cpu()
            .numpy()
        )
    reference_hats_float32 = independent_hat_weights_float32(
        reference_u, reference_knots
    )
    runtime_hat_semantic = _float32_semantic_comparison(
        runtime_hats_float32,
        reference_hats_float32,
        operation_count=runtime_operation_count,
        path_factor=runtime_path_factor,
    )
    basis_checks = {
        "knot_timesteps": bool(np.array_equal(knot_times, config_knot_times)),
        "knots_strict_increase": bool(np.all(np.diff(knots) > 0.0)),
        "alpha_bar_knot_constants": _close_array(alpha_bar[knot_times], expected_alpha, schedule_atol, schedule_rtol),
        "u_knot_constants": _close_array(knots, expected_u, schedule_atol, schedule_rtol),
        "e2_knots": _close_array(knots, reference_knots, schedule_atol, schedule_rtol),
        "e2_hat_values": _close_array(hats, reference_hats, schedule_atol, schedule_rtol),
        "e2_runtime_float32_hat_values": bool(runtime_hat_semantic["passed"]),
        "hat_nonnegative": bool(np.min(hats) >= float(tolerances["hat_negative_floor"])),
        "partition_of_unity": bool(np.max(np.abs(np.sum(hats, axis=1) - 1.0)) <= schedule_atol),
        "support_coverage": bool(np.all(np.sum(hats > 0.0, axis=1) >= 1)),
        "knot_interpolation": _close_array(knot_hats, np.eye(5), schedule_atol, schedule_rtol),
    }

    node_tau = tau[knot_times]
    nodes = rho_star(node_tau)
    approximate_rho = hats @ nodes
    nodal_kernels = _nodal_kernels(nodes)
    base = nodal_kernels[0]
    residuals = nodal_kernels[1:] - base
    adapter_kernels = base[None, ...] + np.einsum("tm,moihw->toihw", hats[:, 1:], residuals)
    interpolated_kernels = np.einsum("tm,moihw->toihw", hats, nodal_kernels)
    containment_error = float(np.max(np.abs(adapter_kernels - interpolated_kernels)))
    with module.torch.no_grad():
        adapter.base.copy_(module.torch.from_numpy(base).to(dtype=adapter.base.dtype))
        adapter.residual.copy_(
            module.torch.from_numpy(residuals).to(dtype=adapter.residual.dtype)
        )
        runtime_adapter_kernels_float32 = (
            adapter.kernels(runtime_u)
            .detach()
            .cpu()
            .numpy()
        )
    reference_kernels_float32 = independent_kernel_interpolation_float32(
        reference_hats_float32, base, residuals
    )
    runtime_kernel_semantic = _float32_semantic_comparison(
        runtime_adapter_kernels_float32,
        reference_kernels_float32,
        operation_count=runtime_operation_count,
        path_factor=runtime_path_factor,
    )
    runtime_target_comparison = _runtime_vs_float64_target_comparison(
        runtime_adapter_kernels_float32,
        reference_kernels_float32,
        interpolated_kernels,
        runtime_kernel_semantic,
    )
    runtime_containment_error = float(runtime_kernel_semantic["absolute_max_error"])
    runtime_target_error = float(runtime_target_comparison["absolute_max_error"])
    containment_checks = {
        "base_shape": tuple(adapter.base.shape) == (3, 3, 3, 3),
        "residual_shape": tuple(adapter.residual.shape) == (4, 3, 3, 3, 3),
        "parameter_count_405": sum(parameter.numel() for parameter in adapter.parameters()) == 405,
        "kernel_interpolation": containment_error
        <= schedule_atol + schedule_rtol * max(float(np.max(np.abs(adapter_kernels))), float(np.max(np.abs(interpolated_kernels)))),
        "e2_runtime_kernel_interpolation": bool(runtime_kernel_semantic["passed"]),
        "e2_runtime_vs_float64_target": bool(runtime_target_comparison["passed"]),
        "rho_finite": bool(np.all(np.isfinite(approximate_rho))),
        "rho_convex_hull": bool(
            np.all(approximate_rho >= np.min(nodes) - schedule_atol)
            and np.all(approximate_rho <= np.max(nodes) + schedule_atol)
        ),
        "rho_legal_domain": bool(np.all(approximate_rho != -0.5)),
    }

    profile_rows, approximation = _bridge_profile(tau, u, hats, nodes, config)
    regret_values = np.asarray([row["regret"] for row in profile_rows], dtype=np.float64)
    profile_checks = {
        "profile_complete": len(profile_rows) == 1000 and [row["timestep"] for row in profile_rows] == list(range(1000)),
        "regret_nonnegative": bool(np.min(regret_values) >= float(tolerances["nonnegative_floor"])),
        "nodal_rho_exact": _close_array(approximate_rho[knot_times], nodes, schedule_atol, schedule_rtol),
    }

    binding_checks = {**schedule_checks, **basis_checks, **containment_checks, **profile_checks}
    integrity = "PASS" if all(binding_checks.values()) else "E2_REVIEW_REQUIRED"

    weights = [1.0 / 1000.0] * 1000
    static_objective = lambda rho: incremental_weighted_objective(rho, tau.tolist(), weights)
    try:
        static_solution = golden_section_minimize(
            static_objective,
            interval_tolerance=float(tolerances["optimizer_interval"]),
            maximum_iterations=int(tolerances["optimizer_max_iterations"]),
        )
        dense_start, dense_stop, dense_count = config["b"]["dense_rho"]
        dense_rho = np.linspace(
            dense_start, dense_stop, int(dense_count), dtype=np.float64
        )
        dense_rho = dense_rho[dense_rho != np.float64(-0.5)]
        dense_values = np.asarray(
            [static_objective(rho) for rho in dense_rho], dtype=np.float64
        )
        if not np.all(np.isfinite(dense_values)):
            raise NumericalFailure("D best-static dense cross-check is nonfinite")
        static_dense_minimum = float(np.min(dense_values))
        approximation["best_static_computation_status"] = "PASS"
        approximation["best_static_integrated_risk"] = static_solution.objective
        approximation["best_static_rho"] = static_solution.x
        approximation["best_static_solver"] = static_solution.to_dict()
        approximation["best_static_dense_grid_minimum"] = static_dense_minimum
        approximation["best_static_dense_grid_crosscheck"] = bool(
            static_solution.objective <= static_dense_minimum + 1e-10
        )
        approximation["exact_policy_vs_static_advantage"] = (
            static_solution.objective - approximation["exact_policy_integrated_risk"]
        )
    except NumericalFailure as error:
        approximation["best_static_computation_status"] = "INCONCLUSIVE_NUMERICAL_DIAGNOSTIC"
        approximation["best_static_computation_error"] = str(error)
        approximation["best_static_integrated_risk"] = None
        approximation["best_static_rho"] = None
        approximation["best_static_solver"] = None
        approximation["best_static_dense_grid_minimum"] = None
        approximation["best_static_dense_grid_crosscheck"] = None
        approximation["exact_policy_vs_static_advantage"] = None
    approximation["best_static_dense_grid_role"] = (
        "diagnostic duplicate of binding B-POL-04; does not alter D bridge integrity "
        "or approximation advisory"
    )
    advisory_thresholds = d["advisory_reference_thresholds"]
    bin_checks = [
        row["capture"] >= float(advisory_thresholds["bin_capture_min"])
        for row in approximation["bins"]
        if row["capture"] is not None
    ]
    advisory_checks = {
        "overall_capture": approximation["overall_capture"] >= float(advisory_thresholds["overall_capture_min"]),
        "all_defined_bins_capture": all(bin_checks),
        "max_per_coordinate_regret": approximation["per_coordinate_regret_max"]
        <= float(advisory_thresholds["max_per_coordinate_regret"]),
        "mean_per_coordinate_regret": approximation["per_coordinate_regret_mean"]
        <= float(advisory_thresholds["mean_per_coordinate_regret"]),
    }
    if integrity != "PASS":
        advisory = "NOT_EVALUABLE"
    elif all(advisory_checks.values()):
        advisory = "WITHIN_ADVISORY_REFERENCE_THRESHOLDS"
    else:
        advisory = "BRIDGE_APPROXIMATION_REVIEW_RECOMMENDED"
    return {
        "module": "D",
        "case_id": "D-BRIDGE",
        "theory_scope": "FW-v1 scalar slice mapped to the frozen E2 DDPM schedule and five-knot basis",
        "evidence_type": "binding interface regression plus non-binding approximation advisory",
        "claim_status_changed": False,
        "e2_modified": False,
        "heldout_accessed": False,
        "D_BRIDGE_INTEGRITY_VERDICT": integrity,
        "D_APPROXIMATION_ADVISORY": advisory,
        "binding_checks": binding_checks,
        "schedule": {
            "checks": schedule_checks,
            "beta_start": float(beta[0]),
            "beta_end": float(beta[-1]),
            "alpha_bar_start": float(alpha_bar[0]),
            "alpha_bar_end": float(alpha_bar[-1]),
            "tau_start": float(tau[0]),
            "tau_end": float(tau[-1]),
            "u_start": float(u[0]),
            "u_end": float(u[-1]),
        },
        "basis": {
            "checks": basis_checks,
            "knot_timesteps": knot_times,
            "knots": knots,
            "actual_e2_float32_hat_semantic_comparison": runtime_hat_semantic,
        },
        "containment": {
            "checks": containment_checks,
            "float64_algebraic_max_absolute_error": containment_error,
            "e2_runtime_max_absolute_error": runtime_containment_error,
            "e2_runtime_vs_float64_target_max_absolute_error": runtime_target_error,
            "runtime_dtype": str(adapter.base.dtype),
            "nodal_rho": nodes,
            "tap_orientation": "same-channel center plus horizontal left/right",
            "validation_layers": {
                "FLOAT64_ALGEBRAIC_CONTAINMENT": {
                    "passed": bool(containment_checks["kernel_interpolation"]),
                    "absolute_max_error": containment_error,
                    "absolute_tolerance": schedule_atol
                    + schedule_rtol
                    * max(
                        float(np.max(np.abs(adapter_kernels))),
                        float(np.max(np.abs(interpolated_kernels))),
                    ),
                    "dtype": "numpy.float64",
                },
                "ACTUAL_E2_FLOAT32_SEMANTIC_CONTAINMENT": {
                    "hat_values": runtime_hat_semantic,
                    "kernel_interpolation": runtime_kernel_semantic,
                    "actual_runtime": "frozen TimeAdapter in torch.float32",
                    "independent_reference": "NumPy float32 explicit hats and ordered base-plus-four-residual accumulation",
                },
                "FLOAT32_RUNTIME_VS_IDEAL_FLOAT64_TARGET": runtime_target_comparison,
            },
        },
        "approximation_metrics": approximation,
        "ADVISORY_REFERENCE_THRESHOLDS": advisory_thresholds,
        "advisory_checks": advisory_checks,
        "profile_rows": profile_rows,
    }


def write_profile_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def smoke_check() -> dict[str, Any]:
    beta, alpha, tau, u = build_schedule_float64(7, 1e-4, 0.02)
    knot_times = np.asarray((6, 4, 3, 1, 0), dtype=np.int64)
    knots = u[knot_times]
    hats = independent_hat_weights(u, knots)
    nodes = np.asarray((0.02, 0.08, 0.12, 0.18, 0.25), dtype=np.float64)
    kernels = _nodal_kernels(nodes)
    base = kernels[0]
    residual = kernels[1:] - base
    left = base[None, ...] + np.einsum("tm,moihw->toihw", hats[:, 1:], residual)
    right = np.einsum("tm,moihw->toihw", hats, kernels)
    hats32 = independent_hat_weights_float32(u, knots)
    kernels32 = independent_kernel_interpolation_float32(hats32, base, residual)
    semantic32 = _float32_semantic_comparison(
        kernels32,
        kernels32.copy(),
        operation_count=8,
        path_factor=2,
    )
    runtime_target32 = _runtime_vs_float64_target_comparison(
        kernels32,
        kernels32.copy(),
        right,
        semantic32,
    )
    checks = {
        "schedule_finite": bool(np.all(np.isfinite(beta)) and np.all(np.isfinite(alpha)) and np.all(np.isfinite(tau))),
        "knots_increasing": bool(np.all(np.diff(knots) > 0.0)),
        "partition": bool(np.max(np.abs(np.sum(hats, axis=1) - 1.0)) <= 5e-13),
        "toy_containment": bool(np.max(np.abs(left - right)) <= 5e-13),
        "float32_reference_finite": bool(
            np.all(np.isfinite(hats32)) and np.all(np.isfinite(kernels32))
        ),
        "float32_semantic_budget": bool(semantic32["passed"]),
        "float32_runtime_to_ideal_budget": bool(runtime_target32["passed"]),
    }
    return {
        "mode": "SMOKE_ONLY_SEVEN_STEP_TOY_SCHEDULE",
        "formal_e2_schedule_used": False,
        "formal_case_ids_used": [],
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = base_parser(__doc__ or "T0 E2 bridge validator")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.smoke:
            result = smoke_check()
            print(result)
            if not result["passed"]:
                raise SystemExit(1)
            return
        source_lock = verify_source_lock(config)
    except SourceDriftError as error:
        print(f"SOURCE_DRIFT_REVIEW_REQUIRED: {error}")
        raise SystemExit(2) from error
    run_dir = new_run_directory(config, "d")
    result = run_bridge(config)
    write_json(run_dir / "RUN_METADATA.json", {"module": "D", "environment": environment_metadata(), "source_lock": source_lock})
    write_json(run_dir / "D_E2_BRIDGE_RESULTS.json", result)
    write_profile_csv(run_dir / "D_E2_BRIDGE_PROFILE.csv", result["profile_rows"])
    print(run_dir)
    if result["D_BRIDGE_INTEGRITY_VERDICT"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
