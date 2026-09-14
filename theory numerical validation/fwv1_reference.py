"""Independent float64 FW-v1 constructions and exact scalar references for T0."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from common_linear_gaussian import (
    NumericalFailure,
    cholesky_solve,
    make_rng,
    psd_square_root,
    rank_diagnostics,
)


def normalized_pair_synthesis() -> np.ndarray:
    matrix = np.zeros((8, 4), dtype=np.float64)
    scale = np.float64(1.0 / math.sqrt(2.0))
    for column in range(4):
        matrix[2 * column, column] = scale
        matrix[2 * column + 1, column] = scale
    return matrix


def circular_shift_8() -> np.ndarray:
    shift = np.zeros((8, 8), dtype=np.float64)
    for row in range(8):
        shift[row, (row - 1) % 8] = 1.0
    return shift


@dataclass(frozen=True)
class FWV1:
    pair: np.ndarray
    target: np.ndarray
    shift: np.ndarray
    adjacency: np.ndarray
    h1: np.ndarray
    w: np.ndarray
    s0: np.ndarray
    sigma0: np.ndarray


class OutOfDomainRankError(NumericalFailure):
    """Structured full-rank-domain rejection for the FW-v1 measurement API."""

    def __init__(self, diagnostics: Any) -> None:
        self.diagnostics = diagnostics
        self.structured_result = f"OUT_OF_DOMAIN_RANK_{diagnostics.rank}"
        super().__init__(self.structured_result)


def build_fwv1() -> FWV1:
    pair = normalized_pair_synthesis()
    shift = circular_shift_8()
    adjacency = (shift + shift.T) / np.float64(2.0)
    h1 = np.eye(8, dtype=np.float64) / 2.0 + adjacency / 4.0
    w = cholesky_solve(h1, np.eye(8, dtype=np.float64))
    s0 = 2.0 * w - np.eye(8, dtype=np.float64)
    sigma0 = np.kron(s0, np.eye(8, dtype=np.float64))
    target = np.kron(pair, pair)
    return FWV1(pair, target, shift, adjacency, h1, w, s0, sigma0)


def k_rho(model: FWV1, rho: float) -> np.ndarray:
    return np.eye(8, dtype=np.float64) + np.float64(2.0 * rho) * model.adjacency


def a_rho(model: FWV1, rho: float) -> np.ndarray:
    return np.kron(k_rho(model, rho), np.eye(8, dtype=np.float64))


def measurement(model: FWV1, rho: float) -> np.ndarray:
    return a_rho(model, rho).T @ model.target


def q_polynomial(rho: float | np.ndarray, tau: float | np.ndarray) -> np.ndarray:
    r = np.asarray(rho, dtype=np.float64)
    t = np.asarray(tau, dtype=np.float64)
    return (14.0 + 34.0 * t) * r * r + (14.0 + 18.0 * t) * r + 7.0 + 17.0 * t


def d_polynomial(tau: float | np.ndarray) -> np.ndarray:
    t = np.asarray(tau, dtype=np.float64)
    return 49.0 + 350.0 * t + 497.0 * t * t


def f_value(tau: float | np.ndarray) -> np.ndarray:
    t = np.asarray(tau, dtype=np.float64)
    return 16.0 - 12.0 / (3.0 + 5.0 * t) - 4.0 / (1.0 + 3.0 * t)


def rational_risk(rho: float | np.ndarray, tau: float | np.ndarray) -> np.ndarray:
    r = np.asarray(rho, dtype=np.float64)
    t = np.asarray(tau, dtype=np.float64)
    denominator = q_polynomial(r, t)
    if np.any(~np.isfinite(denominator)) or np.any(denominator <= 0.0):
        raise NumericalFailure("q(rho,tau) must be positive and finite")
    return f_value(t) - 56.0 * (r + 1.0) ** 2 / denominator


def rho_star(tau: float | np.ndarray) -> np.ndarray:
    t = np.asarray(tau, dtype=np.float64)
    if np.any(t <= 0.0) or np.any(~np.isfinite(t)):
        raise NumericalFailure("rho_star requires finite tau>0")
    return 8.0 * t / (7.0 + 25.0 * t)


def optimal_value(tau: float | np.ndarray) -> np.ndarray:
    t = np.asarray(tau, dtype=np.float64)
    return f_value(t) - 56.0 * (7.0 + 33.0 * t) / d_polynomial(t)


def exact_regret(rho: float | np.ndarray, tau: float | np.ndarray) -> np.ndarray:
    r = np.asarray(rho, dtype=np.float64)
    t = np.asarray(tau, dtype=np.float64)
    denominator = d_polynomial(t) * q_polynomial(r, t)
    if np.any(~np.isfinite(denominator)) or np.any(denominator <= 0.0):
        raise NumericalFailure("regret denominator must be positive and finite")
    return 56.0 * ((7.0 + 25.0 * t) * r - 8.0 * t) ** 2 / denominator


def identity_gap(tau: float | np.ndarray) -> np.ndarray:
    t = np.asarray(tau, dtype=np.float64)
    denominator = d_polynomial(t) * (7.0 + 17.0 * t)
    return 3584.0 * t * t / denominator


def witness_relative_gain(tau: float | np.ndarray) -> np.ndarray:
    t = np.asarray(tau, dtype=np.float64)
    return 4.0 * (47.0 * t - 7.0) / ((7.0 + 17.0 * t) * (13.0 + 27.0 * t))


def four_mode_risk(rho: float, tau: float) -> float:
    r = np.float64(rho)
    t = np.float64(tau)
    q = float(q_polynomial(r, t))
    c0 = 1.0 + 2.0 * r
    c1 = 1.0 + r
    d0 = c0 * c0 * (3.0 + 5.0 * t) / 3.0
    d1 = q / 7.0
    d2 = 1.0 + 3.0 * t
    if min(d0, d1, d2) <= 0.0:
        raise NumericalFailure("four-mode denominator is nonpositive")
    return float(16.0 - 4.0 * (c0 * c0 / d0 + 2.0 * c1 * c1 / d1 + 1.0 / d2))


def ambient_measurement_risk(
    model: FWV1, rho: float, tau: float, sigma0: np.ndarray | None = None
) -> dict[str, Any]:
    covariance = model.sigma0 if sigma0 is None else np.asarray(sigma0, dtype=np.float64)
    sigma_tau = np.eye(64, dtype=np.float64) + np.float64(tau) * covariance
    measured = measurement(model, rho)
    diagnostics = rank_diagnostics(measured)
    if not diagnostics.full_column_rank:
        raise OutOfDomainRankError(diagnostics)
    gram = measured.T @ sigma_tau @ measured
    cross = model.target.T @ measured
    risk = 16.0 - np.trace(cross @ cholesky_solve(gram, cross.T))
    return {
        "risk": float(risk),
        "measurement_rank": diagnostics.to_dict(),
        "gram_condition": rank_diagnostics(gram).condition_number,
    }


def ambient_full_risk(model: FWV1, tau: float, sigma0: np.ndarray | None = None) -> float:
    covariance = model.sigma0 if sigma0 is None else np.asarray(sigma0, dtype=np.float64)
    sigma_tau = np.eye(64, dtype=np.float64) + np.float64(tau) * covariance
    return float(16.0 - np.trace(model.target.T @ cholesky_solve(sigma_tau, model.target)))


def l_tau_eigenvalues(tau: float) -> tuple[float, float, float, float]:
    t = np.float64(tau)
    ell0 = 3.0 * (7.0 + 25.0 * t) / ((3.0 + 5.0 * t) * (7.0 + 41.0 * t))
    ell1 = 7.0 * (7.0 + 25.0 * t) / d_polynomial(t)
    ell2 = 1.0 / (1.0 + 3.0 * t)
    return float(ell0), float(ell1), float(ell2), float(ell1)


def l_tau_matrix(tau: float) -> np.ndarray:
    ell0, ell1, ell2, _ = l_tau_eigenvalues(tau)
    c0 = (ell0 + 2.0 * ell1 + ell2) / 4.0
    c1 = (ell0 - ell2) / 4.0
    c2 = (ell0 - 2.0 * ell1 + ell2) / 4.0
    first = np.asarray((c0, c1, c2, c1), dtype=np.float64)
    return np.vstack([np.roll(first, row) for row in range(4)]).astype(np.float64)


@dataclass(frozen=True)
class GoldenResult:
    x: float
    objective: float
    iterations: int
    interval: tuple[float, float]
    candidates: tuple[tuple[float, float], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "objective": self.objective,
            "iterations": self.iterations,
            "interval": list(self.interval),
            "candidates": [list(item) for item in self.candidates],
        }


def golden_section_minimize(
    objective: Callable[[np.float64], np.float64],
    *,
    lo: float = 0.0,
    hi: float = 8.0 / 25.0,
    interval_tolerance: float = 1e-12,
    maximum_iterations: int = 128,
) -> GoldenResult:
    lower = np.float64(lo)
    upper = np.float64(hi)
    g = np.float64((math.sqrt(5.0) - 1.0) / 2.0)

    def evaluate(x: np.float64) -> np.float64:
        if x < lo or x > hi:
            raise NumericalFailure(f"golden-section evaluation outside [{lo},{hi}]: {x}")
        value = np.float64(objective(np.float64(x)))
        if not np.isfinite(value):
            raise NumericalFailure("golden-section objective is nonfinite")
        return value

    endpoint_lo = evaluate(lower)
    endpoint_hi = evaluate(upper)
    x1 = upper - g * (upper - lower)
    x2 = lower + g * (upper - lower)
    f1 = evaluate(x1)
    f2 = evaluate(x2)
    iterations = 0
    while float(upper - lower) > interval_tolerance and iterations < maximum_iterations:
        if f1 <= f2:
            upper, x2, f2 = x2, x1, f1
            x1 = upper - g * (upper - lower)
            f1 = evaluate(x1)
        else:
            lower, x1, f1 = x1, x2, f2
            x2 = lower + g * (upper - lower)
            f2 = evaluate(x2)
        iterations += 1
    if float(upper - lower) > interval_tolerance:
        raise NumericalFailure("golden-section interval failed to converge in 128 iterations")
    midpoint = np.float64((lower + upper) / 2.0)
    candidates = [
        (np.float64(lo), endpoint_lo),
        (np.float64(hi), endpoint_hi),
        (midpoint, evaluate(midpoint)),
        (x1, f1),
        (x2, f2),
    ]
    candidates.sort(key=lambda item: (float(item[1]), float(item[0])))
    winner = candidates[0]
    return GoldenResult(
        x=float(winner[0]),
        objective=float(winner[1]),
        iterations=iterations,
        interval=(float(lower), float(upper)),
        candidates=tuple((float(x), float(y)) for x, y in candidates),
    )


def incremental_weighted_objective(
    rho: float, tau_values: Sequence[float], weights: Sequence[float]
) -> np.float64:
    if len(tau_values) != len(weights):
        raise NumericalFailure("tau and weight lengths differ")
    total = np.float64(0.0)
    for tau, weight in zip(tau_values, weights, strict=True):
        total = np.float64(
            total + np.float64(weight) * np.float64(rational_risk(rho, tau))
        )
    return total


def incremental_optimal_objective(
    tau_values: Sequence[float], weights: Sequence[float]
) -> np.float64:
    if len(tau_values) != len(weights):
        raise NumericalFailure("tau and weight lengths differ")
    total = np.float64(0.0)
    for tau, weight in zip(tau_values, weights, strict=True):
        total = np.float64(total + np.float64(weight) * np.float64(optimal_value(tau)))
    return total


def fwv1_monte_carlo(
    model: FWV1,
    *,
    tau: float,
    tau_token: str,
    rho: float,
    policy_tag: str,
    root_seed: int,
    mc_config: Mapping[str, Any],
    sample_checkpoints: Sequence[int] | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    from common_linear_gaussian import RunningMoments, assess_mc

    allowed_tags = {"identity", "optimal", "quarter"}
    if policy_tag not in allowed_tags:
        raise NumericalFailure(f"illegal B MC policy tag: {policy_tag}")
    policy_rhos = {
        "identity": 0.0,
        "optimal": float(rho_star(tau)),
        "quarter": 0.25,
    }
    if float(rho) != policy_rhos[policy_tag]:
        raise NumericalFailure(
            f"B MC rho/tag mismatch: tag={policy_tag}, rho={rho}, "
            f"expected={policy_rhos[policy_tag]}"
        )
    case_id = f"B-MC/{tau_token}/{policy_tag}"
    rng, seed = make_rng(root_seed, case_id, "mc-samples")
    checkpoints = tuple(
        int(item)
        for item in (sample_checkpoints or mc_config["sample_checkpoints"])
    )
    batch = int(batch_size or mc_config["batch_size"])
    if sorted(checkpoints) != list(checkpoints) or any(item % batch for item in checkpoints):
        raise NumericalFailure("B MC checkpoints must be increasing multiples of batch size")

    sigma_tau = np.eye(64, dtype=np.float64) + np.float64(tau) * model.sigma0
    prediction_coefficients: dict[str, np.ndarray] = {}
    exact_joint: dict[str, float] = {}
    for tag in ("identity", "optimal", "quarter"):
        measured_for_tag = measurement(model, policy_rhos[tag])
        gram_for_tag = measured_for_tag.T @ sigma_tau @ measured_for_tag
        beta_for_tag = cholesky_solve(
            gram_for_tag, measured_for_tag.T @ model.target
        )
        prediction_coefficients[tag] = measured_for_tag @ beta_for_tag
        exact_joint[f"risk_{tag}"] = float(
            ambient_measurement_risk(model, policy_rhos[tag], tau)["risk"]
        )
    beta_full = cholesky_solve(sigma_tau, model.target)
    square_root_diagnostics: dict[str, Any] = {}
    sigma0_root = psd_square_root(
        model.sigma0, diagnostics=square_root_diagnostics
    )
    exact_full = float(ambient_full_risk(model, tau))
    exact_joint["risk_full"] = exact_full
    for tag in ("identity", "optimal", "quarter"):
        exact_joint[f"gap_{tag}"] = exact_joint[f"risk_{tag}"] - exact_full
    exact = {
        "risk_measurement": exact_joint[f"risk_{policy_tag}"],
        "risk_full": exact_full,
    }
    exact["gap"] = exact["risk_measurement"] - exact["risk_full"]

    moments = {
        "risk_measurement": RunningMoments(),
        "risk_full": RunningMoments(),
        "gap": RunningMoments(),
    }
    joint_moments = {
        key: RunningMoments()
        for key in (
            "risk_full",
            "risk_identity",
            "risk_optimal",
            "risk_quarter",
            "gap_identity",
            "gap_optimal",
            "gap_quarter",
        )
    }
    history: list[dict[str, Any]] = []
    generated = 0
    for checkpoint in checkpoints:
        while generated < checkpoint:
            count = min(batch, checkpoint - generated)
            clean = rng.standard_normal((count, 64)) @ sigma0_root.T
            noise = rng.standard_normal((count, 64))
            z_tau = np.float64(math.sqrt(tau)) * clean + noise
            y = noise @ model.target
            prediction_full = z_tau @ beta_full
            loss_full = np.sum((y - prediction_full) ** 2, axis=1, dtype=np.float64)
            losses_by_tag: dict[str, np.ndarray] = {}
            for tag in ("identity", "optimal", "quarter"):
                prediction = z_tau @ prediction_coefficients[tag]
                losses_by_tag[tag] = np.sum(
                    (y - prediction) ** 2, axis=1, dtype=np.float64
                )
            loss_measurement = losses_by_tag[policy_tag]
            moments["risk_full"].update(loss_full)
            moments["risk_measurement"].update(loss_measurement)
            moments["gap"].update(loss_measurement - loss_full)
            joint_moments["risk_full"].update(loss_full)
            for tag in ("identity", "optimal", "quarter"):
                joint_moments[f"risk_{tag}"].update(losses_by_tag[tag])
                joint_moments[f"gap_{tag}"].update(losses_by_tag[tag] - loss_full)
            generated += count
        summaries = {key: item.summary() for key, item in moments.items()}
        joint_summaries = {
            key: item.summary() for key, item in joint_moments.items()
        }
        assessment = assess_mc(summaries, exact, mc_config)
        history.append(
            {
                "checkpoint": checkpoint,
                "summaries": summaries,
                "assessment": assessment,
                "joint_same_draw_summaries": joint_summaries,
                "joint_exact_references": exact_joint,
            }
        )
        if assessment["all_precision_sufficient"]:
            return {
                "status": "PASS" if assessment["all_consistent"] else "FAIL",
                "case_id": case_id,
                "stream_tag": "mc-samples",
                "root_seed": int(root_seed),
                "derived_seed128": seed,
                "tau_token": tau_token,
                "policy_tag": policy_tag,
                "rho": float(rho),
                "psd_square_root_diagnostics": square_root_diagnostics,
                "history": history,
                "final": history[-1],
            }
    final_assessment = history[-1]["assessment"]
    return {
        "status": (
            "FAIL"
            if not final_assessment["all_consistent"]
            else "INCONCLUSIVE_PRECISION"
        ),
        "case_id": case_id,
        "stream_tag": "mc-samples",
        "root_seed": int(root_seed),
        "derived_seed128": seed,
        "tau_token": tau_token,
        "policy_tag": policy_tag,
        "rho": float(rho),
        "psd_square_root_diagnostics": square_root_diagnostics,
        "history": history,
        "final": history[-1],
    }
