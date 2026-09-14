"""Shared float64 linear-Gaussian machinery for T0.

The functions in this module implement numerical reference calculations only.
They do not establish, strengthen, or change any mathematical claim.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


T0_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = T0_DIR.parents[2]
APPROVED_CONFIG_PATH = T0_DIR / "t0_config.json"
APPROVED_CONFIG_SHA256 = "9481792706d0f3ec65ec9e65b447ad05900218310fac67ce21f3e8b1b425b0cf"
FROZEN_CONDITION_GATE = 1e12


class T0Error(RuntimeError):
    """Base class for fail-closed T0 errors."""


class SourceDriftError(T0Error):
    """Raised before formal computation when the approved source lock differs."""


class NumericalFailure(T0Error):
    """Raised when a frozen numerical precondition or solver route fails."""


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    if config_path != APPROVED_CONFIG_PATH.resolve():
        raise SourceDriftError(
            f"formal/smoke CLI must use the approved config path: {APPROVED_CONFIG_PATH}"
        )
    observed_hash = sha256_file(config_path)
    if observed_hash != APPROVED_CONFIG_SHA256:
        raise SourceDriftError(
            "approved T0 config SHA-256 mismatch: "
            f"expected={APPROVED_CONFIG_SHA256}, observed={observed_hash}"
        )
    with config_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if value.get("dtype") != "float64":
        raise NumericalFailure("T0 configuration must freeze dtype=float64")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_lock(config: Mapping[str, Any], root: Path = PROJECT_ROOT) -> dict[str, Any]:
    lock = config["source_lock"]
    expected = {str(k): str(v) for k, v in lock["files"].items()}
    recursive_roots = tuple(str(item) for item in lock["recursive_roots"])
    actual_recursive: set[str] = set()
    for relative_root in recursive_roots:
        directory = root / Path(relative_root)
        if not directory.is_dir():
            raise SourceDriftError(f"source-lock directory missing: {relative_root}")
        actual_recursive.update(
            path.relative_to(root).as_posix()
            for path in directory.rglob("*")
            if path.is_file()
        )
    expected_recursive = {
        path
        for path in expected
        if any(path == prefix or path.startswith(prefix + "/") for prefix in recursive_roots)
    }
    if actual_recursive != expected_recursive:
        added = sorted(actual_recursive - expected_recursive)
        removed = sorted(expected_recursive - actual_recursive)
        raise SourceDriftError(
            "source-lock path-set mismatch: "
            f"added={added!r}, removed={removed!r}"
        )

    observed: dict[str, str] = {}
    mismatches: list[dict[str, str]] = []
    for relative, expected_hash in sorted(expected.items()):
        source = root / Path(relative)
        if not source.is_file():
            mismatches.append(
                {"path": relative, "expected": expected_hash, "observed": "MISSING"}
            )
            continue
        observed_hash = sha256_file(source)
        observed[relative] = observed_hash
        if observed_hash != expected_hash:
            mismatches.append(
                {"path": relative, "expected": expected_hash, "observed": observed_hash}
            )
    if mismatches:
        raise SourceDriftError(f"source-lock SHA-256 mismatch: {mismatches!r}")
    return {"status": "PASS", "files": observed, "recursive_roots": list(recursive_roots)}


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(jsonable(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def overwrite_smoke_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(jsonable(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def new_run_directory(config: Mapping[str, Any], module: str | None = None) -> Path:
    root = PROJECT_ROOT / Path(str(config["results_root"]))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    suffix = f"-{module.lower()}" if module else ""
    target = root / f"{stamp}{suffix}"
    target.mkdir(parents=True, exist_ok=False)
    return target


def environment_metadata() -> dict[str, Any]:
    configuration_buffer = io.StringIO()
    with contextlib.redirect_stdout(configuration_buffer):
        np.show_config()
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy_build_configuration": configuration_buffer.getvalue().strip(),
        "dtype": "float64",
        "pid": os.getpid(),
    }


def parse_numeric_token(token: str | float | int) -> np.float64:
    if isinstance(token, (float, int)):
        return np.float64(token)
    text = str(token)
    if "/" in text:
        numerator, denominator = text.split("/", maxsplit=1)
        return np.float64(int(numerator) / int(denominator))
    return np.float64(text)


def stable_seed(root_seed: int, case_id: str, stream_tag: str) -> int:
    payload = (
        b"T0-RNG-v1\0"
        + int(root_seed).to_bytes(8, "little", signed=False)
        + b"\0"
        + case_id.encode("ascii")
        + b"\0"
        + stream_tag.encode("ascii")
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "little", signed=False)


def make_rng(root_seed: int, case_id: str, stream_tag: str) -> tuple[np.random.Generator, int]:
    seed = stable_seed(root_seed, case_id, stream_tag)
    return np.random.Generator(np.random.PCG64DXSM(seed)), seed


def deterministic_orthogonal(
    root_seed: int, case_id: str, stream_tag: str, n: int
) -> tuple[np.ndarray, int]:
    rng, seed = make_rng(root_seed, case_id, stream_tag)
    matrix = np.asarray(rng.standard_normal((n, n)), dtype=np.float64, order="C")
    q, r = np.linalg.qr(matrix, mode="reduced")
    diagonal = np.diag(r)
    if np.any(diagonal == 0.0):
        raise NumericalFailure(f"zero QR diagonal in {case_id}/{stream_tag}")
    signs = np.where(diagonal >= 0.0, 1.0, -1.0).astype(np.float64)
    q = np.asarray(q @ np.diag(signs), dtype=np.float64)
    if np.linalg.matrix_rank(q) != n:
        raise NumericalFailure(f"orthogonal construction rank mismatch in {case_id}/{stream_tag}")
    return q, seed


def scalar_close(x: float, y: float, atol: float, rtol: float) -> bool:
    return bool(abs(float(x) - float(y)) <= atol + rtol * max(abs(float(x)), abs(float(y))))


def matrix_close(x: np.ndarray, y: np.ndarray, atol: float, rtol: float) -> bool:
    difference = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    return bool(
        np.linalg.norm(difference, ord="fro")
        <= atol + rtol * max(np.linalg.norm(x, ord="fro"), np.linalg.norm(y, ord="fro"))
    )


def operator_norm_symmetric(matrix: np.ndarray) -> float:
    values = np.linalg.eigvalsh(np.asarray(matrix, dtype=np.float64))
    return float(np.max(np.abs(values)))


@dataclass(frozen=True)
class RankDiagnostics:
    shape: tuple[int, int]
    singular_values: tuple[float, ...]
    threshold: float
    rank: int
    columns: int
    condition_number: float

    @property
    def full_column_rank(self) -> bool:
        return self.rank == self.columns

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "singular_values": list(self.singular_values),
            "threshold": self.threshold,
            "rank": self.rank,
            "columns": self.columns,
            "full_column_rank": self.full_column_rank,
            "condition_number": self.condition_number,
        }


def rank_diagnostics(matrix: np.ndarray, relative_tolerance: float = 1e-12) -> RankDiagnostics:
    array = np.asarray(matrix, dtype=np.float64)
    singular = np.linalg.svd(array, compute_uv=False)
    if singular.size == 0 or not np.all(np.isfinite(singular)):
        raise NumericalFailure("nonfinite or empty singular-value diagnostic")
    threshold = float(relative_tolerance * singular[0])
    rank = int(np.count_nonzero(singular > threshold))
    condition = math.inf if singular[-1] == 0.0 else float(singular[0] / singular[-1])
    return RankDiagnostics(
        shape=array.shape,
        singular_values=tuple(float(item) for item in singular),
        threshold=threshold,
        rank=rank,
        columns=array.shape[1],
        condition_number=condition,
    )


def _forward_substitution(lower: np.ndarray, right: np.ndarray) -> np.ndarray:
    factor = np.asarray(lower, dtype=np.float64)
    rhs = np.asarray(right, dtype=np.float64)
    vector_rhs = rhs.ndim == 1
    rhs_matrix = rhs.reshape(-1, 1) if vector_rhs else rhs
    if factor.ndim != 2 or factor.shape[0] != factor.shape[1]:
        raise NumericalFailure("forward substitution requires a square factor")
    if rhs_matrix.ndim != 2 or rhs_matrix.shape[0] != factor.shape[0]:
        raise NumericalFailure("forward substitution right-hand side shape mismatch")
    solution = np.empty_like(rhs_matrix, dtype=np.float64)
    for row in range(factor.shape[0]):
        pivot = factor[row, row]
        if not np.isfinite(pivot) or pivot <= 0.0:
            raise NumericalFailure("forward substitution has nonpositive/nonfinite pivot")
        residual = rhs_matrix[row] - factor[row, :row] @ solution[:row]
        solution[row] = residual / pivot
    return solution[:, 0] if vector_rhs else solution


def _back_substitution_upper(upper: np.ndarray, right: np.ndarray) -> np.ndarray:
    factor = np.asarray(upper, dtype=np.float64)
    rhs = np.asarray(right, dtype=np.float64)
    vector_rhs = rhs.ndim == 1
    rhs_matrix = rhs.reshape(-1, 1) if vector_rhs else rhs
    if factor.ndim != 2 or factor.shape[0] != factor.shape[1]:
        raise NumericalFailure("back substitution requires a square factor")
    if rhs_matrix.ndim != 2 or rhs_matrix.shape[0] != factor.shape[0]:
        raise NumericalFailure("back substitution right-hand side shape mismatch")
    solution = np.empty_like(rhs_matrix, dtype=np.float64)
    for row in range(factor.shape[0] - 1, -1, -1):
        pivot = factor[row, row]
        if not np.isfinite(pivot) or pivot <= 0.0:
            raise NumericalFailure("back substitution has nonpositive/nonfinite pivot")
        residual = rhs_matrix[row] - factor[row, row + 1 :] @ solution[row + 1 :]
        solution[row] = residual / pivot
    return solution[:, 0] if vector_rhs else solution


def _numerical_context(case_id: str | None, system_name: str | None) -> str:
    return (
        f"case_id={case_id or 'UNSPECIFIED'}, "
        f"system={system_name or 'UNSPECIFIED'}: "
    )


def cholesky_solve(
    matrix: np.ndarray,
    right: np.ndarray,
    *,
    case_id: str | None = None,
    system_name: str | None = None,
) -> np.ndarray:
    system = np.asarray(matrix, dtype=np.float64)
    rhs = np.asarray(right, dtype=np.float64)
    context = _numerical_context(case_id, system_name)
    if not np.all(np.isfinite(system)) or not np.all(np.isfinite(rhs)):
        raise NumericalFailure(f"{context}nonfinite Cholesky input")
    diagnostic = rank_diagnostics(system)
    if not diagnostic.full_column_rank:
        raise NumericalFailure(
            f"{context}Cholesky system failed frozen numerical-rank gate: "
            f"shape={diagnostic.shape}, rank={diagnostic.rank}, "
            f"columns={diagnostic.columns}, singular_values={diagnostic.singular_values}"
        )
    if diagnostic.condition_number > FROZEN_CONDITION_GATE:
        raise NumericalFailure(
            f"{context}Cholesky system exceeded frozen conditioning gate: "
            f"condition={diagnostic.condition_number:.17g}, gate={FROZEN_CONDITION_GATE:.17g}"
        )
    try:
        factor = np.linalg.cholesky(system)
        intermediate = _forward_substitution(factor, rhs)
        result = _back_substitution_upper(factor.T, intermediate)
    except np.linalg.LinAlgError as error:
        raise NumericalFailure(f"{context}Cholesky solve failed: {error}") from error
    if not np.all(np.isfinite(result)):
        raise NumericalFailure(f"{context}nonfinite Cholesky result")
    return np.asarray(result, dtype=np.float64)


def psd_square_root(
    matrix: np.ndarray,
    negative_relative_tolerance: float = 1e-12,
    diagnostics: dict[str, Any] | None = None,
    *,
    case_id: str | None = None,
    system_name: str | None = None,
) -> np.ndarray:
    original = np.asarray(matrix, dtype=np.float64)
    symmetric = np.asarray((original + original.T) / 2.0, dtype=np.float64)
    values, vectors = np.linalg.eigh(symmetric)
    scale = max(1.0, float(values[-1]))
    if float(values[0]) < -negative_relative_tolerance * scale:
        raise NumericalFailure(
            f"{_numerical_context(case_id, system_name or 'PSD square root')}"
            f"matrix is not PSD: lambda_min={values[0]!r}"
        )
    clipped = np.maximum(values, 0.0)
    if diagnostics is not None:
        diagnostics.update(
            {
                "input_antisymmetric_frobenius": float(
                    np.linalg.norm(original - original.T, ord="fro")
                ),
                "lambda_min_before_clip": float(values[0]),
                "lambda_max": float(values[-1]),
                "negative_clip_count": int(np.count_nonzero(values < 0.0)),
                "negative_relative_tolerance": float(negative_relative_tolerance),
            }
        )
    return np.asarray((vectors * np.sqrt(clipped)) @ vectors.T, dtype=np.float64)


def theorem1_formula(
    sigma0: np.ndarray,
    a: float,
    b: float,
    target: np.ndarray,
    measurement: np.ndarray,
    *,
    case_id: str | None = None,
) -> dict[str, Any]:
    d = sigma0.shape[0]
    sigma_t = np.float64(a) ** 2 * sigma0 + np.float64(b) ** 2 * np.eye(d)
    full_solve = cholesky_solve(
        sigma_t, target, case_id=case_id, system_name="formula/Sigma_t"
    )
    gram = measurement.T @ sigma_t @ measurement
    measurement_solve = cholesky_solve(
        gram,
        measurement.T @ target,
        case_id=case_id,
        system_name="formula/M^T Sigma_t M",
    )
    full_reduction = np.float64(b) ** 2 * np.trace(target.T @ full_solve)
    measurement_reduction = np.float64(b) ** 2 * np.trace(
        target.T @ measurement @ measurement_solve
    )
    return {
        "gap": float(full_reduction - measurement_reduction),
        "full_reduction": float(full_reduction),
        "measurement_reduction": float(measurement_reduction),
        "sigma_t_condition": rank_diagnostics(sigma_t).condition_number,
        "measurement_gram_condition": rank_diagnostics(gram).condition_number,
    }


def direct_gaussian_conditioning(
    sigma0: np.ndarray,
    a: float,
    b: float,
    target: np.ndarray,
    measurement: np.ndarray,
    *,
    case_id: str | None = None,
) -> dict[str, Any]:
    d = sigma0.shape[0]
    m = target.shape[1]
    sigma_t = np.float64(a) ** 2 * sigma0 + np.float64(b) ** 2 * np.eye(d)
    covariance_yx = np.float64(b) * target.T
    conditional_full = np.eye(m) - covariance_yx @ cholesky_solve(
        sigma_t,
        covariance_yx.T,
        case_id=case_id,
        system_name="direct/Sigma_t",
    )
    covariance_yz = np.float64(b) * target.T @ measurement
    covariance_z = measurement.T @ sigma_t @ measurement
    conditional_measurement = np.eye(m) - covariance_yz @ cholesky_solve(
        covariance_z,
        covariance_yz.T,
        case_id=case_id,
        system_name="direct/M^T Sigma_t M",
    )
    risk_full = float(np.trace(conditional_full))
    risk_measurement = float(np.trace(conditional_measurement))
    return {
        "risk_full": risk_full,
        "risk_measurement": risk_measurement,
        "gap": risk_measurement - risk_full,
        "conditional_full": conditional_full,
        "conditional_measurement": conditional_measurement,
    }


def whitened_residual_gap(
    sigma0: np.ndarray,
    a: float,
    b: float,
    target: np.ndarray,
    measurement: np.ndarray,
    *,
    rank_relative: float = 1e-12,
    case_id: str | None = None,
) -> dict[str, Any]:
    d = sigma0.shape[0]
    sigma_t = np.float64(a) ** 2 * sigma0 + np.float64(b) ** 2 * np.eye(d)
    symmetric = np.asarray((sigma_t + sigma_t.T) / 2.0, dtype=np.float64)
    values, vectors = np.linalg.eigh(symmetric)
    if float(values[0]) <= 0.0:
        raise NumericalFailure(
            f"{_numerical_context(case_id, 'whitened/Sigma_t')}"
            "whitened residual requires Sigma_t SPD"
        )
    square_root = (vectors * np.sqrt(values)) @ vectors.T
    inverse_square_root_target = (vectors * (1.0 / np.sqrt(values))) @ vectors.T @ target
    whitened_measurement = square_root @ measurement
    diagnostics = rank_diagnostics(whitened_measurement, rank_relative)
    if not diagnostics.full_column_rank:
        raise NumericalFailure(
            f"{_numerical_context(case_id, 'whitened/Sigma_t^(1/2) M')}"
            "whitened measurement lost full column rank"
        )
    q, _ = np.linalg.qr(whitened_measurement, mode="reduced")
    residual = inverse_square_root_target - q @ (q.T @ inverse_square_root_target)
    gap = np.float64(b) ** 2 * np.sum(residual * residual, dtype=np.float64)
    return {
        "gap": float(gap),
        "residual_frobenius": float(np.linalg.norm(residual, ord="fro")),
        "rank": diagnostics.to_dict(),
    }


def subspace_parent_gap(
    sigma0: np.ndarray,
    a: float,
    b: float,
    retained: np.ndarray,
    complement: np.ndarray,
    target: np.ndarray,
    *,
    case_id: str | None = None,
) -> dict[str, Any]:
    sigma_ss = retained.T @ sigma0 @ retained
    sigma_s_perp = retained.T @ sigma0 @ complement
    sigma_perp_perp = complement.T @ sigma0 @ complement
    v_s = np.float64(a) ** 2 * sigma_ss + np.float64(b) ** 2 * np.eye(retained.shape[1])
    v_perp = (
        np.float64(a) ** 2 * sigma_perp_perp
        + np.float64(b) ** 2 * np.eye(complement.shape[1])
    )
    v_s_inv_cross = cholesky_solve(
        v_s,
        sigma_s_perp,
        case_id=case_id,
        system_name="parent/V_S",
    )
    schur = v_perp - np.float64(a) ** 4 * sigma_s_perp.T @ v_s_inv_cross
    cross = (
        np.float64(b) * target.T @ complement
        - np.float64(b)
        * np.float64(a) ** 2
        * target.T
        @ retained
        @ v_s_inv_cross
    )
    solved = cholesky_solve(
        schur,
        cross.T,
        case_id=case_id,
        system_name="parent/Schur_complement",
    )
    gap = float(np.trace(cross @ solved))
    return {
        "gap": gap,
        "conditional_cross": cross,
        "schur": schur,
        "clean_cross": sigma_s_perp,
    }


@dataclass
class RunningMoments:
    count: int = 0
    total: np.float64 = np.float64(0.0)
    total_square: np.float64 = np.float64(0.0)

    def update(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        self.count += int(array.size)
        self.total = np.float64(self.total + np.sum(array, dtype=np.float64))
        self.total_square = np.float64(
            self.total_square + np.sum(array * array, dtype=np.float64)
        )

    def summary(self) -> dict[str, float | int]:
        if self.count < 2:
            raise NumericalFailure("at least two samples are required for MC variance")
        mean = float(self.total / np.float64(self.count))
        centered = float(
            (self.total_square - self.total * self.total / np.float64(self.count))
            / np.float64(self.count - 1)
        )
        variance = max(0.0, centered)
        standard_error = math.sqrt(variance / self.count)
        return {
            "count": self.count,
            "mean": mean,
            "unbiased_variance": variance,
            "standard_error": standard_error,
        }


def assess_mc(
    summaries: Mapping[str, Mapping[str, float | int]],
    exact: Mapping[str, float],
    mc_config: Mapping[str, Any],
) -> dict[str, Any]:
    sigma_multiplier = float(mc_config["sigma_multiplier"])
    floor = float(mc_config["consistency_floor"])
    checks: dict[str, Any] = {}
    all_consistent = True
    all_precise = True
    for key in ("risk_measurement", "risk_full", "gap"):
        summary = summaries[key]
        estimate = float(summary["mean"])
        se = float(summary["standard_error"])
        reference = float(exact[key])
        consistent = abs(estimate - reference) <= sigma_multiplier * se + floor
        if key == "gap":
            half_width_limit = max(
                float(mc_config["gap_half_width_abs"]),
                float(mc_config["gap_half_width_rel"])
                * max(float(mc_config["gap_half_width_scale"]), reference),
            )
        else:
            half_width_limit = max(
                float(mc_config["risk_half_width_abs"]),
                float(mc_config["risk_half_width_rel"]) * max(1.0, reference),
            )
        half_width = sigma_multiplier * se
        precise = half_width <= half_width_limit
        checks[key] = {
            "exact": reference,
            "estimate": estimate,
            "absolute_error": abs(estimate - reference),
            "five_se_half_width": half_width,
            "half_width_limit": half_width_limit,
            "consistent": consistent,
            "precision_sufficient": precise,
        }
        all_consistent = all_consistent and consistent
        all_precise = all_precise and precise
    return {
        "checks": checks,
        "all_consistent": all_consistent,
        "all_precision_sufficient": all_precise,
    }


def gaussian_monte_carlo(
    sigma0: np.ndarray,
    a: float,
    b: float,
    target: np.ndarray,
    measurement: np.ndarray,
    *,
    root_seed: int,
    case_id: str,
    mc_config: Mapping[str, Any],
    sample_checkpoints: Sequence[int] | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    rng, seed = make_rng(root_seed, case_id, "mc-samples")
    checkpoints = tuple(
        int(item)
        for item in (sample_checkpoints or mc_config["sample_checkpoints"])
    )
    batch = int(batch_size or mc_config["batch_size"])
    if sorted(checkpoints) != list(checkpoints) or any(item % batch for item in checkpoints):
        raise NumericalFailure("MC checkpoints must be increasing multiples of batch size")

    d = sigma0.shape[0]
    sigma_t = np.float64(a) ** 2 * sigma0 + np.float64(b) ** 2 * np.eye(d)
    beta_full = np.float64(b) * cholesky_solve(
        sigma_t,
        target,
        case_id=case_id,
        system_name="mc/Sigma_t",
    )
    gram = measurement.T @ sigma_t @ measurement
    beta_measurement = np.float64(b) * cholesky_solve(
        gram,
        measurement.T @ target,
        case_id=case_id,
        system_name="mc/M^T Sigma_t M",
    )
    square_root_diagnostics: dict[str, Any] = {}
    sigma0_root = psd_square_root(
        sigma0,
        diagnostics=square_root_diagnostics,
        case_id=case_id,
        system_name="mc/clean Sigma_0 PSD square root",
    )
    direct = direct_gaussian_conditioning(
        sigma0, a, b, target, measurement, case_id=case_id
    )
    exact = {
        "risk_measurement": float(direct["risk_measurement"]),
        "risk_full": float(direct["risk_full"]),
        "gap": float(direct["gap"]),
    }

    moments = {
        "risk_measurement": RunningMoments(),
        "risk_full": RunningMoments(),
        "gap": RunningMoments(),
    }
    history: list[dict[str, Any]] = []
    generated = 0
    for checkpoint in checkpoints:
        while generated < checkpoint:
            count = min(batch, checkpoint - generated)
            clean = rng.standard_normal((count, d)) @ sigma0_root.T
            noise = rng.standard_normal((count, d))
            x_t = np.float64(a) * clean + np.float64(b) * noise
            y = noise @ target
            pred_full = x_t @ beta_full
            pred_measurement = (x_t @ measurement) @ beta_measurement
            loss_full = np.sum((y - pred_full) ** 2, axis=1, dtype=np.float64)
            loss_measurement = np.sum(
                (y - pred_measurement) ** 2, axis=1, dtype=np.float64
            )
            moments["risk_full"].update(loss_full)
            moments["risk_measurement"].update(loss_measurement)
            moments["gap"].update(loss_measurement - loss_full)
            generated += count
        summaries = {key: value.summary() for key, value in moments.items()}
        assessment = assess_mc(summaries, exact, mc_config)
        history.append(
            {"checkpoint": checkpoint, "summaries": summaries, "assessment": assessment}
        )
        if assessment["all_precision_sufficient"]:
            status = "PASS" if assessment["all_consistent"] else "FAIL"
            return {
                "status": status,
                "root_seed": int(root_seed),
                "derived_seed128": seed,
                "case_id": case_id,
                "stream_tag": "mc-samples",
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
        "root_seed": int(root_seed),
        "derived_seed128": seed,
        "case_id": case_id,
        "stream_tag": "mc-samples",
        "psd_square_root_diagnostics": square_root_diagnostics,
        "history": history,
        "final": history[-1],
    }


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="approved T0 JSON configuration")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run only non-formal toy fixtures; never creates results/<run-id>",
    )
    return parser


def overall_status(case_statuses: Iterable[str]) -> str:
    statuses = tuple(case_statuses)
    if not statuses:
        return "FAIL"
    if any(item == "FAIL" for item in statuses):
        return "FAIL"
    if any(item in {"INCONCLUSIVE", "INCONCLUSIVE_PRECISION"} for item in statuses):
        return "INCONCLUSIVE"
    expected_nonfail = {
        "PASS",
        "EXPECTED_OUT_OF_DOMAIN",
        "EXPECTED_ILL_CONDITIONED",
    }
    return "PASS" if all(item in expected_nonfail for item in statuses) else "FAIL"
