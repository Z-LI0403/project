"""Compute paired E2-v2 contrasts and image-cluster bootstrap intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from model import CONDITIONS, PROJECT_ROOT, load_e2v2_config, project_path, save_json, sha256_file


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def contrast_formula(name: str) -> str:
    formulas = {
        "D_time": "MSE(time_pre) - MSE(static_pre)",
        "D_pre_static": "MSE(static_pre) - MSE(static_post)",
        "D_pre_time": "MSE(time_pre) - MSE(time_post)",
        "D_true_time": "MSE(time_pre) - MSE(time_pre_null)",
    }
    return formulas[name]


def contrast_role(name: str) -> str:
    roles = {
        "D_true_time": "primary capacity-controlled true-time test",
        "D_time": "supporting bridge comparison",
        "D_pre_static": "secondary structural pre/post comparison",
        "D_pre_time": "secondary structural pre/post comparison",
    }
    return roles[name]


def direction(point: float, interval: tuple[float, float]) -> str:
    if interval[1] < 0.0:
        return "negative (theory-consistent direction)"
    if interval[0] > 0.0:
        return "positive"
    if point == 0.0:
        return "zero"
    return "interval crosses zero"


def markdown_report(
    config: dict[str, Any],
    aggregates: dict[str, float],
    profiles: dict[str, list[float]],
    contrasts: dict[str, dict[str, Any]],
    selected: dict[str, Any],
    mechanism: dict[str, Any] | None,
) -> str:
    lines = [
        "# E2-v2 Results",
        "",
        "**Status:** completed paired evaluation and bootstrap analysis.",
        "",
        "## Theoretical statement",
        "",
        "The final theory draft gives exact finite Gaussian/Haar results for interface deficiency, constrained pre-projection repair, time-conditioned strict gain, and pre/post information separation. E2-v2 tests whether these structural consequences appear in a controlled CIFAR-10 neural setting; it does not prove the exact theorem, reproduce its closed-form optimizer, or identify a unique causal mechanism.",
        "",
        "## Empirical observation",
        "",
        "Equal-bin held-out projected-epsilon MSE:",
        "",
        "| Condition | Equal-bin MSE | Selected step |",
        "|---|---:|---:|",
    ]
    for condition in CONDITIONS:
        lines.append(
            f"| `{condition}` | {aggregates[condition]:.12f} | {int(selected[condition]['selected_step'])} |"
        )
    lines.extend(
        [
            "",
            "Primary capacity-controlled true-time test:",
            "",
            "`D_true_time = MSE(time_pre) - MSE(time_pre_null)` compares equal-capacity, equal-training-budget adapters and is the primary test of true-time conditioning beyond the independently sampled null conditioning.",
            "",
            "| Contrast | Point estimate | 95% paired image-cluster interval | Direction | Predefined support rule |",
            "|---|---:|---:|---|---|",
        ]
    )
    for name in ("D_true_time",):
        record = contrasts[name]
        lo, hi = record["interval_95"]
        lines.append(
            f"| `{name}` | {record['point_estimate']:.12g} | [{lo:.12g}, {hi:.12g}] | "
            f"{record['direction']} | {record['paired_95_interval_entirely_below_zero']} |"
        )
    lines.extend(
        [
            "",
            "Supporting bridge comparison:",
            "",
            "`D_time = MSE(time_pre) - MSE(static_pre)` retains the original time-versus-static bridge comparison as supporting evidence; its adapter capacities are not matched.",
            "",
            "| Contrast | Point estimate | 95% paired image-cluster interval | Direction | Predefined support rule |",
            "|---|---:|---:|---|---|",
        ]
    )
    for name in ("D_time",):
        record = contrasts[name]
        lo, hi = record["interval_95"]
        lines.append(
            f"| `{name}` | {record['point_estimate']:.12g} | [{lo:.12g}, {hi:.12g}] | "
            f"{record['direction']} | {record['paired_95_interval_entirely_below_zero']} |"
        )
    lines.extend(
        [
            "",
            "Secondary structural pre/post contrasts:",
            "",
            "| Contrast | Point estimate | 95% paired image-cluster interval | Direction | Predefined support rule |",
            "|---|---:|---:|---|---|",
        ]
    )
    for name in ("D_pre_static", "D_pre_time"):
        record = contrasts[name]
        lo, hi = record["interval_95"]
        lines.append(
            f"| `{name}` | {record['point_estimate']:.12g} | [{lo:.12g}, {hi:.12g}] | "
            f"{record['direction']} | {record['paired_95_interval_entirely_below_zero']} |"
        )
    lines.extend(["", "Per-bin MSE profiles:", "", "| Bin | " + " | ".join(CONDITIONS) + " |", "|---:|" + "---:|" * len(CONDITIONS)])
    for bin_id in range(len(profiles[CONDITIONS[0]])):
        lines.append(
            f"| {bin_id:02d} | " + " | ".join(f"{profiles[c][bin_id]:.9f}" for c in CONDITIONS) + " |"
        )
    lines.extend(
        [
            "",
            "## Mechanism-consistent interpretation",
            "",
            "The operator readout is supporting evidence only. A nonzero pre-projection factorization residual means the learned pre operator uses directions outside the fixed block-constant coarse subspace; it does not by itself establish target-specific strict risk gain or a unique causal explanation.",
        ]
    )
    if mechanism is None:
        lines.append("Mechanism diagnostics have not yet been generated.")
    else:
        for condition, record in mechanism.get("conditions", {}).items():
            variation = record["time_variation"]
            residuals = [item["normalized_fine_residual"] for item in record["factorization"]]
            lines.append(
                f"- `{condition}`: mean reference-relative operator variation "
                f"{variation['mean_reference_relative_frobenius']:.6g}; "
                f"factorization residual range [{min(residuals):.6g}, {max(residuals):.6g}]."
            )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- One common training seed and one controlled 32-to-16 CIFAR-10 interface were used.",
            "- The evaluation subset is the 5,000-image holdout from the CIFAR-10 training corpus that is unused by this experiment and disjoint from training and validation.",
            "- Bootstrap intervals quantify paired image-cluster resampling uncertainty conditional on the trained checkpoints and fixed evaluation bank; they are not across-seed uncertainty and are not a proof.",
            "- The result does not claim that CIFAR-10 satisfies the finite Gaussian model, that the network learns the closed-form rho*(tau), or that operator diagnostics prove causality.",
            "",
            f"Experiment configuration: `{config['protocol_id']}`. Full machine-readable output: `runtime/evaluation/E2V2_ANALYSIS.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    config = load_e2v2_config(PROJECT_ROOT)
    evaluation_root = project_path(PROJECT_ROOT, config["evaluation_root"])
    raw_path = evaluation_root / "e2_v2_evaluation_raw.npy"
    metadata_path = evaluation_root / "E2V2_EVALUATION_METADATA.json"
    if not raw_path.is_file() or not metadata_path.is_file():
        raise RuntimeError("E2-v2 evaluation raw arrays and metadata are required")
    metadata = load_json(metadata_path)
    if metadata.get("status") != "PASS" or metadata.get("conditions") != list(CONDITIONS):
        raise RuntimeError("E2-v2 evaluation metadata is incomplete")
    if metadata.get("raw_sha256") != sha256_file(raw_path):
        raise RuntimeError("E2-v2 raw evaluation hash mismatch")
    raw = np.load(raw_path, mmap_mode="r")
    bins = int(config["evaluation"]["bins"])
    images = int(config["evaluation"]["images"])
    draws = int(config["evaluation"]["draws_per_image_bin"])
    if raw.shape != (len(CONDITIONS), bins, images, draws) or raw.dtype != np.float32:
        raise RuntimeError(f"unexpected E2-v2 raw evaluation shape/dtype: {raw.shape} {raw.dtype}")
    if not np.isfinite(raw).all():
        raise FloatingPointError("E2-v2 raw evaluation contains a nonfinite value")

    condition_id = {condition: index for index, condition in enumerate(CONDITIONS)}
    profiles_array = np.asarray(raw, dtype=np.float64).mean(axis=(2, 3), dtype=np.float64)
    aggregate_array = profiles_array.mean(axis=1, dtype=np.float64)
    profiles = {
        condition: [float(value) for value in profiles_array[condition_id[condition]]]
        for condition in CONDITIONS
    }
    aggregates = {condition: float(aggregate_array[condition_id[condition]]) for condition in CONDITIONS}

    per_image = np.asarray(raw, dtype=np.float64).mean(axis=3, dtype=np.float64)
    contrast_pairs = {
        "D_time": ("time_pre", "static_pre"),
        "D_pre_static": ("static_pre", "static_post"),
        "D_pre_time": ("time_pre", "time_post"),
        "D_true_time": ("time_pre", "time_pre_null"),
    }
    contrast_per_image = np.stack(
        [per_image[condition_id[left]] - per_image[condition_id[right]] for left, right in contrast_pairs.values()]
    )
    point = contrast_per_image.mean(axis=(1, 2), dtype=np.float64)

    replicates = int(config["analysis"]["bootstrap_replicates"])
    seed = int(config["analysis"]["bootstrap_seed"])
    bootstrap = np.empty((replicates, len(contrast_pairs)), dtype=np.float64)
    generator = np.random.Generator(np.random.PCG64(seed))
    batch_size = 16
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        indices = generator.integers(0, images, size=(stop - start, images), dtype=np.int64)
        sampled = np.take(contrast_per_image, indices, axis=2)
        # sampled shape: contrast x bin x bootstrap-replicate x image.
        bootstrap[start:stop] = sampled.mean(axis=3, dtype=np.float64).mean(axis=1, dtype=np.float64).T
        if (start // batch_size) % 50 == 0:
            print(f"bootstrap {stop}/{replicates}", flush=True)

    contrast_records: dict[str, Any] = {}
    for index, name in enumerate(contrast_pairs):
        interval = tuple(float(value) for value in np.quantile(
            bootstrap[:, index],
            [0.025, 0.975],
            method=str(config["analysis"]["quantile_method"]),
        ))
        contrast_records[name] = {
            "formula": contrast_formula(name),
            "role": contrast_role(name),
            "left_condition": contrast_pairs[name][0],
            "right_condition": contrast_pairs[name][1],
            "point_estimate": float(point[index]),
            "interval_95": [interval[0], interval[1]],
            "direction": direction(float(point[index]), interval),
            "paired_95_interval_entirely_below_zero": bool(interval[1] < 0.0),
            "bootstrap_replicates": replicates,
            "bootstrap_seed": seed,
            "bootstrap_unit": "image cluster; paired condition/bin/draw structure preserved",
        }

    selection = load_json(project_path(PROJECT_ROOT, config["runs_root"]) / "selected_checkpoints.json")
    mechanism_path = project_path(PROJECT_ROOT, config["mechanism_root"]) / "mechanism_diagnostics.json"
    if not mechanism_path.is_file():
        raise RuntimeError("run mechanism.py before final E2-v2 analysis")
    mechanism = load_json(mechanism_path)
    bootstrap_path = evaluation_root / "E2V2_BOOTSTRAP_CONTRASTS.npy"
    np.save(bootstrap_path, bootstrap)
    result = {
        "schema_version": 1,
        "status": "PASS",
        "protocol_id": config["protocol_id"],
        "theory_alignment": "structural neural analogues only; not a proof of Theorem 2 or Corollaries 2.1-2.2",
        "conditions": list(CONDITIONS),
        "selected_checkpoints": {
            condition: {
                "step": int(selection["conditions"][condition]["selected_step"]),
                "checkpoint": selection["conditions"][condition]["selected_checkpoint"],
            }
            for condition in CONDITIONS
        },
        "evaluation_raw": {
            "path": str(raw_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "sha256": sha256_file(raw_path),
            "shape": list(raw.shape),
            "dtype": str(raw.dtype),
        },
        "equal_bin_mse": aggregates,
        "per_bin_mse": profiles,
        "contrasts": contrast_records,
        "bootstrap_raw": {
            "path": str(bootstrap_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "sha256": sha256_file(bootstrap_path),
            "shape": list(bootstrap.shape),
        },
        "fixed_baseline_role": "descriptive_only",
        "mechanism_diagnostics": (
            str(mechanism_path.relative_to(PROJECT_ROOT)).replace("\\", "/") if mechanism is not None else None
        ),
    }
    result_path = evaluation_root / "E2V2_ANALYSIS.json"
    save_json(result_path, result)
    report_path = project_path(PROJECT_ROOT, "experiments/e2_v2_true_time_conditioning/results.md")
    report_path.write_text(
        markdown_report(config, aggregates, profiles, contrast_records, selection["conditions"], mechanism),
        encoding="utf-8",
    )
    print(f"E2V2_ANALYSIS_PASS result={result_path}")
    print(f"E2V2_RESULTS_REPORT={report_path}")


if __name__ == "__main__":
    main()
