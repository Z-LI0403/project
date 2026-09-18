"""Independently recompute validation aggregates and select E2-v2 checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from model import (
    CONDITIONS,
    PROJECT_ROOT,
    load_e2v2_config,
    project_path,
    save_json,
    sha256_file,
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def expected_steps(config: dict[str, Any]) -> list[int]:
    return list(
        range(
            int(config["training"]["validation_every"]),
            int(config["training"]["steps"]) + 1,
            int(config["training"]["validation_every"]),
        )
    )


def independent_record(
    run_root: Path,
    step: int,
    validation_record: dict[str, Any],
    expected_adapter_parameters: int,
    validation_indices: np.ndarray,
    validation_times: np.ndarray,
    validation_null_times: np.ndarray,
) -> dict[str, Any]:
    raw_path = run_root / str(validation_record["raw_statistics"])
    checkpoint_path = run_root / str(validation_record["checkpoint"])
    if not raw_path.is_file() or not checkpoint_path.is_file():
        raise RuntimeError(f"missing validation raw/checkpoint for step {step}")
    with np.load(raw_path, allow_pickle=False) as raw:
        if set(raw.files) != {
            "per_image_draw_mse",
            "bin_mse",
            "equal_bin_mse",
            "official_indices",
            "timesteps",
            "null_timesteps",
            "step",
        }:
            raise RuntimeError(f"unexpected raw validation fields at {raw_path}")
        values = np.asarray(raw["per_image_draw_mse"], dtype=np.float32)
        if values.shape != (20, 5_000, 2) or not np.isfinite(values).all():
            raise RuntimeError(f"invalid raw validation array at {raw_path}")
        if not np.array_equal(np.asarray(raw["official_indices"], dtype=np.int64), validation_indices):
            raise RuntimeError(f"validation indices differ at {raw_path}")
        if not np.array_equal(np.asarray(raw["timesteps"], dtype=np.int64), validation_times):
            raise RuntimeError(f"validation true-time bank differs at {raw_path}")
        if not np.array_equal(np.asarray(raw["null_timesteps"], dtype=np.int64), validation_null_times):
            raise RuntimeError(f"validation null-time bank differs at {raw_path}")
        if int(raw["step"]) != step:
            raise RuntimeError(f"raw validation step mismatch at {raw_path}")
        bin_mse = values.mean(axis=(1, 2), dtype=np.float64)
        aggregate = float(bin_mse.mean(dtype=np.float64))
        stored_bin = np.asarray(raw["bin_mse"], dtype=np.float64)
        stored_aggregate = float(raw["equal_bin_mse"])
    if not np.allclose(stored_bin, bin_mse, rtol=0.0, atol=1.0e-12):
        raise RuntimeError(f"stored validation bin aggregate mismatch at {raw_path}")
    if not np.isclose(stored_aggregate, aggregate, rtol=0.0, atol=1.0e-12):
        raise RuntimeError(f"stored validation aggregate mismatch at {raw_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("protocol_id") != validation_record.get("protocol_id", checkpoint.get("protocol_id"))
        or checkpoint.get("condition") != validation_record.get("condition")
        or checkpoint.get("step") != step
        or int(checkpoint.get("adapter_parameters", -1)) != expected_adapter_parameters
        or checkpoint.get("fresh_from_common_predictor_initialization") is not True
    ):
        raise RuntimeError(f"checkpoint configuration mismatch at {checkpoint_path}")
    return {
        "step": int(step),
        "equal_bin_mse": aggregate,
        "bin_mse": [float(value) for value in bin_mse],
        "raw_statistics": raw_path.name,
        "raw_statistics_sha256": sha256_file(raw_path),
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    config = load_e2v2_config(PROJECT_ROOT)
    if tuple(config["conditions"]) != CONDITIONS:
        raise RuntimeError("E2-v2 condition set differs from the implementation")
    split_path = project_path(PROJECT_ROOT, config["split_file"])
    bank_path = project_path(PROJECT_ROOT, config["validation_bank_file"])
    with np.load(split_path, allow_pickle=False) as split_payload:
        validation_indices = np.asarray(split_payload["validation_indices"], dtype=np.int64)
    with np.load(bank_path, allow_pickle=False) as bank:
        bank_indices = np.asarray(bank["official_indices"], dtype=np.int64)
        validation_times = np.asarray(bank["timesteps"], dtype=np.int64)
        validation_null_times = np.asarray(bank["null_timesteps"], dtype=np.int64)
    if not np.array_equal(bank_indices, validation_indices):
        raise RuntimeError("validation bank does not match the stored split")

    steps = expected_steps(config)
    selected: dict[str, Any] = {}
    for condition in CONDITIONS:
        run_root = project_path(PROJECT_ROOT, config["runs_root"]) / condition
        if not (run_root / "numerical_health.json").is_file():
            raise RuntimeError(f"{condition}: numerical-health record is missing")
        health = load_json(run_root / "numerical_health.json")
        if health.get("status") != "PASS" or int(health.get("completed_steps", -1)) != int(config["training"]["steps"]):
            raise RuntimeError(f"{condition}: training is not complete and healthy")
        history = load_json(run_root / "validation_history.json")
        if [int(record["step"]) for record in history] != steps:
            raise RuntimeError(f"{condition}: validation history is incomplete")
        recomputed = []
        for step, record in zip(steps, history):
            record = dict(record)
            record["protocol_id"] = config["protocol_id"]
            record["condition"] = condition
            recomputed.append(
                independent_record(
                    run_root,
                    step,
                    record,
                    int(config["adapter_parameter_counts"][condition]),
                    validation_indices,
                    validation_times,
                    validation_null_times,
                )
            )
        chosen = min(recomputed, key=lambda record: (record["equal_bin_mse"], record["step"]))
        selected[condition] = {
            "condition": condition,
            "adapter_parameters": int(config["adapter_parameter_counts"][condition]),
            "fresh_from_common_predictor_initialization": True,
            "selection_rule": "minimum full validation equal-bin MSE; earliest exact tie",
            "selected_step": int(chosen["step"]),
            "selected_equal_bin_mse": float(chosen["equal_bin_mse"]),
            "selected_checkpoint": str((run_root / chosen["checkpoint"]).relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "selected_checkpoint_sha256": chosen["checkpoint_sha256"],
            "selected_raw_statistics": str((run_root / chosen["raw_statistics"]).relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "selected_raw_statistics_sha256": chosen["raw_statistics_sha256"],
            "complete_validation_history": recomputed,
        }

    output = {
        "schema_version": 1,
        "status": "PASS",
        "protocol_id": config["protocol_id"],
        "all_six_conditions_complete": True,
        "evaluation_not_used_for_selection": True,
        "split_sha256": sha256_file(split_path),
        "validation_bank_sha256": sha256_file(bank_path),
        "conditions": selected,
    }
    output_path = project_path(PROJECT_ROOT, config["runs_root"]) / "selected_checkpoints.json"
    save_json(output_path, output)
    print(f"E2V2_SELECTION_PASS output={output_path}")
    for condition in CONDITIONS:
        print(
            f"{condition}: step={selected[condition]['selected_step']} "
            f"validation={selected[condition]['selected_equal_bin_mse']:.12f}"
        )


if __name__ == "__main__":
    main()
