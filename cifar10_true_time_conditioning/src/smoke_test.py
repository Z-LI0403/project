"""Compact local E2-v2 smoke checks; no training steps or evaluation inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from model import (
    CONDITIONS,
    EXPECTED_ADAPTER_PARAMETERS,
    E2V2Model,
    E2V2TrainingStreams,
    PROJECT_ROOT,
    adapter_u_for_condition,
    configure_torch_runtime,
    load_e2v2_config,
    load_schedule_tensors,
    normalized_haar_lowpass,
    project_path,
    save_json,
    schedule_knots,
)


def load_splits(config: dict[str, Any]) -> dict[str, np.ndarray]:
    with np.load(project_path(PROJECT_ROOT, config["split_file"]), allow_pickle=False) as payload:
        return {key: np.asarray(payload[key], dtype=np.int64) for key in payload.files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = load_e2v2_config(PROJECT_ROOT)
    configure_torch_runtime(config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA smoke device but CUDA is unavailable")

    if tuple(config["conditions"]) != CONDITIONS:
        raise RuntimeError("config condition set is not the exact six-condition E2-v2 set")
    counts: dict[str, int] = {}
    parameter_total: dict[str, int] = {}
    knots = schedule_knots(config)
    for condition in CONDITIONS:
        model = E2V2Model(condition, config, knots)
        counts[condition] = model.adapter_parameter_count()
        parameter_total[condition] = sum(parameter.numel() for parameter in model.parameters())
        if counts[condition] != EXPECTED_ADAPTER_PARAMETERS[condition]:
            raise RuntimeError(f"{condition}: adapter parameter count mismatch")
        if counts[condition] != int(config["adapter_parameter_counts"][condition]):
            raise RuntimeError(f"{condition}: config parameter count mismatch")

    time_pre = E2V2Model("time_pre", config, knots)
    time_null = E2V2Model("time_pre_null", config, knots)
    if set(time_pre.state_dict()) != set(time_null.state_dict()):
        raise RuntimeError("time_pre and time_pre_null do not share the same state structure")
    if parameter_total["time_pre"] != parameter_total["time_pre_null"]:
        raise RuntimeError("time_pre and time_pre_null total parameter counts differ")

    alpha_bar, u_values = load_schedule_tensors(config, device)
    x_t = torch.randn((2, 3, 32, 32), device=device)
    t = torch.tensor([17, 743], device=device, dtype=torch.long)
    null_t = torch.tensor([981, 201], device=device, dtype=torch.long)
    noise = torch.randn((2, 3, 32, 32), device=device)
    target = normalized_haar_lowpass(noise)
    finite_forward_backward: dict[str, bool] = {}
    for condition in CONDITIONS:
        model = E2V2Model(condition, config, knots).to(device)
        prediction = model(
            x_t,
            t,
            u_values[t],
            adapter_u_for_condition(condition, t, null_t, u_values),
        )
        if prediction.shape != target.shape or not torch.isfinite(prediction).all():
            raise RuntimeError(f"{condition}: invalid forward output shape/values")
        loss = (prediction - target).square().mean()
        loss.backward()
        gradients_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        if not torch.isfinite(loss) or not gradients_finite:
            raise RuntimeError(f"{condition}: nonfinite smoke loss/gradient")
        finite_forward_backward[condition] = True

    streams = E2V2TrainingStreams(np.arange(4096, dtype=np.int64), config)
    batch = streams.next(4096)
    if np.array_equal(batch.timesteps, batch.null_timesteps):
        raise RuntimeError("true and null training timestep streams are identical")
    correlation = float(np.corrcoef(batch.timesteps.astype(np.float64), batch.null_timesteps.astype(np.float64))[0, 1])
    if abs(correlation) >= 0.1:
        raise RuntimeError(f"true/null timestep smoke correlation is unexpectedly large: {correlation}")
    namespace_true = int(config["rng_namespaces"]["timestep_sampling"])
    namespace_null = int(config["rng_namespaces"]["null_conditioning_timestep"])
    if namespace_true == namespace_null:
        raise RuntimeError("true and null training RNG namespaces collide")

    splits = load_splits(config)
    expected_split_fields = {"train_indices", "validation_indices", "unused_train_indices", "evaluation_indices"}
    if set(splits) != expected_split_fields:
        raise RuntimeError("E2-v2 split fields are incomplete")
    for left_name, right_name in (
        ("train_indices", "validation_indices"),
        ("train_indices", "unused_train_indices"),
        ("validation_indices", "unused_train_indices"),
    ):
        if np.intersect1d(splits[left_name], splits[right_name]).size:
            raise RuntimeError(f"split overlap: {left_name}/{right_name}")
    if str(config["data"]["evaluation_source"]) != "cifar10_train_holdout":
        raise RuntimeError("E2-v2 smoke expects the approved training-corpus holdout source")
    if not np.array_equal(splits["evaluation_indices"], splits["unused_train_indices"]):
        raise RuntimeError("evaluation split is not exactly the unused training-corpus holdout")
    for left_name, right_name in (
        ("train_indices", "evaluation_indices"),
        ("validation_indices", "evaluation_indices"),
    ):
        if np.intersect1d(splits[left_name], splits[right_name]).size:
            raise RuntimeError(f"split overlap: {left_name}/{right_name}")

    with np.load(project_path(PROJECT_ROOT, config["evaluation_bank_file"]), allow_pickle=False) as bank:
        true_times = np.asarray(bank["timesteps"], dtype=np.int64)
        null_times = np.asarray(bank["null_timesteps"], dtype=np.int64)
        bank_indices = np.asarray(bank["official_indices"], dtype=np.int64)
    if not np.array_equal(bank_indices, splits["evaluation_indices"]):
        raise RuntimeError("evaluation bank indices do not match the split")
    if np.array_equal(true_times, null_times):
        raise RuntimeError("evaluation true/null timestep banks are identical")
    if np.any(null_times < 0) or np.any(null_times >= 1000):
        raise RuntimeError("null evaluation timestep outside the marginal timestep support")
    if np.any(true_times < 0) or np.any(true_times >= 1000):
        raise RuntimeError("true evaluation timestep outside the schedule")
    for bin_id in range(true_times.shape[0]):
        if np.any(true_times[bin_id] < bin_id * 50) or np.any(true_times[bin_id] >= (bin_id + 1) * 50):
            raise RuntimeError("true evaluation timestep escaped its pre-specified bin")
    eval_correlation = float(np.corrcoef(true_times.reshape(-1).astype(np.float64), null_times.reshape(-1).astype(np.float64))[0, 1])
    if abs(eval_correlation) >= 0.05:
        raise RuntimeError(f"true/null evaluation timestep smoke correlation is unexpectedly large: {eval_correlation}")

    report = {
        "status": "PASS",
        "protocol_id": config["protocol_id"],
        "condition_set": list(CONDITIONS),
        "adapter_parameter_counts": counts,
        "total_parameter_counts": parameter_total,
        "finite_forward_backward": finite_forward_backward,
        "true_null_training_namespace": [namespace_true, namespace_null],
        "true_null_training_sample_correlation": correlation,
        "true_null_evaluation_sample_correlation": eval_correlation,
        "split_sizes": {name: int(values.size) for name, values in splits.items()},
        "evaluation_source": config["data"]["evaluation_source"],
        "evaluation_equals_unused_train_holdout": True,
        "split_disjointness": True,
        "pre_post_interface_paths_checked": True,
        "shared_evaluation_bank": True,
        "heldout_images_loaded": False,
        "optimizer_steps": 0,
    }
    output = project_path(PROJECT_ROOT, "experiments/e2_v2_true_time_conditioning/runtime/smoke.json")
    save_json(output, report)
    print(f"E2V2_SMOKE_PASS report={output}")


if __name__ == "__main__":
    main()
