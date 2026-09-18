"""Evaluate all six selected E2-v2 checkpoints on one paired evaluation bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torchvision.datasets import CIFAR10

from model import (
    CONDITIONS,
    PROJECT_ROOT,
    adapter_u_for_condition,
    configure_torch_runtime,
    images_to_float_tensor,
    load_e2v2_config,
    load_schedule_tensors,
    model_from_checkpoint,
    normalized_haar_lowpass,
    project_path,
    save_json,
    sha256_file,
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config = load_e2v2_config(PROJECT_ROOT)
    configure_torch_runtime(config)
    device = torch.device(args.device or str(config["runtime"]["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the approved full evaluation command")

    selection_path = project_path(PROJECT_ROOT, config["runs_root"]) / "selected_checkpoints.json"
    selection = load_json(selection_path)
    if selection.get("status") != "PASS" or selection.get("all_six_conditions_complete") is not True:
        raise RuntimeError("E2-v2 evaluation requires a complete six-condition validation selection")
    if set(selection.get("conditions", {})) != set(CONDITIONS):
        raise RuntimeError("selected checkpoint record does not contain exactly six conditions")

    output_root = project_path(PROJECT_ROOT, config["evaluation_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    raw_path = output_root / "e2_v2_evaluation_raw.npy"
    metadata_path = output_root / "E2V2_EVALUATION_METADATA.json"
    if raw_path.exists() or metadata_path.exists():
        raise RuntimeError("E2-v2 evaluation output already exists; refusing a second evaluation")

    with np.load(project_path(PROJECT_ROOT, config["split_file"]), allow_pickle=False) as split_payload:
        evaluation_indices = np.asarray(split_payload["evaluation_indices"], dtype=np.int64)
        unused_train_indices = np.asarray(split_payload["unused_train_indices"], dtype=np.int64)
    evaluation_source = str(config["data"]["evaluation_source"])
    if evaluation_source == "cifar10_train_holdout":
        if not np.array_equal(evaluation_indices, unused_train_indices):
            raise RuntimeError("E2-v2 evaluation is not the frozen unused training-corpus holdout")
        dataset = CIFAR10(root=str(project_path(PROJECT_ROOT, config["dataset_root"])), train=True, download=False)
        evaluation_images = np.asarray(dataset.data, dtype=np.uint8)
    else:
        raise RuntimeError(f"unsupported E2-v2 evaluation source: {evaluation_source}")
    if evaluation_indices.size != int(config["evaluation"]["images"]):
        raise RuntimeError("E2-v2 evaluation split has the wrong size")

    bank_path = project_path(PROJECT_ROOT, config["evaluation_bank_file"])
    noise_path = project_path(PROJECT_ROOT, config["evaluation_noise_file"])
    with np.load(bank_path, allow_pickle=False) as bank:
        bank_indices = np.asarray(bank["official_indices"], dtype=np.int64)
        true_times = np.asarray(bank["timesteps"], dtype=np.int64)
        null_times = np.asarray(bank["null_timesteps"], dtype=np.int64)
    if not np.array_equal(bank_indices, evaluation_indices):
        raise RuntimeError("evaluation bank indices differ from the frozen E2-v2 split")
    expected_shape = (
        int(config["evaluation"]["bins"]),
        int(config["evaluation"]["images"]),
        int(config["evaluation"]["draws_per_image_bin"]),
    )
    if true_times.shape != expected_shape or null_times.shape != expected_shape:
        raise RuntimeError("evaluation timestep bank shape mismatch")
    noise_bank = np.load(noise_path, mmap_mode="r")
    if noise_bank.shape != (*expected_shape, 3, 32, 32) or noise_bank.dtype != np.float32:
        raise RuntimeError("evaluation noise bank shape/dtype mismatch")

    alpha_bar, u_values = load_schedule_tensors(config, device)
    raw = np.empty((len(CONDITIONS), *expected_shape), dtype=np.float32)
    checkpoint_records: dict[str, Any] = {}
    batch_size = int(config["evaluation"]["batch_size"])
    for condition_id, condition in enumerate(CONDITIONS):
        selected = selection["conditions"][condition]
        checkpoint_path = project_path(PROJECT_ROOT, selected["selected_checkpoint"])
        if sha256_file(checkpoint_path) != selected["selected_checkpoint_sha256"]:
            raise RuntimeError(f"selected checkpoint hash mismatch for {condition}")
        model = model_from_checkpoint(condition, config, checkpoint_path, device)
        model.eval()
        for bin_id in range(expected_shape[0]):
            for start in range(0, expected_shape[1], batch_size):
                stop = min(start + batch_size, expected_shape[1])
                x0 = images_to_float_tensor(evaluation_images[evaluation_indices[start:stop]], device)
                for draw in range(expected_shape[2]):
                    t = torch.from_numpy(true_times[bin_id, start:stop, draw]).to(device=device, dtype=torch.long)
                    null_t = torch.from_numpy(null_times[bin_id, start:stop, draw]).to(device=device, dtype=torch.long)
                    noise = torch.from_numpy(
                        np.asarray(noise_bank[bin_id, start:stop, draw], dtype=np.float32)
                    ).to(device=device)
                    alpha = alpha_bar[t].reshape(-1, 1, 1, 1)
                    x_t = torch.sqrt(alpha) * x0 + torch.sqrt(1.0 - alpha) * noise
                    target = normalized_haar_lowpass(noise)
                    prediction = model(
                        x_t,
                        t,
                        u_values[t],
                        adapter_u_for_condition(condition, t, null_t, u_values),
                    )
                    errors = (prediction - target).square().mean(dim=(1, 2, 3))
                    raw[condition_id, bin_id, start:stop, draw] = errors.detach().cpu().numpy()
            print(f"condition={condition} evaluation_bin={bin_id + 1}/{expected_shape[0]}", flush=True)
        checkpoint_records[condition] = {
            "selected_step": int(selected["selected_step"]),
            "checkpoint": selected["selected_checkpoint"],
            "checkpoint_sha256": selected["selected_checkpoint_sha256"],
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not np.isfinite(raw).all():
        raise FloatingPointError("E2-v2 evaluation produced a nonfinite raw value")
    np.save(raw_path, raw)
    metadata = {
        "schema_version": 1,
        "status": "PASS",
        "protocol_id": config["protocol_id"],
        "evaluation_images_loaded_after_all_training_and_selection": True,
        "evaluation_source": evaluation_source,
        "conditions": list(CONDITIONS),
        "raw_shape": list(raw.shape),
        "raw_dtype": str(raw.dtype),
        "raw_sha256": sha256_file(raw_path),
        "split_sha256": sha256_file(project_path(PROJECT_ROOT, config["split_file"])),
        "evaluation_bank_sha256": sha256_file(bank_path),
        "evaluation_noise_sha256": sha256_file(noise_path),
        "paired_true_timesteps_noise_across_conditions": True,
        "time_pre_null_uses_independent_null_timesteps": True,
        "checkpoint_records": checkpoint_records,
    }
    save_json(metadata_path, metadata)
    print(f"E2V2_EVALUATION_PASS raw={raw_path}")


if __name__ == "__main__":
    main()
