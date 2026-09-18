"""Train one fresh E2-v2 condition with the common paired protocol."""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torchvision.datasets import CIFAR10

from model import (
    CONDITIONS,
    EXPECTED_ADAPTER_PARAMETERS,
    E2V2Model,
    E2V2TrainingStreams,
    PROJECT_ROOT,
    adapter_u_for_condition,
    build_schedule,
    configure_torch_runtime,
    images_to_float_tensor,
    learning_rate,
    load_e2v2_config,
    load_schedule_tensors,
    normalized_haar_lowpass,
    project_path,
    save_json,
    schedule_knots,
    sha256_file,
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_prepared(
    config: dict[str, Any],
    project_root: Path,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset = CIFAR10(root=str(project_path(project_root, config["dataset_root"])), train=True, download=False)
    images = np.asarray(dataset.data, dtype=np.uint8)
    if images.shape != (50_000, 32, 32, 3):
        raise RuntimeError(f"unexpected CIFAR-10 train image shape {images.shape}")
    with np.load(project_path(project_root, config["split_file"]), allow_pickle=False) as split_payload:
        splits = {key: np.asarray(split_payload[key], dtype=np.int64) for key in split_payload.files}
    expected = {"train_indices", "validation_indices", "unused_train_indices", "evaluation_indices"}
    if set(splits) != expected:
        raise RuntimeError(f"E2-v2 split fields differ: {set(splits)}")
    if str(config["data"]["evaluation_source"]) != "cifar10_train_holdout":
        raise RuntimeError("E2-v2 training expects the approved training-corpus holdout source")
    if not np.array_equal(splits["evaluation_indices"], splits["unused_train_indices"]):
        raise RuntimeError("E2-v2 training/evaluation split is not the frozen unused training holdout")
    for left_name, right_name in (
        ("train_indices", "evaluation_indices"),
        ("validation_indices", "evaluation_indices"),
    ):
        if np.intersect1d(splits[left_name], splits[right_name]).size:
            raise RuntimeError(f"E2-v2 reserved evaluation overlap: {left_name}/{right_name}")
    with np.load(project_path(project_root, config["validation_bank_file"]), allow_pickle=False) as bank:
        validation_times = np.asarray(bank["timesteps"], dtype=np.int64)
        validation_null_times = np.asarray(bank["null_timesteps"], dtype=np.int64)
        validation_indices = np.asarray(bank["official_indices"], dtype=np.int64)
        if not np.array_equal(validation_indices, splits["validation_indices"]):
            raise RuntimeError("validation bank indices differ from E2-v2 validation split")
        if validation_times.shape != (20, 5_000, 2) or validation_null_times.shape != validation_times.shape:
            raise RuntimeError("E2-v2 validation time-bank shape mismatch")
    validation_noise = np.load(
        project_path(project_root, config["validation_noise_file"]),
        mmap_mode="r",
    )
    if validation_noise.shape != (20, 5_000, 2, 3, 32, 32) or validation_noise.dtype != np.float32:
        raise RuntimeError("E2-v2 validation noise shape/dtype mismatch")
    return images, splits, validation_times, validation_null_times, validation_indices, validation_noise


@torch.no_grad()
def validate(
    model: E2V2Model,
    images: np.ndarray,
    indices: np.ndarray,
    true_times: np.ndarray,
    null_times: np.ndarray,
    noise_bank: np.ndarray,
    alpha_bar: torch.Tensor,
    u_values: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    bins, image_count, draws = true_times.shape
    per_image_draw = np.empty((bins, image_count, draws), dtype=np.float32)
    for bin_id in range(bins):
        for start in range(0, image_count, batch_size):
            stop = min(start + batch_size, image_count)
            batch_indices = indices[start:stop]
            x0 = images_to_float_tensor(images[batch_indices], device)
            for draw in range(draws):
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
                    adapter_u_for_condition(model.condition, t, null_t, u_values),
                )
                per_image_draw[bin_id, start:stop, draw] = (
                    prediction - target
                ).square().mean(dim=(1, 2, 3)).detach().cpu().numpy()
    bin_mse = per_image_draw.mean(axis=(1, 2), dtype=np.float64)
    aggregate = float(bin_mse.mean(dtype=np.float64))
    model.train()
    return per_image_draw, bin_mse, aggregate


def cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def save_training_state(
    path: Path,
    model: E2V2Model,
    optimizer: torch.optim.Optimizer,
    streams: E2V2TrainingStreams,
    step: int,
    validation_history: list[dict[str, Any]],
    condition: str,
    config: dict[str, Any],
) -> None:
    torch.save(
        {
            "protocol_id": config["protocol_id"],
            "condition": condition,
            "step": int(step),
            "model_state": cpu_state_dict(model),
            "optimizer_state": optimizer.state_dict(),
            "stream_state": streams.state_dict(),
            "validation_history": validation_history,
        },
        path,
    )


def prepare_model_and_optimizer(
    condition: str,
    config: dict[str, Any],
    project_root: Path,
    device: torch.device,
) -> tuple[E2V2Model, torch.optim.Optimizer]:
    model = E2V2Model(condition, config, schedule_knots(config))
    expected = int(config["adapter_parameter_counts"][condition])
    if model.adapter_parameter_count() != expected or expected != EXPECTED_ADAPTER_PARAMETERS[condition]:
        raise RuntimeError(f"{condition}: adapter parameter count mismatch")
    init_path = project_path(project_root, config["common_predictor_init"])
    init_payload = torch.load(init_path, map_location="cpu", weights_only=True)
    model.predictor.load_state_dict(init_payload["predictor_state"], strict=True)
    model.to(device)
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["peak_learning_rate"]),
        betas=(float(training["beta1"]), float(training["beta2"])),
        eps=float(training["epsilon"]),
        weight_decay=float(training["weight_decay"]),
    )
    return model, optimizer


def run_condition(
    condition: str,
    config: dict[str, Any],
    project_root: Path,
    images: np.ndarray,
    splits: dict[str, np.ndarray],
    validation_times: np.ndarray,
    validation_null_times: np.ndarray,
    validation_indices: np.ndarray,
    validation_noise: np.ndarray,
    device: torch.device,
    resume: bool,
) -> None:
    if condition not in CONDITIONS:
        raise ValueError(condition)
    run_root = project_path(project_root, config["runs_root"]) / condition
    run_root.mkdir(parents=True, exist_ok=True)
    state_path = run_root / "training_state.pt"
    if not resume and any(run_root.iterdir()):
        raise RuntimeError(f"{condition}: run directory is nonempty; use --resume only for an interrupted run")

    model, optimizer = prepare_model_and_optimizer(condition, config, project_root, device)
    streams = E2V2TrainingStreams(splits["train_indices"], config)
    start_step = 0
    validation_history: list[dict[str, Any]] = []
    if resume:
        if not state_path.is_file():
            raise RuntimeError(f"{condition}: --resume requested but training_state.pt is absent")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("protocol_id") != config["protocol_id"] or state.get("condition") != condition:
            raise RuntimeError(f"{condition}: incompatible training state")
        model.load_state_dict(state["model_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        streams.load_state_dict(state["stream_state"])
        start_step = int(state["step"])
        validation_history = list(state["validation_history"])
        if start_step % int(config["training"]["validation_every"]) != 0:
            raise RuntimeError(f"{condition}: resume step is not a validation boundary")

    effective = {
        "protocol_id": config["protocol_id"],
        "condition": condition,
        "adapter_parameters": model.adapter_parameter_count(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "training": config["training"],
        "validation": config["validation"],
        "rng_namespaces": config["rng_namespaces"],
        "paired_condition_inputs": "true t/noise shared; null t generated from independent namespace and used only by time_pre_null adapter",
        "common_predictor_init": config["common_predictor_init"],
        "common_predictor_init_sha256": sha256_file(project_path(project_root, config["common_predictor_init"])),
    }
    with (run_root / "effective_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(effective, handle, sort_keys=False)
    save_json(
        run_root / "run_metadata.json",
        {
            "protocol_id": config["protocol_id"],
            "condition": condition,
            "adapter_parameters": model.adapter_parameter_count(),
            "fresh_from_common_predictor_initialization": True,
            "train_images": int(splits["train_indices"].size),
            "validation_images": int(validation_indices.size),
            "validation_bank": config["validation_bank_file"],
            "started_utc": datetime.now(timezone.utc).isoformat(),
        },
    )

    alpha_bar, u_values = load_schedule_tensors(config, device)
    training = config["training"]
    total_steps = int(training["steps"])
    validation_every = int(training["validation_every"])
    batch_size = int(training["batch_size"])
    validation_batch = int(config["validation"]["batch_size"])
    log_every = int(training["log_every"])
    log_path = run_root / "training_history.jsonl"
    health_path = run_root / "numerical_health.json"
    start_time = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    mode = "a" if start_step else "w"
    with log_path.open(mode, encoding="utf-8", buffering=1) as log_handle:
        for step in range(start_step + 1, total_steps + 1):
            batch = streams.next(batch_size)
            x0 = images_to_float_tensor(images[batch.indices], device)
            t = torch.from_numpy(batch.timesteps).to(device=device, dtype=torch.long)
            null_t = torch.from_numpy(batch.null_timesteps).to(device=device, dtype=torch.long)
            noise = batch.noise.to(device=device)
            alpha = alpha_bar[t].reshape(-1, 1, 1, 1)
            x_t = torch.sqrt(alpha) * x0 + torch.sqrt(1.0 - alpha) * noise
            target = normalized_haar_lowpass(noise)
            rate = learning_rate(step, total_steps, config)
            for group in optimizer.param_groups:
                group["lr"] = rate
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                x_t,
                t,
                u_values[t],
                adapter_u_for_condition(condition, t, null_t, u_values),
            )
            loss = (prediction - target).square().mean()
            if not torch.isfinite(loss):
                save_json(health_path, {"status": "FAIL", "condition": condition, "step": step, "reason": "nonfinite_loss"})
                raise FloatingPointError(f"{condition}: nonfinite loss at step {step}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
            if not torch.isfinite(gradient_norm):
                save_json(health_path, {"status": "FAIL", "condition": condition, "step": step, "reason": "nonfinite_gradient"})
                raise FloatingPointError(f"{condition}: nonfinite gradient at step {step}")
            optimizer.step()
            if step == 1 or step % log_every == 0:
                log_handle.write(
                    json.dumps(
                        {
                            "step": step,
                            "loss": float(loss.detach().cpu()),
                            "gradient_norm": float(gradient_norm.detach().cpu()),
                            "learning_rate": rate,
                            "elapsed_seconds": time.time() - start_time,
                        },
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
            if step % validation_every == 0:
                per_image_draw, bin_mse, aggregate = validate(
                    model,
                    images,
                    validation_indices,
                    validation_times,
                    validation_null_times,
                    validation_noise,
                    alpha_bar,
                    u_values,
                    device,
                    validation_batch,
                )
                raw_path = run_root / f"validation_step_{step:06d}.npz"
                np.savez(
                    raw_path,
                    per_image_draw_mse=per_image_draw,
                    bin_mse=bin_mse,
                    equal_bin_mse=np.asarray(aggregate, dtype=np.float64),
                    official_indices=validation_indices,
                    timesteps=validation_times,
                    null_timesteps=validation_null_times,
                    step=np.asarray(step, dtype=np.int64),
                )
                checkpoint_path = run_root / f"checkpoint_step_{step:06d}.pt"
                torch.save(
                    {
                        "protocol_id": config["protocol_id"],
                        "condition": condition,
                        "step": int(step),
                        "adapter_parameters": model.adapter_parameter_count(),
                        "model_state": cpu_state_dict(model),
                        "fresh_from_common_predictor_initialization": True,
                    },
                    checkpoint_path,
                )
                record = {
                    "step": int(step),
                    "equal_bin_mse": float(aggregate),
                    "bin_mse": [float(value) for value in bin_mse],
                    "raw_statistics": raw_path.name,
                    "checkpoint": checkpoint_path.name,
                }
                validation_history.append(record)
                save_json(run_root / "validation_history.json", validation_history)
                save_training_state(
                    state_path,
                    model,
                    optimizer,
                    streams,
                    step,
                    validation_history,
                    condition,
                    config,
                )
                print(f"condition={condition} step={step} validation={aggregate:.12f}", flush=True)
                cooldown = int(config["runtime"]["cooldown_after_validation_seconds"])
                if cooldown > 0:
                    time.sleep(cooldown)

    expected_steps = list(range(validation_every, total_steps + 1, validation_every))
    if [int(record["step"]) for record in validation_history] != expected_steps:
        raise RuntimeError(f"{condition}: validation history is incomplete or irregular")
    save_json(
        health_path,
        {
            "status": "PASS",
            "condition": condition,
            "completed_steps": total_steps,
            "validation_events": len(validation_history),
            "elapsed_seconds_this_invocation": time.time() - start_time,
            "max_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            "nonfinite_loss": False,
            "nonfinite_gradient": False,
        },
    )
    print(f"condition={condition} COMPLETE steps={total_steps}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=(*CONDITIONS, "all"), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default=None, help="runtime device; defaults to the protocol config")
    args = parser.parse_args()
    config = load_e2v2_config(PROJECT_ROOT)
    if tuple(config["conditions"]) != CONDITIONS:
        raise RuntimeError("E2-v2 condition set differs from the implementation")
    configure_torch_runtime(config)
    device_name = args.device or str(config["runtime"]["device"])
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full E2-v2 training")
    if device.type == "cuda" and torch.cuda.device_count() != 1:
        raise RuntimeError("E2-v2 training is defined for one visible GPU; use CUDA_VISIBLE_DEVICES to isolate one GPU")
    images, splits, validation_times, validation_null_times, validation_indices, validation_noise = load_prepared(config, PROJECT_ROOT)
    requested = CONDITIONS if args.condition == "all" else (args.condition,)
    for condition in requested:
        run_condition(
            condition,
            config,
            PROJECT_ROOT,
            images,
            splits,
            validation_times,
            validation_null_times,
            validation_indices,
            validation_noise,
            device,
            args.resume,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        cooldown = int(config["runtime"]["inter_condition_idle_cooldown_seconds"])
        if condition != requested[-1] and cooldown > 0:
            time.sleep(cooldown)


if __name__ == "__main__":
    main()
