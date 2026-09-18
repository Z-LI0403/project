"""Compute the theory-aligned operator diagnostics for selected pre adapters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from model import (
    PRE_CONDITIONS,
    PROJECT_ROOT,
    E2V2Model,
    circular_conv2d,
    configure_torch_runtime,
    load_e2v2_config,
    model_from_checkpoint,
    normalized_haar_lowpass,
    project_path,
    save_json,
    sha256_file,
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def grid_timesteps(config: dict[str, Any]) -> np.ndarray:
    bins = int(config["evaluation"]["bins"])
    width = int(config["evaluation"]["bin_width"])
    return np.arange(bins, dtype=np.int64) * width + width // 2


@torch.no_grad()
def effective_kernels(model: E2V2Model, timesteps: np.ndarray, device: torch.device) -> torch.Tensor:
    if model.adapter is None:
        raise RuntimeError(f"{model.condition} has no adapter")
    if hasattr(model.adapter, "kernels"):
        u = torch.from_numpy(timesteps).to(device=device, dtype=torch.float32)
        return model.adapter.kernels(u).detach()
    weight = model.adapter.weight.detach()
    return weight.unsqueeze(0).expand(timesteps.size, *weight.shape).clone()


def time_variation(kernels: torch.Tensor, timesteps: np.ndarray) -> dict[str, Any]:
    values = kernels.detach().cpu().numpy().astype(np.float64)
    count = values.shape[0]
    norms = np.linalg.norm(values.reshape(count, -1), axis=1)
    reference = values[0]
    reference_norm = max(float(np.linalg.norm(reference)), 1.0e-12)
    reference_relative = np.linalg.norm((values - reference).reshape(count, -1), axis=1) / reference_norm
    pairwise = np.empty((count, count), dtype=np.float64)
    for left in range(count):
        for right in range(count):
            denominator = max(float(norms[left]), float(norms[right]), 1.0e-12)
            pairwise[left, right] = float(np.linalg.norm(values[left] - values[right]) / denominator)
    return {
        "reference_timestep": int(timesteps[0]),
        "reference_relative_frobenius": [float(value) for value in reference_relative],
        "pairwise_normalized_frobenius": pairwise.tolist(),
        "mean_reference_relative_frobenius": float(reference_relative.mean()),
        "max_pairwise_normalized_frobenius": float(pairwise.max()),
    }


@torch.no_grad()
def factorization_metric(
    kernel: torch.Tensor,
    basis: torch.Tensor,
    batch_size: int,
) -> dict[str, float]:
    """Compute ||P_c A(I-Pi)|| / ||P_c A|| by direct finite-dimensional action.

    The fine-space basis has dimension 3*32*32.  The fixed coarse map is the
    normalized 2x2 block average P_c; its adjoint lifts a coarse value to each
    block entry with factor 1/2, so Pi=P_c^T P_c is the block-constant
    orthogonal projector used by the diagnostic.
    """
    total_squared = torch.zeros((), device=basis.device, dtype=torch.float64)
    residual_squared = torch.zeros((), device=basis.device, dtype=torch.float64)
    count = int(basis.shape[0])
    for start in range(0, count, batch_size):
        fine = basis[start : start + batch_size].reshape(-1, 3, 32, 32)
        coarse = normalized_haar_lowpass(fine)
        lifted = coarse.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3).div(2.0)
        full_output = normalized_haar_lowpass(circular_conv2d(fine, kernel))
        residual_output = normalized_haar_lowpass(circular_conv2d(fine - lifted, kernel))
        total_squared += full_output.double().square().sum()
        residual_squared += residual_output.double().square().sum()
    total = float(torch.sqrt(total_squared).cpu())
    residual = float(torch.sqrt(residual_squared).cpu())
    return {
        "operator_frobenius": total,
        "fine_residual_frobenius": residual,
        "normalized_fine_residual": residual / max(total, 1.0e-12),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config = load_e2v2_config(PROJECT_ROOT)
    configure_torch_runtime(config)
    device = torch.device(args.device or str(config["runtime"]["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA mechanism diagnostics but CUDA is unavailable")
    selection = load_json(project_path(PROJECT_ROOT, config["runs_root"]) / "selected_checkpoints.json")
    if selection.get("status") != "PASS" or set(selection.get("conditions", {})) != set(config["conditions"]):
        raise RuntimeError("mechanism diagnostics require complete six-condition selection")
    output_root = project_path(PROJECT_ROOT, config["mechanism_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "mechanism_diagnostics.json"
    if output_path.exists():
        raise RuntimeError("mechanism diagnostic output already exists; refusing replacement")

    timesteps = grid_timesteps(config)
    basis = torch.eye(3 * 32 * 32, device=device, dtype=torch.float32)
    batch_size = int(config["mechanism"]["operator_batch_size"])
    records: dict[str, Any] = {}
    for condition in config["mechanism"]["report_pre_conditions"]:
        if condition not in PRE_CONDITIONS:
            raise RuntimeError(f"mechanism report condition is not pre-projection: {condition}")
        selected = selection["conditions"][condition]
        checkpoint_path = project_path(PROJECT_ROOT, selected["selected_checkpoint"])
        if sha256_file(checkpoint_path) != selected["selected_checkpoint_sha256"]:
            raise RuntimeError(f"selected checkpoint hash mismatch for {condition}")
        model = model_from_checkpoint(condition, config, checkpoint_path, device)
        kernels = effective_kernels(model, timesteps, device)
        factorization = [
            factorization_metric(kernels[index], basis, batch_size)
            for index in range(kernels.shape[0])
        ]
        records[condition] = {
            "selected_step": int(selected["selected_step"]),
            "selected_checkpoint": selected["selected_checkpoint"],
            "selected_checkpoint_sha256": selected["selected_checkpoint_sha256"],
            "operator_domain": "3-channel 32x32 fine space",
            "coarse_projection": "normalized 2x2 block average P_c",
            "projector": "Pi=P_c^T P_c, block-constant orthogonal projector",
            "timesteps": [int(value) for value in timesteps],
            "time_variation": time_variation(kernels, timesteps),
            "factorization": factorization,
        }
        del model, kernels
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = {
        "schema_version": 1,
        "status": "PASS",
        "protocol_id": config["protocol_id"],
        "interpretation_boundary": "supporting mechanism readout only; not a proof of PSG repair, strict risk gain, or causality",
        "conditions": records,
    }
    save_json(output_path, report)
    print(f"E2V2_MECHANISM_PASS output={output_path}")


if __name__ == "__main__":
    main()
