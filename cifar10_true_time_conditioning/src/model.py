"""Shared model, data-stream, and numerical utilities for the experiment."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor, nn
import yaml


CODE_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = CODE_DIR.parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from predictor import (  # noqa: E402
    CoarsePredictor,
    StaticAdapter,
    TimeAdapter,
    build_schedule,
    circular_conv2d,
    hat_weights,
    images_to_float_tensor,
    initialize_predictor,
    learning_rate,
    load_config,
    normalized_haar_lowpass,
    project_path,
    schedule_knots,
)


CONDITIONS = (
    "fixed",
    "static_pre",
    "static_post",
    "time_pre",
    "time_post",
    "time_pre_null",
)

EXPECTED_ADAPTER_PARAMETERS = {
    "fixed": 0,
    "static_pre": 81,
    "static_post": 81,
    "time_pre": 405,
    "time_post": 405,
    "time_pre_null": 405,
}

PRE_CONDITIONS = ("static_pre", "time_pre", "time_pre_null")
TIME_CONDITIONS = ("time_pre", "time_post", "time_pre_null")


def load_e2v2_config(project_root: Path | None = None) -> dict[str, Any]:
    root = PROJECT_ROOT if project_root is None else Path(project_root)
    return load_config(root / "experiments/e2_v2_true_time_conditioning/config/experiment.yaml")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
    digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def clone_state_dict(module: nn.Module) -> dict[str, Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def configure_torch_runtime(config: dict[str, Any]) -> None:
    """Apply the protocol's explicit precision and determinism settings."""
    runtime = config["runtime"]
    torch.backends.cuda.matmul.allow_tf32 = bool(runtime["cuda_matmul_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(runtime["cudnn_tf32"])
    torch.backends.cudnn.benchmark = bool(runtime["cudnn_benchmark"])
    torch.backends.cudnn.deterministic = bool(runtime["cudnn_deterministic"])
    torch.set_float32_matmul_precision("highest")


class E2V2Model(nn.Module):
    """The coarse predictor with exactly one controlled interface."""

    def __init__(self, condition: str, config: dict[str, Any], knots: Iterable[float]) -> None:
        super().__init__()
        if condition not in CONDITIONS:
            raise ValueError(f"unknown E2-v2 condition {condition!r}")
        self.condition = condition
        self.predictor = CoarsePredictor(config)
        if condition in ("static_pre", "static_post"):
            self.adapter: nn.Module | None = StaticAdapter()
        elif condition in TIME_CONDITIONS:
            self.adapter = TimeAdapter(knots)
        else:
            self.adapter = None

    def adapter_parameter_count(self) -> int:
        if self.adapter is None:
            return 0
        return sum(parameter.numel() for parameter in self.adapter.parameters())

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        true_u: Tensor,
        adapter_u: Tensor | None = None,
    ) -> Tensor:
        if self.condition == "fixed":
            coarse = normalized_haar_lowpass(x_t)
        elif self.condition == "static_pre":
            assert isinstance(self.adapter, StaticAdapter)
            coarse = normalized_haar_lowpass(self.adapter(x_t))
        elif self.condition == "time_pre":
            assert isinstance(self.adapter, TimeAdapter)
            coarse = normalized_haar_lowpass(self.adapter(x_t, true_u))
        elif self.condition == "time_pre_null":
            assert isinstance(self.adapter, TimeAdapter)
            if adapter_u is None:
                raise ValueError("time_pre_null requires an independent adapter conditioning value")
            coarse = normalized_haar_lowpass(self.adapter(x_t, adapter_u))
        elif self.condition == "static_post":
            assert isinstance(self.adapter, StaticAdapter)
            coarse = self.adapter(normalized_haar_lowpass(x_t))
        elif self.condition == "time_post":
            assert isinstance(self.adapter, TimeAdapter)
            coarse = self.adapter(normalized_haar_lowpass(x_t), true_u)
        else:  # pragma: no cover
            raise RuntimeError(self.condition)
        return self.predictor(coarse, t)


@dataclass
class TrainingBatch:
    indices: np.ndarray
    timesteps: np.ndarray
    null_timesteps: np.ndarray
    noise: Tensor


class E2V2TrainingStreams:
    """Paired true-time/noise streams plus an independent null-time stream."""

    def __init__(self, train_indices: np.ndarray, config: dict[str, Any]) -> None:
        seed = int(config["training"]["seed"])
        namespaces = config["rng_namespaces"]
        self.train_indices = np.asarray(train_indices, dtype=np.int64)
        self.order_rng = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence([seed, int(namespaces["data_order"])]))
        )
        self.time_rng = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence([seed, int(namespaces["timestep_sampling"])]))
        )
        self.null_time_rng = np.random.Generator(
            np.random.PCG64(
                np.random.SeedSequence([seed, int(namespaces["null_conditioning_timestep"])])
            )
        )
        self.noise_generator = torch.Generator(device="cpu")
        self.noise_generator.manual_seed(int(namespaces["corruption_noise"]))
        self.permutation = np.empty(0, dtype=np.int64)
        self.position = 0

    def _take_indices(self, count: int) -> np.ndarray:
        pieces: list[np.ndarray] = []
        remaining = count
        while remaining:
            if self.position >= self.permutation.size:
                self.permutation = self.order_rng.permutation(self.train_indices)
                self.position = 0
            available = self.permutation.size - self.position
            take = min(remaining, available)
            pieces.append(self.permutation[self.position : self.position + take])
            self.position += take
            remaining -= take
        return np.concatenate(pieces)

    def next(self, batch_size: int) -> TrainingBatch:
        indices = self._take_indices(batch_size)
        true_timesteps = self.time_rng.integers(0, 1000, size=batch_size, dtype=np.int64)
        null_timesteps = self.null_time_rng.integers(0, 1000, size=batch_size, dtype=np.int64)
        noise = torch.randn(
            (batch_size, 3, 32, 32),
            generator=self.noise_generator,
            dtype=torch.float32,
        )
        return TrainingBatch(
            indices=indices,
            timesteps=true_timesteps,
            null_timesteps=null_timesteps,
            noise=noise,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "order_rng": copy.deepcopy(self.order_rng.bit_generator.state),
            "time_rng": copy.deepcopy(self.time_rng.bit_generator.state),
            "null_time_rng": copy.deepcopy(self.null_time_rng.bit_generator.state),
            "noise_generator": self.noise_generator.get_state(),
            "permutation": self.permutation.copy(),
            "position": int(self.position),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.order_rng.bit_generator.state = state["order_rng"]
        self.time_rng.bit_generator.state = state["time_rng"]
        self.null_time_rng.bit_generator.state = state["null_time_rng"]
        self.noise_generator.set_state(state["noise_generator"])
        self.permutation = np.asarray(state["permutation"], dtype=np.int64)
        self.position = int(state["position"])


def model_from_checkpoint(
    condition: str,
    config: dict[str, Any],
    checkpoint: Path,
    device: torch.device,
) -> E2V2Model:
    model = E2V2Model(condition, config, schedule_knots(config))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("protocol_id") != config["protocol_id"] or payload.get("condition") != condition:
        raise RuntimeError(f"checkpoint configuration mismatch for {condition}")
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    return model


def independent_bank_time(
    root_seed: int,
    official_index: int,
    bin_id: int,
    draw: int,
    *,
    within_bin: bool,
    bin_width: int,
) -> int:
    generator = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([int(root_seed), int(official_index), int(bin_id), int(draw)]))
    )
    if within_bin:
        return int(generator.integers(bin_id * bin_width, (bin_id + 1) * bin_width))
    return int(generator.integers(0, 1000))


def keyed_noise(root_seed: int, official_index: int, bin_id: int, draw: int) -> np.ndarray:
    generator = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([int(root_seed), int(official_index), int(bin_id), int(draw)]))
    )
    return generator.standard_normal((3, 32, 32), dtype=np.float32)


def load_schedule_tensors(config: dict[str, Any], device: torch.device) -> tuple[Tensor, Tensor]:
    _, alpha_bar, u_values = build_schedule(config)
    return (
        torch.from_numpy(alpha_bar).to(device=device, dtype=torch.float32),
        torch.from_numpy(u_values).to(device=device, dtype=torch.float32),
    )


def adapter_u_for_condition(
    condition: str,
    true_timesteps: Tensor,
    null_timesteps: Tensor,
    u_values: Tensor,
) -> Tensor | None:
    if condition == "time_pre_null":
        return u_values[null_timesteps]
    if condition in TIME_CONDITIONS:
        return u_values[true_timesteps]
    return None


def condition_parameter_counts(config: dict[str, Any]) -> dict[str, int]:
    knots = schedule_knots(config)
    return {
        condition: E2V2Model(condition, config, knots).adapter_parameter_count()
        for condition in CONDITIONS
    }


__all__ = [
    "CONDITIONS",
    "EXPECTED_ADAPTER_PARAMETERS",
    "PRE_CONDITIONS",
    "TIME_CONDITIONS",
    "E2V2Model",
    "E2V2TrainingStreams",
    "TrainingBatch",
    "PROJECT_ROOT",
    "adapter_u_for_condition",
    "build_schedule",
    "circular_conv2d",
    "clone_state_dict",
    "condition_parameter_counts",
    "configure_torch_runtime",
    "independent_bank_time",
    "images_to_float_tensor",
    "initialize_predictor",
    "keyed_noise",
    "learning_rate",
    "load_config",
    "load_e2v2_config",
    "load_schedule_tensors",
    "model_from_checkpoint",
    "normalized_haar_lowpass",
    "project_path",
    "save_json",
    "schedule_knots",
    "sha256_array",
    "sha256_file",
    "sha256_json",
]
