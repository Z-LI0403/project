"""Shared predictor and adapter implementation for the E2-v2 experiment."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
import yaml


CONDITIONS = ("fixed", "static_pre", "static_post", "time_pre", "time_post")


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def project_path(root: str | Path, value: str) -> Path:
    return Path(root).resolve() / value


def build_schedule(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spec = config["diffusion"]
    total = int(spec["timesteps"])
    beta_start = float(spec["beta_start"])
    beta_end = float(spec["beta_end"])
    indices = np.arange(total, dtype=np.float64)
    betas = beta_start + (beta_end - beta_start) * indices / float(total - 1)
    alpha_bar = np.cumprod(1.0 - betas, dtype=np.float64)
    log_snr = np.log(alpha_bar / (1.0 - alpha_bar))
    u = np.clip(log_snr, -12.0, 12.0) / 12.0
    return betas, alpha_bar, u


def schedule_knots(config: dict[str, Any]) -> np.ndarray:
    _, _, u = build_schedule(config)
    times = np.asarray(config["diffusion"]["knot_timesteps"], dtype=np.int64)
    return u[times]


def hat_weights(u: Tensor, knots: Tensor) -> Tensor:
    """Standard nonnegative five-knot hat weights on the realized schedule range."""
    if knots.numel() != 5:
        raise ValueError("the experiment requires exactly five knots")
    u = u.reshape(-1)
    weights = [torch.clamp((knots[1] - u) / (knots[1] - knots[0]), min=0.0)]
    for m in (1, 2, 3):
        left = (u - knots[m - 1]) / (knots[m] - knots[m - 1])
        right = (knots[m + 1] - u) / (knots[m + 1] - knots[m])
        weights.append(torch.clamp(torch.minimum(left, right), min=0.0))
    weights.append(torch.clamp((u - knots[3]) / (knots[4] - knots[3]), min=0.0))
    return torch.stack(weights, dim=1)


def normalized_haar_lowpass(x: Tensor) -> Tensor:
    if x.ndim != 4 or x.shape[-2:] != (32, 32):
        raise ValueError(f"expected BCHW tensor with 32x32 spatial shape, got {tuple(x.shape)}")
    batch, channels, _, _ = x.shape
    return x.reshape(batch, channels, 16, 2, 16, 2).sum(dim=(3, 5)) / 2.0


def circular_conv2d(x: Tensor, weight: Tensor) -> Tensor:
    return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="circular"), weight)


class StaticAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(3, 3, 3, 3))
        self.reset_identity()

    @torch.no_grad()
    def reset_identity(self) -> None:
        self.weight.zero_()
        for channel in range(3):
            self.weight[channel, channel, 1, 1] = 1.0

    def forward(self, x: Tensor) -> Tensor:
        return circular_conv2d(x, self.weight)


class TimeAdapter(nn.Module):
    def __init__(self, knots: Iterable[float]) -> None:
        super().__init__()
        self.base = nn.Parameter(torch.zeros(3, 3, 3, 3))
        self.residual = nn.Parameter(torch.zeros(4, 3, 3, 3, 3))
        self.register_buffer("knots", torch.as_tensor(tuple(knots), dtype=torch.float32))
        self.reset_identity()

    @torch.no_grad()
    def reset_identity(self) -> None:
        self.base.zero_()
        self.residual.zero_()
        for channel in range(3):
            self.base[channel, channel, 1, 1] = 1.0

    def kernels(self, u: Tensor) -> Tensor:
        weights = hat_weights(u, self.knots)[:, 1:]
        return self.base.unsqueeze(0) + torch.einsum("bm,moihw->boihw", weights, self.residual)

    def forward(self, x: Tensor, u: Tensor) -> Tensor:
        kernels = self.kernels(u)
        batch, channels, height, width = x.shape
        padded = F.pad(x, (1, 1, 1, 1), mode="circular")
        grouped_input = padded.reshape(1, batch * channels, height + 2, width + 2)
        grouped_weight = kernels.reshape(batch * 3, 3, 3, 3)
        result = F.conv2d(grouped_input, grouped_weight, groups=batch)
        return result.reshape(batch, 3, height, width)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        if dim != 128:
            raise ValueError("the experiment fixes the time embedding at 128 dimensions")
        frequencies = 10000.0 ** (-torch.arange(64, dtype=torch.float32) / 63.0)
        self.register_buffer("frequencies", frequencies)

    def forward(self, t: Tensor) -> Tensor:
        phases = t.float().reshape(-1, 1) * self.frequencies.reshape(1, -1)
        return torch.stack((torch.sin(phases), torch.cos(phases)), dim=-1).reshape(t.shape[0], 128)


class ResidualBlock(nn.Module):
    def __init__(self, width: int, groups: int, eps: float, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, width, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(width, width, 3, padding=0, bias=True)
        self.time_projection = nn.Linear(time_dim, width, bias=True)
        self.norm2 = nn.GroupNorm(groups, width, eps=eps, affine=True)
        self.conv2 = nn.Conv2d(width, width, 3, padding=0, bias=True)

    def forward(self, x: Tensor, time_embedding: Tensor) -> Tensor:
        hidden = self.norm1(x)
        hidden = F.silu(hidden)
        hidden = circular_conv2d(hidden, self.conv1.weight)
        if self.conv1.bias is not None:
            hidden = hidden + self.conv1.bias.reshape(1, -1, 1, 1)
        hidden = hidden + self.time_projection(time_embedding).reshape(x.shape[0], -1, 1, 1)
        hidden = self.norm2(hidden)
        hidden = F.silu(hidden)
        hidden = circular_conv2d(hidden, self.conv2.weight)
        if self.conv2.bias is not None:
            hidden = hidden + self.conv2.bias.reshape(1, -1, 1, 1)
        return x + hidden


class CoarsePredictor(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        spec = config["model"]
        width = int(spec["predictor_width"])
        groups = int(spec["group_norm_groups"])
        eps = float(spec["group_norm_epsilon"])
        time_dim = int(spec["time_embedding_dim"])
        self.time_basis = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.stem = nn.Conv2d(3, width, 3, padding=0, bias=True)
        self.blocks = nn.ModuleList(
            ResidualBlock(width, groups, eps, time_dim)
            for _ in range(int(spec["residual_blocks"]))
        )
        self.output_norm = nn.GroupNorm(groups, width, eps=eps, affine=True)
        self.output = nn.Conv2d(width, 3, 3, padding=0, bias=True)

    def forward(self, z: Tensor, t: Tensor) -> Tensor:
        time_embedding = self.time_mlp(self.time_basis(t))
        hidden = circular_conv2d(z, self.stem.weight)
        if self.stem.bias is not None:
            hidden = hidden + self.stem.bias.reshape(1, -1, 1, 1)
        for block in self.blocks:
            hidden = block(hidden, time_embedding)
        hidden = F.silu(self.output_norm(hidden))
        result = circular_conv2d(hidden, self.output.weight)
        if self.output.bias is not None:
            result = result + self.output.bias.reshape(1, -1, 1, 1)
        return result


@torch.no_grad()
def initialize_predictor(predictor: CoarsePredictor, seed: int) -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    second_convolutions = {id(block.conv2) for block in predictor.blocks}
    for module in predictor.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if id(module) in second_convolutions:
                module.weight.zero_()
            else:
                if isinstance(module, nn.Conv2d):
                    fan_in = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
                else:
                    fan_in = module.in_features
                values = torch.randn(module.weight.shape, generator=generator, dtype=torch.float32)
                module.weight.copy_(values * math.sqrt(2.0 / float(fan_in)))
            if module.bias is not None:
                module.bias.zero_()
        elif isinstance(module, nn.GroupNorm):
            module.weight.fill_(1.0)
            module.bias.zero_()


class PredictorAdapterModel(nn.Module):
    def __init__(self, condition: str, config: dict[str, Any], knots: Iterable[float]) -> None:
        super().__init__()
        if condition not in CONDITIONS:
            raise ValueError(f"unknown condition {condition!r}")
        self.condition = condition
        self.predictor = CoarsePredictor(config)
        if condition in ("static_pre", "static_post"):
            self.adapter: nn.Module | None = StaticAdapter()
        elif condition in ("time_pre", "time_post"):
            self.adapter = TimeAdapter(knots)
        else:
            self.adapter = None

    def adapter_parameter_count(self) -> int:
        if self.adapter is None:
            return 0
        return sum(parameter.numel() for parameter in self.adapter.parameters())

    def forward(self, x_t: Tensor, t: Tensor, u: Tensor) -> Tensor:
        if self.condition == "fixed":
            coarse = normalized_haar_lowpass(x_t)
        elif self.condition == "static_pre":
            assert isinstance(self.adapter, StaticAdapter)
            coarse = normalized_haar_lowpass(self.adapter(x_t))
        elif self.condition == "time_pre":
            assert isinstance(self.adapter, TimeAdapter)
            coarse = normalized_haar_lowpass(self.adapter(x_t, u))
        elif self.condition == "static_post":
            assert isinstance(self.adapter, StaticAdapter)
            coarse = self.adapter(normalized_haar_lowpass(x_t))
        elif self.condition == "time_post":
            assert isinstance(self.adapter, TimeAdapter)
            coarse = self.adapter(normalized_haar_lowpass(x_t), u)
        else:  # pragma: no cover
            raise RuntimeError(self.condition)
        return self.predictor(coarse, t)


def clone_state_dict(module: nn.Module) -> dict[str, Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def state_dicts_equal(left: dict[str, Tensor], right: dict[str, Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)


def images_to_float_tensor(images: np.ndarray, device: torch.device) -> Tensor:
    array = np.ascontiguousarray(images.transpose(0, 3, 1, 2))
    tensor = torch.from_numpy(array).to(device=device, dtype=torch.float32)
    return tensor.div_(127.5).sub_(1.0)


@dataclass
class TrainingBatch:
    indices: np.ndarray
    timesteps: np.ndarray
    noise: Tensor


class PairedTrainingStreams:
    """Independent mutable streams whose reset reproduces paired condition inputs."""

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
        timesteps = self.time_rng.integers(0, 1000, size=batch_size, dtype=np.int64)
        noise = torch.randn(
            (batch_size, 3, 32, 32),
            generator=self.noise_generator,
            dtype=torch.float32,
        )
        return TrainingBatch(indices=indices, timesteps=timesteps, noise=noise)

    def state_dict(self) -> dict[str, Any]:
        return {
            "order_rng": copy.deepcopy(self.order_rng.bit_generator.state),
            "time_rng": copy.deepcopy(self.time_rng.bit_generator.state),
            "noise_generator": self.noise_generator.get_state(),
            "permutation": self.permutation.copy(),
            "position": int(self.position),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.order_rng.bit_generator.state = state["order_rng"]
        self.time_rng.bit_generator.state = state["time_rng"]
        self.noise_generator.set_state(state["noise_generator"])
        self.permutation = np.asarray(state["permutation"], dtype=np.int64)
        self.position = int(state["position"])


def learning_rate(step: int, total_steps: int, config: dict[str, Any]) -> float:
    spec = config["training"]
    warmup = int(spec["warmup_steps"])
    peak = float(spec["peak_learning_rate"])
    final = float(spec["final_learning_rate"])
    if not 1 <= step <= total_steps:
        raise ValueError(f"step {step} outside 1..{total_steps}")
    if step <= warmup:
        return peak * step / warmup
    progress = (step - warmup) / (total_steps - warmup)
    return final + 0.5 * (peak - final) * (1.0 + math.cos(math.pi * progress))


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def deep_copy_config(config: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(config)
