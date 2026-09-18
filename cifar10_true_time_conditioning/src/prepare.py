"""Prepare the isolated E2-v2 split and paired validation/evaluation banks."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
from torchvision.datasets import CIFAR10

from model import (
    PROJECT_ROOT,
    independent_bank_time,
    keyed_noise,
    load_e2v2_config,
    project_path,
)


def class_permutation(labels: np.ndarray, seed: int, class_id: int) -> np.ndarray:
    indices = np.flatnonzero(labels == class_id).astype(np.int64)
    if indices.size != 5000:
        raise RuntimeError(f"CIFAR-10 train class {class_id} has {indices.size} images, expected 5000")
    generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, class_id])))
    return generator.permutation(indices)


def make_splits(config: dict[str, Any], train_labels: np.ndarray) -> dict[str, np.ndarray]:
    spec = config["data"]
    train_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []
    unused_parts: list[np.ndarray] = []
    for class_id in range(10):
        permutation = class_permutation(train_labels, int(spec["split_root_seed"]), class_id)
        train_parts.append(permutation[: int(spec["train_per_class"])])
        validation_start = int(spec["train_per_class"])
        validation_stop = validation_start + int(spec["validation_per_class"])
        validation_parts.append(permutation[validation_start:validation_stop])
        unused_parts.append(permutation[validation_stop:])
    unused_train = np.concatenate(unused_parts).astype(np.int64)
    if str(spec["evaluation_source"]) != "cifar10_train_holdout":
        raise RuntimeError(f"unsupported E2-v2 evaluation source: {spec['evaluation_source']}")
    if str(spec["evaluation_train_holdout_field"]) != "unused_train_indices":
        raise RuntimeError("the approved E2-v2 evaluation holdout must be the unused_train_indices field")
    evaluation = unused_train.copy()
    result = {
        "train_indices": np.concatenate(train_parts).astype(np.int64),
        "validation_indices": np.concatenate(validation_parts).astype(np.int64),
        "unused_train_indices": unused_train,
        "evaluation_indices": evaluation,
    }
    verify_splits(config, result, train_labels)
    return result


def verify_splits(config: dict[str, Any], splits: dict[str, np.ndarray], train_labels: np.ndarray) -> None:
    spec = config["data"]
    expected_sizes = {
        "train_indices": int(spec["train_images"]),
        "validation_indices": int(spec["validation_images"]),
        "unused_train_indices": 5000,
        "evaluation_indices": int(spec["evaluation_images"]),
    }
    for name, expected in expected_sizes.items():
        values = np.asarray(splits[name], dtype=np.int64)
        if values.ndim != 1 or values.size != expected or np.unique(values).size != values.size:
            raise RuntimeError(f"invalid {name}: shape={values.shape} unique={np.unique(values).size}")
    if np.intersect1d(splits["train_indices"], splits["validation_indices"]).size:
        raise RuntimeError("E2-v2 training and validation indices overlap")
    if np.intersect1d(splits["train_indices"], splits["unused_train_indices"]).size:
        raise RuntimeError("E2-v2 training and unused-train indices overlap")
    if np.intersect1d(splits["validation_indices"], splits["unused_train_indices"]).size:
        raise RuntimeError("E2-v2 validation and unused-train indices overlap")
    if str(spec["evaluation_source"]) != "cifar10_train_holdout":
        raise RuntimeError(f"unsupported E2-v2 evaluation source: {spec['evaluation_source']}")
    if not np.array_equal(splits["evaluation_indices"], splits["unused_train_indices"]):
        raise RuntimeError("E2-v2 evaluation must be exactly the currently unused training-corpus holdout")
    if np.intersect1d(splits["evaluation_indices"], np.arange(50_000, dtype=np.int64)).size != splits["evaluation_indices"].size:
        raise RuntimeError("E2-v2 training-corpus evaluation indices are outside CIFAR-10 train data")
    if [int(np.count_nonzero(train_labels[splits["train_indices"]] == c)) for c in range(10)] != [int(spec["train_per_class"])] * 10:
        raise RuntimeError("E2-v2 training split is not class-balanced")
    if [int(np.count_nonzero(train_labels[splits["validation_indices"]] == c)) for c in range(10)] != [int(spec["validation_per_class"])] * 10:
        raise RuntimeError("E2-v2 validation split is not class-balanced")


def save_or_verify_splits(
    path: Path,
    splits: dict[str, np.ndarray],
    *,
    regenerate_mismatched: bool,
) -> None:
    if path.exists():
        matches = True
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != set(splits):
                matches = False
            else:
                for name, values in splits.items():
                    if not np.array_equal(np.asarray(payload[name], dtype=np.int64), values):
                        matches = False
                        break
        if matches:
            pass
        elif not regenerate_mismatched:
            raise RuntimeError("existing E2-v2 split differs; pass --regenerate to replace the preparation artifact")
        else:
            temporary = path.with_name(path.name + ".tmp.npz")
            if temporary.exists():
                raise RuntimeError(f"stale temporary split file exists: {temporary}")
            np.savez(temporary, **splits)
            os.replace(temporary, path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp.npz")
        if temporary.exists():
            raise RuntimeError(f"stale temporary split file exists: {temporary}")
        np.savez(temporary, **splits)
        os.replace(temporary, path)


def expected_bank_shapes(config: dict[str, Any], images: int) -> tuple[tuple[int, int, int], tuple[int, int, int, int, int, int]]:
    bins = int(config["validation"]["bins"])
    draws = int(config["validation"]["draws_per_image_bin"])
    return (bins, images, draws), (bins, images, draws, 3, 32, 32)


def materialize_bank(
    *,
    config: dict[str, Any],
    indices: np.ndarray,
    bank_path: Path,
    noise_path: Path,
    spec_name: str,
    regenerate_mismatched: bool,
) -> None:
    spec = config[spec_name]
    images = int(spec["images"])
    bins = int(spec["bins"])
    draws = int(spec["draws_per_image_bin"])
    shape = (bins, images, draws)
    noise_shape = (bins, images, draws, 3, 32, 32)
    if bank_path.exists() != noise_path.exists():
        raise RuntimeError(f"partial E2-v2 {spec_name} bank exists; refusing implicit replacement")
    if bank_path.exists():
        matches = True
        with np.load(bank_path, allow_pickle=False) as payload:
            required = {"official_indices", "timesteps", "null_timesteps", "bins", "draws_per_image_bin", "bin_width"}
            if set(payload.files) != required:
                matches = False
            else:
                matches = (
                    np.array_equal(payload["official_indices"], indices)
                    and np.asarray(payload["timesteps"]).shape == shape
                    and np.asarray(payload["null_timesteps"]).shape == shape
                    and int(payload["bins"]) == bins
                    and int(payload["draws_per_image_bin"]) == draws
                )
        if matches:
            noise = np.load(noise_path, mmap_mode="r")
            if noise.shape != noise_shape or noise.dtype != np.float32:
                matches = False
            else:
                return
        if not regenerate_mismatched:
            raise RuntimeError(f"existing {spec_name} preparation bank differs; pass --regenerate to replace it")

    bank_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_noise = noise_path.with_name(noise_path.name + ".tmp.npy")
    if temporary_noise.exists():
        raise RuntimeError(f"stale temporary bank file exists: {temporary_noise}")
    temporary_bank = bank_path.with_name(bank_path.name + ".tmp.npz")
    if temporary_bank.exists():
        raise RuntimeError(f"stale temporary bank file exists: {temporary_bank}")
    true_times = np.empty(shape, dtype=np.int16)
    null_times = np.empty(shape, dtype=np.int16)
    noise = np.lib.format.open_memmap(temporary_noise, mode="w+", dtype=np.float32, shape=noise_shape)
    true_seed = int(spec["true_time_root_seed"])
    null_seed = int(spec["null_time_root_seed"])
    noise_seed = int(spec["noise_root_seed"])
    width = int(spec["bin_width"])
    for bin_id in range(bins):
        for position, official_index in enumerate(indices):
            for draw in range(draws):
                true_times[bin_id, position, draw] = independent_bank_time(
                    true_seed,
                    int(official_index),
                    bin_id,
                    draw,
                    within_bin=True,
                    bin_width=width,
                )
                null_times[bin_id, position, draw] = independent_bank_time(
                    null_seed,
                    int(official_index),
                    bin_id,
                    draw,
                    within_bin=False,
                    bin_width=width,
                )
                noise[bin_id, position, draw] = keyed_noise(
                    noise_seed,
                    int(official_index),
                    bin_id,
                    draw,
                )
        noise.flush()
        print(f"prepared {spec_name} bank bin {bin_id + 1}/{bins}", flush=True)
    del noise
    os.replace(temporary_noise, noise_path)
    np.savez(
        temporary_bank,
        official_indices=np.asarray(indices, dtype=np.int64),
        timesteps=true_times,
        null_timesteps=null_times,
        bins=np.asarray(bins, dtype=np.int64),
        draws_per_image_bin=np.asarray(draws, dtype=np.int64),
        bin_width=np.asarray(width, dtype=np.int64),
    )
    os.replace(temporary_bank, bank_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="download CIFAR-10 if the local copy is absent")
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="replace only mismatched E2-v2 preparation artifacts after a scoped split correction",
    )
    args = parser.parse_args()

    config = load_e2v2_config(PROJECT_ROOT)
    dataset_root = project_path(PROJECT_ROOT, config["dataset_root"])
    dataset_root.mkdir(parents=True, exist_ok=True)
    train = CIFAR10(root=str(dataset_root), train=True, download=args.download)
    train_labels = np.asarray(train.targets, dtype=np.int64)
    if train_labels.shape != (50_000,):
        raise RuntimeError(f"unexpected CIFAR-10 train label shape {train_labels.shape}")
    if not Path(config["common_predictor_init"]).is_absolute():
        init_path = project_path(PROJECT_ROOT, config["common_predictor_init"])
    else:
        init_path = Path(config["common_predictor_init"])
    if not init_path.is_file():
        raise RuntimeError(f"common predictor initialization is missing: {init_path}")

    splits = make_splits(config, train_labels)
    save_or_verify_splits(
        project_path(PROJECT_ROOT, config["split_file"]),
        splits,
        regenerate_mismatched=args.regenerate,
    )
    materialize_bank(
        config=config,
        indices=splits["validation_indices"],
        bank_path=project_path(PROJECT_ROOT, config["validation_bank_file"]),
        noise_path=project_path(PROJECT_ROOT, config["validation_noise_file"]),
        spec_name="validation",
        regenerate_mismatched=args.regenerate,
    )
    materialize_bank(
        config=config,
        indices=splits["evaluation_indices"],
        bank_path=project_path(PROJECT_ROOT, config["evaluation_bank_file"]),
        noise_path=project_path(PROJECT_ROOT, config["evaluation_noise_file"]),
        spec_name="evaluation",
        regenerate_mismatched=args.regenerate,
    )
    print("E2V2_PREPARATION_COMPLETE")
    print(f"split={config['split_file']}")
    print(f"validation_bank={config['validation_bank_file']}")
    print(f"evaluation_bank={config['evaluation_bank_file']}")


if __name__ == "__main__":
    main()
