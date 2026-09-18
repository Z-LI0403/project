# True-Time Conditioning

This directory contains the runnable source package for the E2-v2 controlled
CIFAR-10 experiment. The experiment compares fixed, static, true-time, and
null-time adapters at a 32-to-16 interface. The primary comparison is
`D_true_time = time_pre - time_pre_null`; `D_time = time_pre - static_pre` is a
supporting bridge comparison.

## Package layout

```text
cifar10_true_time_conditioning/
├── README.md
├── PROTOCOL.md
├── .gitignore
├── pyproject.toml
├── uv.lock
├── .python-version
├── config/
│   └── experiment.yaml
├── src/
│   ├── predictor.py
│   ├── model.py
│   ├── prepare.py
│   ├── train.py
│   ├── select_checkpoints.py
│   ├── mechanism_diagnostics.py
│   ├── evaluate.py
│   ├── analyze_results.py
│   └── smoke_test.py
└── runtime/
    ├── data/
    ├── runs/
    ├── evaluation/
    └── mechanism/
```

`PROTOCOL.md` defines the scientific question, conditions, contrasts, data
split, and limitations. This README only describes the package layout,
required inputs, and execution order.

## Environment

From this experiment directory:

```bash
uv sync --frozen
```

The locked environment uses Python 3.11.15, NumPy 1.26.4, PyYAML 6.0.2,
PyTorch 2.5.1, and torchvision 0.20.1 with the CUDA 12.1 PyTorch index.

## Required runtime inputs

Supply the CIFAR-10 training corpus and the common predictor initialization at
the paths defined in `config/experiment.yaml`. The split and timestep/noise
banks are created by `prepare.py` and are stored under `runtime/data/`.

## Execution order

```bash
uv run python src/prepare.py
uv run python src/smoke_test.py --device cpu
uv run python src/train.py --condition all --device cuda:0
uv run python src/select_checkpoints.py
uv run python src/mechanism_diagnostics.py --device cuda:0
uv run python src/evaluate.py --device cuda:0
uv run python src/analyze_results.py
```

Training can also be run one condition at a time with `--condition` set to
`fixed`, `static_pre`, `static_post`, `time_pre`, `time_post`, or
`time_pre_null`.

Generated data, checkpoints, logs, evaluation arrays, analysis outputs, and
smoke reports are runtime artifacts under `runtime/` (plus the generated
`results.md`) and are excluded from Git by `.gitignore`.
