# E2-v2 True-Time Conditioning Experiment

## Purpose

This experiment tests whether a constrained pre-projection adapter benefits
from receiving the true diffusion timestep in a controlled CIFAR-10
32-to-16 interface. It evaluates the neural analogue of the theory's
time-conditioned pre-projection comparison; it does not prove the finite
Gaussian/Haar theorem, recover its closed-form optimizer, or establish a
unique causal mechanism.

## Conditions

The experiment uses six conditions:

| Condition | Interface | Adapter parameters |
|---|---|---:|
| `fixed` | fixed normalized-Haar projection | 0 |
| `static_pre` | learned static 3x3 adapter before projection | 81 |
| `static_post` | learned static 3x3 adapter after projection | 81 |
| `time_pre` | five-knot time-conditioned adapter before projection | 405 |
| `time_post` | five-knot time-conditioned adapter after projection | 405 |
| `time_pre_null` | five-knot pre adapter with independently sampled null conditioning | 405 |

The coarse predictor, projected-epsilon target, skip representation, optimizer,
diffusion schedule, precision policy, and common predictor initialization are
shared across conditions. `time_pre_null` receives an independently sampled
timestep only at its adapter; the true timestep still constructs `x_t`, selects
the diffusion schedule, and conditions the coarse predictor.

## Primary and supporting contrasts

The primary capacity-controlled test is

$$
D_{\mathrm{true\_time}}
= \operatorname{MSE}(\mathrm{time\_pre})
- \operatorname{MSE}(\mathrm{time\_pre\_null}).
$$

`time_pre` and `time_pre_null` have the same adapter capacity and training
budget. This contrast tests the value of true timestep information relative to
independently sampled null conditioning.

The supporting bridge comparison is

$$
D_{\mathrm{time}}
= \operatorname{MSE}(\mathrm{time\_pre})
- \operatorname{MSE}(\mathrm{static\_pre}).
$$

Its adapter capacities are not matched, so it is supporting evidence rather
than the primary capacity-controlled test.

The secondary pre/post contrasts are

$$
D_{\mathrm{pre\_static}}
= \operatorname{MSE}(\mathrm{static\_pre})
- \operatorname{MSE}(\mathrm{static\_post}),
$$

and

$$
D_{\mathrm{pre\_time}}
= \operatorname{MSE}(\mathrm{time\_pre})
- \operatorname{MSE}(\mathrm{time\_post}).
$$

Negative values favor the first condition in each contrast. Results are
reported as equal-bin held-out projected-epsilon MSE together with paired
image-cluster bootstrap intervals.

## Data and training

The CIFAR-10 training corpus is divided into:

- 40,000 training images;
- 5,000 validation images;
- 5,000 unused training-corpus holdout images for evaluation.

The three subsets are class-balanced and mutually disjoint. The evaluation
holdout is never used for training or validation checkpoint selection.

Each condition starts from the shared predictor initialization and is trained
for 200,000 optimizer steps. Validation is performed every 5,000 steps on a
shared 20-bin bank with two fixed draws per image and bin. Checkpoint selection
uses only the validation equal-bin MSE, with the earliest checkpoint selected
when there is an exact tie.

The training conditions use paired data streams. The evaluation bank shares
true timesteps and noise draws across all six conditions. The null-timestep
bank is generated independently while preserving the same marginal timestep
support.

## Evaluation and mechanism readout

Evaluation computes per-condition, per-bin, per-image, and per-draw projected-
epsilon MSE on the reserved holdout. The final analysis computes the four
contrasts above and their image-cluster bootstrap intervals.

The mechanism readout uses the learned pre-projection kernels at the 20 bin
centers. It reports time variation and the normalized residual

$$
\frac{\lVert P_c A_t(I-\Pi)\rVert_F}
     {\lVert P_c A_t\rVert_F},
\qquad
\Pi=P_c^{\mathsf T}P_c,
$$

where $\Pi$ is the block-constant projector induced by normalized 2x2
average pooling. This readout is descriptive supporting evidence; it does not
by itself establish a causal mechanism or theorem validity.

## Configuration and runtime outputs

The experiment configuration is [config/experiment.yaml](config/experiment.yaml).
Runtime inputs and generated outputs use the following directories:

- `runtime/data/`: CIFAR-10 data, frozen split/banks, and generated noise banks;
- `runtime/runs/`: per-condition checkpoints, validation records, and logs;
- `runtime/evaluation/`: held-out raw evaluation and contrast analysis;
- `runtime/mechanism/`: mechanism diagnostics.

The source package intentionally does not contain these runtime data or result
artifacts.

## Limitations

- The default experiment uses one training seed and one CIFAR-10 32-to-16
  interface.
- Bootstrap intervals are conditional on the trained checkpoints and fixed
  evaluation bank; they do not quantify across-seed variability.
- The evaluation endpoint is projected-epsilon MSE, not sampling quality, FID,
  or downstream task performance.
- The experiment does not establish generalization to other architectures,
  datasets, resolutions, schedules, or training seeds.
