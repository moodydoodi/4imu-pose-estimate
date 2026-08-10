# 3D Pose Estimation from Four Wrist- and Ankle-Worn IMUs

Estimating 13-joint human pose from four Axivity AX6 inertial sensors (both wrists,
both ankles), with synthetic training data generated from AMASS motion capture.

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="PyTorch" src="https://img.shields.io/badge/pytorch-%E2%89%A52.0-ee4c2c">
  <img alt="Sensors" src="https://img.shields.io/badge/sensors-4%20%C3%97%20AX6-informational">
  <img alt="Joints" src="https://img.shields.io/badge/joints-13-informational">
  <img alt="Status" src="https://img.shields.io/badge/status-course%20project-lightgrey">
</p>

Module **Technical Project** · Hochschule Düsseldorf
Authors: Dalia Salih · Anton Rabanus

---

## At a glance

| | |
|---|---|
| **Input** | 4 × Axivity AX6 (accelerometer + gyroscope, **no magnetometer**), wrists and ankles |
| **Output** | 13-joint, pelvis-centred pose at 50 fps, in a body-fixed frame |
| **Ground truth** | MediaPipe 3D landmarks from a synchronised video |
| **Data** | 6 real recordings (≈ 1 h) + 800 synthetic recordings (4.13 h) from AMASS |
| **Model** | TCN stem → BiLSTM → two-stage bone-direction head, 4.44 M parameters |
| **Best MPJPE** | **89.5 mm** (leave-one-recording-out, 3 seeds) vs. 105.4 mm trivial baseline |

Two findings carry the project. Choosing a **body-fixed target frame** — instead of
the world frame comparable systems use — was by far the largest improvement
(relative gain over the trivial baseline 6.1 % → 13.4 %). **Synthetic pre-training**
adds a smaller but statistically clean −2.86 mm; against expectation, *more*
synthetic data made that effect *smaller*.

**Contents** · [About](#about) · [Results](#results) · [Getting started](#getting-started) ·
[Reproducing the results](#reproducing-the-results) · [Layout](#layout) ·
[Method](#method-in-brief) · [Dashboard](#dashboard) · [Limitations](#limitations) ·
[Known gaps](#known-gaps-in-this-release) · [Data sources](#data-sources-and-licences)

---

## About

Reconstructing full-body pose from four cheap limb-mounted sensors. Two properties
of this setup shape every design decision, and both differ from the published
systems this work is compared against.

**Rotation about the vertical is not observable.** DIP, TransPose, PIP and IMUPoser
are all fed ready-made orientation matrices, which come from a magnetometer-based
sensor fusion. The AX6 has no magnetometer, so heading drifts with the gyroscope
integral — feeding an *estimated* orientation to the network would mean training on
a quantity that drifts away at inference time. The pipeline therefore uses only
observable quantities as input and expresses targets in a body-fixed frame that
contains no heading. Measured on video1, removing the heading drops the trivial
baseline from 135.7 to 103.6 mm: about 32 mm of the total pose variance is heading
alone, and no four-sensor system without a magnetometer can recover it.

**Six recordings are not enough** for 4.44 M parameters and ≈ 9 400 training
windows. Synthetic AMASS-derived data is used to pre-train, the six real recordings
to fine-tune. Mixing both in one pool would drown out six videos among hundreds.

---

## Results

Leave-one-recording-out over six recordings, three seeds, errors in **mm**.

| Condition | MPJPE ↓ | PA-MPJPE ↓ | PCK@100 ↑ | raw output |
|---|---:|---:|---:|---|
| Trivial mean-pose baseline | 105.4 | – | – | `docs/RESULTS.md` §1 |
| Real data only | 92.3 | 71.2 | 66.8 % | baseline of all rows below |
| **+ pre-training, 120 rec., 1 epoch** | **89.5** | **69.1** | **67.7 %** | `results/pilot_e1_vs_real.*` |
| + pre-training, 120 rec., 2 epochs | 89.6 | 69.7 | 67.6 % | `results/pilot_e2_vs_real.*` |
| + pre-training, 800 rec., 1 epoch | 90.8 | 70.2 | 67.0 % | `results/final800_e1_h110_vs_real.*` |

> **Note.** The best number comes from the *pilot* pre-training set (120 CMU
> recordings). The larger "final" 800-recording set performs **worse** and its
> interval includes zero. Both are reported on purpose — see [Limitations](#limitations).

**Is the effect real?** Single folds are useless here: variants differ by one to two
millimetres, and two seeds of the *same* variant differ just as much. The effect is
therefore measured as a paired comparison over 3 seeds × 6 test recordings = 18
pairs, with bootstrap 95 % confidence intervals.

| Pre-training set | ≈ steps | Δ MPJPE | 95 % CI | better in |
|---|---:|---:|---|---:|
| 120 recordings, 1 epoch | 118 | **−2.86** | [−4.35, −1.42] | 16 / 18 |
| 120 recordings, 2 epochs | 237 | −2.77 | [−4.40, −1.14] | 14 / 18 |
| 800 recordings, 1 epoch | 781 | −1.48 | [−3.35, **+0.43**] | 14 / 18 |

For the first condition PA-MPJPE gives −2.09 [−3.47, −0.89] and PCK@100 +0.89
[+0.12, +1.74] — both intervals exclude zero.

**How good can these numbers get?** MediaPipe's own bone lengths fluctuate by
**20.0 mm** frame to frame, so any rigid skeleton is roughly that far from the raw
targets by construction; forcing every subject onto one shared skeleton costs a
further 1.1 mm. Values below ≈ 20 mm would be fitting noise in the ground truth.

Per-joint errors, all checks and the full tables are in
[`docs/RESULTS.md`](docs/RESULTS.md); per-fold output in [`results/`](results/).

---

## Getting started

Python 3.10+ (developed on 3.11). `numpy`, `pandas`, `scipy` for preprocessing,
features and evaluation; `torch ≥ 2.0` for training and inference only;
`matplotlib` optionally for the synthesis plots.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export PYTHONPATH=src/poser        # Windows cmd: set PYTHONPATH=src\poser
```

The `poser` modules import each other flatly (`import skeleton`), so the package
directory has to be on the path.

**1 · Check the mechanics, no data needed (≈ 10 s)**

```bash
python src/poser/train.py --dry-run
```

Verifies the torch forward kinematics against the numpy reference, that bone lengths
in the output are exact, that augmentation leaves the gravity axis untouched, and
that gradients flow.

**2 · Run the bundled sample** — a 90 s excerpt of one recording ships with the
repository so the pipeline can be executed end to end without the (unpublished)
dataset.

```bash
# sensor-to-segment rotation (rewrites data/sample/video1/*_segment.csv)
python src/preprocess/to_segment.py data/sample/video1 --suffix _aligned

# features + cache, reusing the shipped sample skeleton
python src/poser/prepare.py --data data/sample --suffix _segment --frame body \
    --cache cache/sample --skeleton-in config/skeleton_sample.json

# data checks
python src/poser/selftest.py --cache cache/sample --skeleton config/skeleton_sample.json
```

<details>
<summary><b>What the output should look like</b></summary>

`to_segment.py` prints `video1  4/4 sensors   uncertain: left_wrist, right_wrist,
right_ankle`. "Uncertain" only means the Kabsch residual exceeds a conservative
threshold on a 90 s excerpt; the full recordings are longer and fit more tightly.
The rotation is applied either way.

In `selftest.py` the `exact` column must be ≈ 0.00 (the bone-direction
parameterisation reproduces the pose), `|g|` must be 1.000, and `ankle/wrist`
(impact-energy ratio) must be well above 1 — otherwise the sensor assignment is wrong.

</details>

> ⚠ Use `--skeleton-in` to *reuse* a skeleton file. `--skeleton` **writes** a new one
> and will overwrite the version-controlled `config/*.json`.

---

## Reproducing the results

Needs the full dataset under `data/processed/` (not redistributable, see
[Data sources](#data-sources-and-licences)).

```bash
# 1  features and canonical skeleton
python src/poser/prepare.py --data data/processed --exclude video7 --suffix _segment \
    --frame body --cache cache/real_body --skeleton config/skeleton.json
python src/poser/prepare.py --data synthdata/output/recordings --suffix _segment \
    --frame body --cache cache/synth_body --skeleton-in config/skeleton.json

# 2  checks before spending GPU time
python src/poser/selftest.py --cache cache/real_body --skeleton config/skeleton.json
python src/poser/checklag.py --cache cache/real_body
python src/poser/floor.py   --data data/processed --skeleton config/skeleton.json

# 3  baseline: real data only, every recording once as the test case
python src/poser/train.py --cache cache/real_body --skeleton config/skeleton.json \
    --loro --epochs 20 --lr 5e-4 --patience 8 --seed 0 --out models/base_s0

# 4  pre-train on synthetic, then fine-tune on real
python src/poser/train.py --cache cache/synth_body --skeleton config/skeleton.json \
    --epochs 1 --lr 5e-4 --seed 0 --out models/pre_s0
python src/poser/train.py --cache cache/real_body --skeleton config/skeleton.json \
    --loro --epochs 20 --lr 2e-4 --patience 8 --seed 0 \
    --init models/pre_s0/best.pt --out models/ft_s0

# 5  paired comparison
python src/poser/compare.py models/base_s0 models/ft_s0
```

Repeat steps 3–4 with `--seed 1` and `--seed 2` for the 18 pairs behind the intervals.

Each run writes `best_<video>.pt` per fold, `model_card.json` (suffix, frame, fps,
seed, source cache), `pred_<video>.npz` (predicted and true pose per frame) and
`metrics.json`. The last two are enough to redo the evaluation without retraining.
`best.pt` is a convenience copy of the **last** fold, not a best-over-folds model.

<details>
<summary><b>Generating the synthetic data</b></summary>

Needs the AMASS sequences and an SMPL-H body model, neither redistributable.

```bash
python src/synth/check_setup.py      # reports what is missing and where it is expected
python src/synth/run_pipeline.py --sample 600 --max-seconds 60 --foot-impacts
```

Four steps: measure the AX6 noise profile from the real recordings → convert AMASS
to the pose format → generate virtual AX6 signals → compare against a real recording.

</details>

---

## Layout

```
src/poser/          model, features, training, evaluation
    config.py  features.py  skeleton.py  dataio.py  augment.py  model.py
    prepare.py      → cache/*.npz (features + targets, no torch needed)
    train.py        training, LORO evaluation, metrics, --dry-run self-check
    selftest.py  checklag.py  floor.py   data checks and reference values
    infer.py  compare.py  npz_to_dashboard.py
src/preprocess/     sensor-to-segment calibration (estimate_mount, to_segment)
src/synth/          synthetic data from AMASS (check_setup, run_pipeline,
                    sensor_noise_profile, amass_to_pose, retarget, mounting,
                    synth_imu, validate_*)
dashboard/          local web dashboard for inspecting predictions
config/             canonical skeleton (full dataset + bundled sample)
data/sample/        90 s excerpt of one recording so the pipeline can be run
results/            evaluation output of the reported experiments
docs/RESULTS.md     detailed results, per-joint errors, all checks
```

`data/raw/`, `data/processed/`, `cache/`, `models/`, `logs/` and `synthdata/` are not
version-controlled, apart from the bundled sample.

---

## Method in brief

**Input — 12 channels per sensor, 48 in total**

| # | Channel | Why |
|---:|---|---|
| 3 | Gravity direction in the sensor frame | Drift free — the accelerometer keeps correcting the complementary filter. Two of three orientation DOF. |
| 3 | Linear acceleration (acc − gravity) | Observable without a magnetometer. |
| 3 | Angular rate | Observable without a magnetometer. |
| 2 | \|acc\|, \|gyr\| | Magnitudes, invariant under mounting rotation. |
| 1 | Energy in the 20–90 Hz band | Everything is resampled to 50 Hz to match the pose, which would otherwise discard the landing impact. At the ankles ≈ 6× larger than at the wrists. |

Gravity is tracked with a complementary filter (Mahony without the magnetometer
branch); the accelerometer only corrects while `|acc|` ≈ 1 g, so the gyroscope
carries alone through the flight phase of a jump.

**Sensor frame.** Signals are rotated from the *device* frame (depends on how the
band happened to sit) into the *segment* frame (depends only on the body). DIP and
TransPose use a T-pose for this; none was recorded here, but a video exists for every
recording, so the rotation is estimated over the whole recording with a Kabsch fit of
the measured gravity direction against an anatomical frame derived from the pose —
the functional sensor-to-segment calibration used in biomechanics.
*Verified:* changing the simulated band orientation alters `_aligned` by 128 units
and `_mp_spatial` by 118, but `_segment` by **0.34**.

**Output.** Twelve bone directions as unit vectors — not joint positions, not 6D
rotations. With fixed bone lengths that is an exact parameterisation of a
pelvis-centred 13-joint pose (24 DOF); positions follow from a prefix sum along the
chain, so bone lengths are correct by construction and cannot drift.
*Verified:* true directions with true lengths reproduce the pose to 0.00 mm.

**Network.** `48 feat → LayerNorm → TCN stem (dilations 1/2/4 ≈ 0.4 s ≈ one landing)
→ BiLSTM (2×256) → stage 1: the 4 sensor-carrying bones (own loss term) → BiLSTM
→ stage 2: all 12 bones → forward kinematics → 13 joints`. Two-stage output is
leaf-to-full as in TransPose. A BiLSTM rather than a transformer or ST-GCN: with ≈ 1 h
of real data, capacity is not the bottleneck (`docs/RESULTS.md` §5). Loss = Huber on
positions + 0.5·direction cosine + 1.0·Huber on velocity + 0.5·Huber on stage 1.

**Synthetic data.** AMASS sequences are converted to the pose format, the skeleton is
scaled to the real subjects, virtual AX6 signals are derived per sensor site, and the
signals are then degraded with a noise profile *measured from the real recordings*:
axis scale error (0.968–1.072), gyro offset, Gauss-Markov bias drift, coloured noise
with measured AR(1), quantisation and range clipping. One profile per real recording
is drawn at random per generated recording, as is the mounting.
*Validated on the 48 input features:* the median distance between synthetic and real
(0.22 σ) equals the median distance between two real recordings (0.22 σ), and in 69 %
of features the synthetic data is closer to a real recording than two real recordings
are to each other. The one serious defect is the wrist impact band — see below.

---

## Dashboard

```bash
python src/poser/npz_to_dashboard.py --models models/ft_s0 \
    --cache cache/real_body --data data/processed
cd dashboard && python serve.py        # localhost only, opens http://127.0.0.1:8000
```

Choose the project root, pick a recording, compare models under *Evaluation*.
Predictions of body-frame models are rotated back into the world frame using the
**ground-truth heading** so the skeleton lines up with the video. No error figure
changes — both sides are rotated by the same matrix — but it is flagged as
`heading_from_gt`, because heading is not something the model predicts.

---

## Evaluation protocol

One recording for testing, a second for validation (early stopping), the rest for
training; the test recording is never touched. `--loro` repeats this with every
recording as the test case and averages — a single number on a single test recording
is far too noisy to judge a change by. Every configuration runs with seeds 0, 1, 2:
the six-fold mean is stable across seeds (92.05 / 92.26 / 92.68) while single folds
vary widely (video4: 93.15 / 102.39 / 93.52). Metrics are MPJPE, PA-MPJPE (per-frame
Procrustes, removes global rotation and scale), PCK@50, PCK@100, plus the trivial
mean-pose baseline of the same fold. video1/video5 and video4/video6 are the same
subjects respectively; excluding video5 entirely changes video1 by 1.5 mm, so the
LORO numbers are sound (`docs/RESULTS.md` §5).

---

## Limitations

**Deliberate scope.** Global position and heading are not predicted. Without a trunk
sensor and without a magnetometer neither is observable, and a model that outputs
them outputs guesses. Evaluation is pelvis-centred, and PA-MPJPE is reported next to
MPJPE so the share of the error that is pure rotation stays visible.

**Control conditions implemented but not run.** `train.py --scramble-targets` pairs
each recording's sensor data with another recording's poses — input and target
distributions unchanged, only the relation destroyed. If pre-training still helped,
the gain would be a better weight initialisation rather than transferred knowledge.
Together with `--max-recordings` this separates the *amount* of pre-training from its
*content*. These runs have not been carried out, so both explanations remain confounded.

| | |
|---|---|
| **Ground truth is MediaPipe, not a marker system.** | Bone lengths fluctuate (≈ 20 mm) and the side facing away from the camera is measured systematically shorter — left thigh 412 mm vs. right 342 mm in the canonical skeleton. Training and evaluation use the same targets, so comparisons are not distorted, but it bounds how good the targets can be. |
| **More synthetic data made the effect smaller.** | The benefit decays monotonically with pre-training steps. Either it is only a better initialisation that longer pre-training destroys, or the 800-recording set is worse in content (120 = CMU only; 800 adds ACCAD, DanceDB, HDM05, TotalCapture — crawling, dance, martial arts, all further from jumping). |
| **Synthesis artefact at the wrists.** | The impact-band channel is ≈ 10× too large at the wrists (0.22 vs. 0.021 real). Real wrists carry almost no energy there; the pose is smoothed at 25 Hz and differentiated twice, which amplifies high frequencies quadratically. At the ankles the channel is correct once `--foot-impacts` is on. |
| **Ankles are the worst joints** (153 / 166 mm). | The arm error was dominated by trunk uncertainty and largely disappears in the body frame; the leg error is actual movement. |
| **Six recordings, four subjects.** | Every conclusion rests on a very small sample. The paired design and bootstrap intervals exist because of that, but do not substitute for more data. |
| **Offline only.** | The BiLSTM is bidirectional and the impact-band feature needs a sliding window, so the system is not real-time capable as built. |

---

## Known gaps in this release

Listed openly so reviewers do not have to find them:

- The aggregation script that produced the bootstrap intervals in `results/*.json` is
  **not in this repository**. `src/poser/compare.py` does a paired sign test and a
  *t*-statistic, not the bootstrap. The per-fold inputs (`results/*.csv`) are included,
  so the intervals can be recomputed.
- `run_pipeline.py --selection` and `synth_imu.py --mount` expect manifests produced by
  `build_amass_manifest.py` and `mount_calib.py`, which are not included. Without
  `--selection` the pipeline runs on all available AMASS sequences.
- `src/synth/common.py` resolves its input and output directories relative to
  `src/synth/`, not to the repository root; the commands above assume the root.
- Parts of `dashboard/inference.py` are a generic loader for a second, differently
  architected model developed in parallel; those code paths are incomplete.
- `config/skeleton.json` still carries German joint names while
  `config/skeleton_sample.json` and `src/poser/config.py` use English ones. Only
  `parents` and the bone lengths are read, so this is cosmetic.

---

## Data sources and licences

| Source | Use | Availability |
|---|---|---|
| **AMASS** (Mahmood et al., ICCV 2019) | Motion sequences for synthetic pre-training | Not included. Own licence, MPI for Intelligent Systems — https://amass.is.tue.mpg.de |
| **SMPL-H** ("Extended SMPL+H", not SMPL-X) | Joint regression from AMASS parameters | Not included. Same source and licence. |
| **Own recordings** (6 sessions, 4 subjects, AX6 + video) | Fine-tuning and all evaluation | Not published — they contain identifiable video. A 90 s excerpt is in `data/sample/`. |
| **MediaPipe Pose** (Google) | 3D landmark ground truth from video | Apache 2.0 |

`src/synth/check_setup.py` reports which external assets are missing and where they
are expected.

**References.** Mahmood et al., *AMASS*, ICCV 2019 · Huang et al., *Deep Inertial
Poser*, SIGGRAPH Asia 2018 · Yi et al., *TransPose*, SIGGRAPH 2021 (leaf-to-full
staging) · Yi et al., *Physical Inertial Poser*, CVPR 2022 · Mollyn et al.,
*IMUPoser*, CHI 2023 · Mahony et al., *Nonlinear complementary filters on the special
orthogonal group*, 2008

<sub>Technical Project, Hochschule Düsseldorf. Coursework, provided as-is.</sub>
