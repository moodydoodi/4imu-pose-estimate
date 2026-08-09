# 3D Pose Estimation from Four Wrist- and Ankle-Worn IMUs

Estimating 13-joint human pose from four Axivity AX6 inertial sensors (both
wrists, both ankles), with synthetic training data generated from AMASS motion
capture.

Technical Project, Hochschule Düsseldorf.

---

## Problem

Four limb-mounted IMUs, no sensor on the trunk, no magnetometer, and six usable
recordings of real data. Ground truth comes from MediaPipe applied to a
synchronised video. Two consequences shape the entire design:

**Rotation about the vertical is not observable.** Every comparable system
(DIP, TransPose, PIP, IMUPoser) is fed ready-made orientation matrices, which
always come from a magnetometer-based sensor fusion. The AX6 has none, so the
heading drifts with the gyroscope integral. Feeding an estimated orientation to
the network would mean training on a quantity that drifts away at inference
time. The pipeline therefore uses only observable quantities as input, and
targets are expressed in a body-fixed frame that contains no heading.

**Six recordings are not enough.** The model has 4.4 M parameters and about
9 400 training windows. Synthetic data generated from AMASS is used to
pre-train, real data to fine-tune.

## Results

Leave-one-recording-out over six recordings, three seeds, error in mm.

| | MPJPE | PA-MPJPE | PCK@50 | PCK@100 |
|---|---:|---:|---:|---:|
| trivial mean-pose baseline | 105.4 | – | – | – |
| real data only | 92.3 | 71.2 | 34 % | 66.8 % |
| synthetic pre-training + fine-tuning | **89.5** | **69.1** | 36 % | 67.7 % |

The effect of pre-training was measured as a paired comparison over 18
seed × recording pairs: −2.86 mm MPJPE, bootstrap 95 % CI [−4.35, −1.42],
better in 16 of 18 pairs. PA-MPJPE and PCK@100 move in the same direction with
intervals that also exclude zero.

Two reference values put these numbers in context. MediaPipe's own bone lengths
fluctuate frame to frame, so any rigid skeleton is about 20 mm away from the
raw targets; and forcing every subject onto one shared skeleton costs a further
1.1 mm on average. Raw numbers below roughly 20 mm would therefore be fitting
noise in the ground truth.

Choosing a body-fixed target frame was the single largest improvement in the
project: the relative gain over the trivial baseline rose from 6.1 % to 13.4 %,
and PA-MPJPE, which is frame independent and therefore directly comparable,
improved from 77.7 to 70.6 mm.

Raw evaluation output is under `results/`.

## Layout

```
src/poser/        model, features, training, evaluation
src/preprocess/   sensor-to-segment calibration
src/synth/        synthetic data generation from AMASS
dashboard/        local web dashboard for inspecting predictions
config/           canonical skeleton
data/sample/      90 s excerpt of one recording so the pipeline can be run
results/          evaluation output of the reported experiments
docs/             method notes and detailed results
```

`data/`, `cache/`, `models/` and `logs/` are not versioned apart from the sample.

## Method in brief

**Input.** Per sensor twelve channels: gravity direction in the sensor frame
(from a complementary filter, drift free), linear acceleration, angular rate,
the two magnitudes, and the energy in the 20–90 Hz band. The last channel exists
because everything is resampled to 50 Hz to match the pose, which would
otherwise discard the landing impact; at the ankles it is about six times larger
than at the wrists.

**Sensor frame.** Signals are rotated from the device frame into the segment
frame of the limb they sit on. The device frame depends on how the band happened
to sit; the segment frame depends only on the body. The rotation is estimated
over a whole recording from the pose (Kabsch on the gravity direction against an
anatomical frame), which is the functional sensor-to-segment calibration used in
biomechanics, with the video as reference instead of a prescribed T-pose.
Verified: changing the simulated band orientation alters `_aligned` by 128 and
`_mp_spatial` by 118 units, but `_segment` by 0.34.

**Output.** Twelve bone directions as unit vectors, not joint positions and not
6D rotations. With fixed bone lengths that is an exact parameterisation of a
pelvis-centred 13-joint pose (24 degrees of freedom), positions follow from a
prefix sum along the chain, and bone lengths are correct by construction.
Verified: true directions with true lengths reproduce the pose to 0.00 mm.

**Network.** Temporal convolution stem (dilations covering about 0.4 s, the
duration of a landing), bidirectional LSTM, then two output stages: first the
four bones carrying a sensor with their own loss term, then all twelve using the
first stage (leaf-to-full, as in TransPose). 4.4 M parameters.

**Synthetic data.** AMASS sequences are converted to the pose format, the
skeleton is scaled to the real subjects, and virtual AX6 signals are derived per
sensor site. Signals are then degraded with a noise profile measured from the
real recordings: axis scale error (measured range 0.968–1.072), gyro offset,
Gauss-Markov bias drift, coloured noise with measured AR(1), quantisation at the
true resolution and range clipping. One profile per real recording is stored and
drawn at random per generated recording, so device behaviour varies. Mounting is
drawn per recording as well.

Validated in the 48 model input features: the median distance between the
synthetic distribution and a real recording equals the median distance between
two real recordings (0.22 σ), and in 69 % of features the synthetic data is
closer than two real recordings are to each other.

## Running it

```bash
pip install -r requirements.txt
export PYTHONPATH=src/poser          # Windows: set PYTHONPATH=src\poser
```

Check the mechanics without any data (about ten seconds):

```bash
python src/poser/train.py --dry-run
```

Prepare the bundled sample and inspect it:

```bash
python src/preprocess/to_segment.py data/sample/video1 --suffix _aligned
python src/poser/prepare.py --data data/sample --suffix _segment --frame body \
    --cache cache/sample --skeleton config/skeleton_sample.json
python src/poser/selftest.py --cache cache/sample --skeleton config/skeleton_sample.json
```

Full run on a complete dataset:

```bash
# 1  features and canonical skeleton
python src/poser/prepare.py --data data/processed --exclude video7 \
    --suffix _segment --frame body --cache cache/real_body \
    --skeleton config/skeleton.json
python src/poser/prepare.py --data synthdata/output/recordings \
    --suffix _segment --frame body --cache cache/synth_body \
    --skeleton-in config/skeleton.json

# 2  data checks before spending GPU time
python src/poser/selftest.py --cache cache/real_body --skeleton config/skeleton.json
python src/poser/checklag.py --cache cache/real_body

# 3  baseline: real data only, every recording once as the test case
python src/poser/train.py --cache cache/real_body --skeleton config/skeleton.json \
    --loro --epochs 20 --lr 5e-4 --patience 8 --seed 0 --out models/base_s0

# 4  pre-train on synthetic, fine-tune on real
python src/poser/train.py --cache cache/synth_body --skeleton config/skeleton.json \
    --epochs 1 --lr 5e-4 --seed 0 --out models/pre_s0
python src/poser/train.py --cache cache/real_body --skeleton config/skeleton.json \
    --loro --epochs 20 --lr 2e-4 --patience 8 --seed 0 \
    --init models/pre_s0/best.pt --out models/ft_s0

# 5  paired comparison
python src/poser/compare.py models/base_s0 models/ft_s0
```

Generating synthetic data requires the AMASS sequences and an SMPL-H body model,
neither of which may be redistributed. `python src/synth/check_setup.py` reports
what is missing and where it is expected.

```bash
python src/synth/run_pipeline.py --sample 600 --max-seconds 60 --foot-impacts
```

## Dashboard

```bash
python src/poser/npz_to_dashboard.py --models models/ft_s0 \
    --cache cache/real_body --data data/processed
cd dashboard && python serve.py
```

Choose the project root, pick a recording, and compare models under
*Evaluation*. Predictions of body-frame models are rotated back into the world
frame using the ground-truth heading so the skeleton lines up with the video;
this is flagged as `heading_from_gt` in the metrics file, because the heading is
not something the model predicts.

## Controls and limitations

`train.py --scramble-targets` pairs each recording's sensor data with another
recording's poses. Input and target distributions are unchanged and only the
relation between them is destroyed, so if pre-training still helps, the gain is
a better weight initialisation rather than transferred knowledge. Together with
`--max-recordings` this separates the amount of pre-training from its content.

Global position and heading are deliberately not predicted: without a trunk
sensor and without a magnetometer neither is observable, and a model that
outputs them outputs guesses. Evaluation is pelvis-centred and PA-MPJPE is
reported alongside MPJPE so the share of the error that is pure rotation stays
visible.

Ground truth is MediaPipe rather than a marker system. Its bone lengths
fluctuate and the side facing away from the camera is measured systematically
shorter (in the canonical skeleton the left thigh comes out at 412 mm and the
right at 342 mm). Since training and evaluation use the same targets this does
not distort the comparison, but it bounds how good the targets can be.

## Data sources

AMASS (Mahmood et al., ICCV 2019) and the SMPL-H body model, both under their
respective licences from the Max Planck Institute for Intelligent Systems.
Neither is included here.
