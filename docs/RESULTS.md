# Detailed results

All errors in mm, leave-one-recording-out over six recordings, targets on the
canonical skeleton in the body-fixed frame unless stated otherwise.

## 1. Reference frame of the target pose

Identical settings (4.44 M parameters, 20 epochs, lr 5e-4, patience 8); the only
difference is the frame the target pose is expressed in.

| Recording | body MPJPE | baseline | gain | world MPJPE | baseline | gain |
|---|---:|---:|---:|---:|---:|---:|
| video1 | 88.9 | 105.2 | 15.5 % | 128.4 | 137.8 | 6.8 % |
| video2 | 101.6 | 110.1 | 7.7 % | 134.6 | 146.4 | 8.1 % |
| video3 | 84.3 | 99.9 | 15.6 % | 128.3 | 135.8 | 5.5 % |
| video4 | 91.7 | 107.2 | 14.5 % | 132.9 | 136.9 | 2.9 % |
| video5 | 93.5 | 112.0 | 16.5 % | 145.9 | 155.7 | 6.3 % |
| video6 | 87.4 | 97.7 | 10.5 % | 117.9 | 127.0 | 7.2 % |
| **mean** | **91.2** | 105.4 | **13.4 %** | 131.3 | 139.9 | 6.1 % |

The body frame also solves an easier task, so the baselines differ. PA-MPJPE
removes global rotation and is therefore directly comparable between the two:
**70.6 vs 77.7 mm**. The body-frame model predicts the shape of the pose better,
not merely a smaller quantity.

Why: without a magnetometer and without a trunk sensor, rotation about the
vertical is not observable. It showed up at the hips, which sit rigidly 99 mm
from the pelvis yet had 58 mm of error in the world frame - about 31° of pelvis
heading error, inherited by every joint down the chain. Measured on video1,
removing the heading drops the trivial baseline from 135.7 to 103.6 mm, so about
32 mm of the total pose variance is heading alone.

Per-joint effect:

| joint | body | world |
|---|---:|---:|
| hip l / r | 5.3 / 5.2 | 58.1 / 58.2 |
| knee l / r | 80.3 / 84.1 | 119.4 / 113.2 |
| ankle l / r | 153.0 / 165.6 | 170.8 / 158.5 |
| shoulder l / r | 95.7 / 79.4 | 126.4 / 124.5 |
| elbow l / r | 111.7 / 98.3 | 168.6 / 164.7 |
| wrist l / r | 151.3 / 156.2 | 225.2 / 219.7 |

Arms gain heavily, legs barely: the arm error was dominated by trunk
uncertainty, the leg error by actual movement. Ankles are now the worst joints.

## 2. Synthetic pre-training

Paired comparison, three seeds × six test recordings = 18 pairs, bootstrap 95 %
confidence intervals.

| pre-training | steps | Δ MPJPE | 95 % CI | better |
|---|---:|---:|---|---:|
| 120 recordings, 1 epoch | ~118 | **−2.86** | [−4.35, −1.42] | 16/18 |
| 120 recordings, 2 epochs | ~237 | −2.77 | [−4.40, −1.14] | 14/18 |
| 800 recordings, 1 epoch | ~781 | −1.48 | [−3.35, **+0.43**] | 14/18 |

For the first condition PA-MPJPE gives −2.09 [−3.47, −0.89] and PCK@100 gives
+0.89 [+0.12, +1.74], both excluding zero.

The mean over six folds is stable across seeds (92.05 / 92.26 / 92.68, spread
0.32 mm) while single folds vary widely (video4: 93.15 / 102.39 / 93.52). The
effect is about nine times the noise of the mean but would not have been
detectable on a single fold, which is why the paired design was necessary.

**More synthetic data made the effect smaller, not larger.** The benefit decays
monotonically with the number of pre-training steps, and in the 800-recording
condition the interval includes zero. The direction holds across all three seeds
(−0.77 / −1.25 / −2.41 vs −2.73 / −2.29 / −3.55).

Two explanations are confounded in these runs. Either the benefit is a better
weight initialisation that longer pre-training destroys, or the 800-recording
set is worse in content: the 120-recording set was CMU only, the 800-recording
set adds ACCAD, DanceDB, HDM05 and TotalCapture, i.e. crawling, dance and
martial arts, which are further from jumping. `--max-recordings` and
`--scramble-targets` separate the two.

## 3. Quality of the synthetic data

800 recordings, 742 678 frames (4.13 h) against six real recordings, compared in
the 48 model input features. The meaningful yardstick is how far two real
recordings are from each other, since that spread has to be bridged anyway.
Same formula on both sides (mean distance of q05, q50, q95 divided by the
standard deviation of the reference recording):

| | median | mean | p90 | max |
|---|---:|---:|---:|---:|
| synthetic vs real (6 comparisons) | 0.22 | 0.36 | 0.79 | 3.07 |
| real vs real (15 pairs) | 0.22 | 0.28 | 0.53 | 2.00 |

In 69 % of the 48 features the synthetic data is closer to a real recording than
two real recordings are to each other.

The one serious defect is the impact band at the wrists:

| sensor | highband q95 real | synthetic | factor | distance | real vs real |
|---|---:|---:|---:|---:|---:|
| left_wrist | 0.0211 | 0.2185 | 10.4 | 2.19 | 0.09 |
| right_wrist | 0.0206 | 0.1831 | 8.9 | 1.91 | 0.09 |
| left_ankle | 0.1354 | 0.1155 | 0.85 | 0.09 | 0.10 |
| right_ankle | 0.1401 | 0.1322 | 0.94 | 0.09 | 0.10 |

At the ankles the channel is correct once foot impacts are enabled. At the
wrists real recordings carry almost no energy in that band, so a factor of ten
cannot be motion content: the pose is smoothed at 25 Hz and then differentiated
twice, and double differentiation amplifies high frequencies quadratically.

Second largest deviation is the gravity direction at the ankles (grav_z at 0.81
and 0.90 against 0.13 to 0.34 between real recordings). That is a genuine
distribution difference - AMASS foot orientations while walking cover different
angles than jumping does.

## 4. Reference values

| | |
|---|---|
| MediaPipe bone-length jitter | 20.0 mm (19.3 to 21.3) |
| cost of one shared skeleton | 1.1 mm (0.5 to 2.0) |
| trivial mean pose, body frame | 105.4 mm |
| best model | 89.5 mm |

No rigid-skeleton model can go below roughly 20 mm, because MediaPipe's measured
bone lengths vary from frame to frame.

## 5. Checks that were run

**Subject leak.** video1 and video5 are the same person, as are video4 and
video6. Excluding video5 entirely changes video1 from 126.6/137.8 to 128.1/138.9
- 1.5 mm. The leave-one-recording-out numbers are therefore sound.

**Synchronisation.** Cross-correlating impact energy against joint acceleration
gives zero-lag correlations of 0.21 to 0.51 across all 24 sensor-recording
combinations, and in 22 of 24 the gain from shifting is below 0.08. Apparent
offsets of ±0.3 s are side lobes at the jump period, not timing errors.

**Model capacity is not the bottleneck.** Reducing from 4.44 M to 1.33 M
parameters gave 126.6 and 137.6 instead of 124.5 and 130.1 on the comparable
folds. Doubling the number of training windows and strengthening augmentation
changed nothing (91.7 vs 91.2).

**Learning-rate schedule.** With 60 scheduled epochs and early stopping at 15,
OneCycle never reached its decay phase and the optimum fell at epoch 2-3. With
20 epochs validation keeps improving to epoch 10-18.
