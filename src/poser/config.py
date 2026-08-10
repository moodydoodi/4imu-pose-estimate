"""Shared constants: sensors, skeleton topology, sampling rates, scaling."""
import numpy as np

# --------------------------------------------------------------------- sensors
SENSORS = ["left_wrist", "right_wrist", "left_ankle", "right_ankle"]
ACC_COLS = ["acc_x", "acc_y", "acc_z"]
GYR_COLS = ["gyr_x", "gyr_y", "gyr_z"]
G = 9.80665

# -------------------------------------------------------------------- skeleton
# 13 joints, joint 0 is the pelvis (mid-hip) and the root.
MP_INDEX = {
    0: None,                  # pelvis: mean of MediaPipe 23 and 24
    1: 23, 2: 25, 3: 27,      # left leg
    4: 24, 5: 26, 6: 28,      # right leg
    7: 11, 8: 13, 9: 15,      # left arm
    10: 12, 11: 14, 12: 16,   # right arm
}
PARENTS = [0, 0, 1, 2, 0, 4, 5, 1, 7, 8, 4, 10, 11]
N_JOINTS = 13
N_BONES = 12
JOINT_NAMES = ["pelvis", "hip_l", "knee_l", "ankle_l",
               "hip_r", "knee_r", "ankle_r",
               "shoulder_l", "elbow_l", "wrist_l",
               "shoulder_r", "elbow_r", "wrist_r"]

# Joints carrying a sensor; they get their own output stage.
LEAF_JOINTS = [9, 12, 3, 6]
LEAF_BONES = [9, 12, 3, 6]

# -------------------------------------------------------------------- sampling
FPS = 50.0            # target rate, matches the pose ground truth
WIN = 200             # window length in frames (4 s)
HOP = 20              # training stride
HIGHBAND = (20.0, 90.0)   # impact band

# ----------------------------------------------------------------------- input
# Per sensor: gravity (3), linear acc (3), gyro (3), |acc|, |gyr|, impact band.
FEAT_PER_SENSOR = 12
N_FEAT = FEAT_PER_SENSOR * len(SENSORS)

ACC_SCALE = 30.0
GYR_SCALE = 10.0


def leaf_sensor_pairs():
    return list(zip(SENSORS, LEAF_JOINTS))
