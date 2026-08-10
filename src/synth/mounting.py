"""Rotations between the anatomical frame of a limb and the device frame."""
import json
from pathlib import Path

import numpy as np

SENSOR_ORDER = ["left_wrist", "right_wrist", "left_ankle", "right_ankle"]

DEFAULT_BONE_AXIS = {
    "left_wrist":  [0.06, -1.00, -0.06],
    "right_wrist": [0.06, 0.02, 1.00],
    "left_ankle":  [0.09, -0.01, 1.00],
    "right_ankle": [0.14, 0.04, -0.99],
}


def _perp(u):
    a = np.zeros(3)
    a[int(np.argmin(np.abs(u)))] = 1.0
    v = np.cross(u, a)
    return v / (np.linalg.norm(v) + 1e-12)


def _roty(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def load_bone_axes(path=None) -> dict:
    if path and Path(path).exists():
        d = json.loads(Path(path).read_text())
        return {s: d[s]["bone_in_device"] for s in SENSOR_ORDER if s in d}
    return DEFAULT_BONE_AXIS


def mount_matrix(sensor: str, bone_axes: dict, roll_deg: float) -> np.ndarray:
    b = np.asarray(bone_axes[sensor], float)
    b = b / (np.linalg.norm(b) + 1e-12)
    p = _perp(b)
    A = np.stack([p, b, np.cross(p, b)], axis=1)          # column 1 = b
    return A @ _roty(np.deg2rad(roll_deg))


def draw_rolls(rng, fixed=None) -> dict:
    """-> roll about the bone axis per sensor, in degrees."""
    if fixed:
        return {s: float(fixed.get(s, 0.0)) for s in SENSOR_ORDER}
    return {s: float(rng.uniform(0.0, 360.0)) for s in SENSOR_ORDER}


# --------------------------------------------------------------- _mp_spatial
TARGET_BASIS_RIGHT = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], float)
TARGET_BASIS_LEFT = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)


def calibration_window(acc_by_sensor: dict, t_by_sensor: dict, fs: float,
                       duration_s: float = 5.0):
    t_min = max(t[0] for t in t_by_sensor.values())
    t_max = min(t[-1] for t in t_by_sensor.values())
    t_common = np.arange(t_min, t_max, 1.0 / fs)
    if len(t_common) < int(fs * duration_s):
        return {s: (0, min(len(t), int(fs * duration_s))) for s, t in t_by_sensor.items()}
    total = np.zeros_like(t_common)
    win = max(2, int(fs))
    for s, acc in acc_by_sensor.items():
        mag = np.linalg.norm(acc, axis=1)
        c = np.cumsum(np.r_[0.0, mag])
        c2 = np.cumsum(np.r_[0.0, mag ** 2])
        n = len(mag)
        lo = np.clip(np.arange(n) - win // 2, 0, n)
        hi = np.clip(np.arange(n) + win // 2, 0, n)
        cnt = np.maximum(hi - lo, 1)
        var = (c2[hi] - c2[lo]) / cnt - ((c[hi] - c[lo]) / cnt) ** 2
        total += np.interp(t_common, t_by_sensor[s], var)
    blk = int(fs * duration_s)
    k = np.cumsum(np.r_[0.0, total])
    roll = (k[blk:] - k[:-blk]) / blk
    centre = t_common[int(np.argmin(roll)) + blk // 2]
    out = {}
    for s, t in t_by_sensor.items():
        a = int(np.searchsorted(t, centre - duration_s / 2))
        b = int(np.searchsorted(t, centre + duration_s / 2))
        out[s] = (a, max(b, a + 2))
    return out


def calib_matrix(acc_calib: np.ndarray, sensor: str):
    mean_acc = acc_calib.mean(axis=0)
    u_X = mean_acc / np.linalg.norm(mean_acc)
    flipped = False
    v_Z = np.array([0.0, 0.0, 1.0])
    u_Y = np.cross(u_X, v_Z)
    n = np.linalg.norm(u_Y)
    u_Y = np.array([0.0, 1.0, 0.0]) if n < 1e-6 else u_Y / n
    u_Z = np.cross(u_X, u_Y)
    u_Z /= np.linalg.norm(u_Z)
    R_imu = np.column_stack((u_X, u_Y, u_Z))
    R_target = TARGET_BASIS_LEFT if "left" in sensor else TARGET_BASIS_RIGHT
    R = R_target @ R_imu.T
    assert np.linalg.det(R) > 0, f"{sensor}: R_calib is a reflection"
    return R, flipped


def to_mp_spatial(acc: np.ndarray, gyr: np.ndarray, acc_calib: np.ndarray,
                  sensor: str):
    R, flipped = calib_matrix(acc_calib, sensor)
    a, g = acc.copy(), gyr.copy()
    if flipped:
        a[:, 0] *= -1; a[:, 2] *= -1
        g[:, 0] *= -1; g[:, 2] *= -1
    return (R @ a.T).T, (R @ g.T).T

# Measured mounting rotations. Spread across recordings and total angle K:
#   right_ankle    6 recordings    9 deg spread    86 deg
#   left_ankle     9 recordings   14 deg spread   170 deg
#   right_wrist   13 recordings   41 deg spread   110 deg
#   left_wrist    14 recordings   39 deg spread   164 deg
MOUNT_K = {
    "right_wrist": [
        [
            0.225237198676222,
            -0.9653333733180242,
            -0.13190785682032793
        ],
        [
            0.20077694780307181,
            -0.08649315632069919,
            0.9758112272056342
        ],
        [
            -0.953392270559905,
            -0.24627304413623322,
            0.17433521207289754
        ]
    ],
    "left_wrist": [
        [
            -0.9544400081710848,
            0.12178647134901147,
            0.2724193939475258
        ],
        [
            0.07472633104810067,
            0.9813836996604756,
            -0.1769237391895334
        ],
        [
            -0.28889487058525404,
            -0.14850619326167092,
            -0.9457725224981203
        ]
    ],
    "right_ankle": [
        [
            0.9300314226205539,
            0.017957845329817763,
            0.36704096328543556
        ],
        [
            -0.3668078720001427,
            0.10574721242684701,
            0.924267013423436
        ],
        [
            -0.022215714643388947,
            -0.9942308800550619,
            0.10493530944263653
        ]
    ],
    "left_ankle": [
        [
            -0.9522699970086769,
            -0.034567493512980524,
            -0.3032934901861947
        ],
        [
            -0.297777853108803,
            -0.11339621552202693,
            0.9478763888309448
        ],
        [
            -0.06715804490160206,
            0.9929483303261616,
            0.09769037981030801
        ]
    ]
}


def mount_matrix_calibrated(sensor: str, bone_axes: dict) -> np.ndarray:
    M0 = mount_matrix(sensor, bone_axes, 0.0)
    K = MOUNT_K.get(sensor)
    return M0 if K is None else np.asarray(K, float) @ M0
