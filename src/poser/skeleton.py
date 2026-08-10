"""13-joint skeleton, bone lengths and forward kinematics."""
import json
from pathlib import Path

import numpy as np

from config import PARENTS, N_JOINTS, N_BONES, MP_INDEX, JOINT_NAMES

assert all(PARENTS[i] < i for i in range(1, N_JOINTS)), \
    "PARENTS must be topologically sorted for the prefix sum to be valid."

UP = np.array([0.0, -1.0, 0.0])   # MediaPipe world frame: +y is down


def mp33_to_13(P33):
    """(T,33,3) MediaPipe world landmarks -> (T,13,3), pelvis-centred."""
    P33 = np.asarray(P33, float)
    pel = (P33[:, 23] + P33[:, 24]) / 2.0
    out = np.empty((len(P33), N_JOINTS, 3))
    out[:, 0] = pel
    for j, mp in MP_INDEX.items():
        if mp is not None:
            out[:, j] = P33[:, mp]
    return out - pel[:, None, :]


def bone_vectors(P13):
    P13 = np.asarray(P13, float)
    return np.stack([P13[:, i] - P13[:, PARENTS[i]] for i in range(1, N_JOINTS)], axis=1)


def bone_dirs_and_lengths(P13):
    V = bone_vectors(P13)
    L = np.linalg.norm(V, axis=2)
    D = np.divide(V, L[..., None], out=np.zeros_like(V), where=L[..., None] > 1e-9)
    return D, L


def forward(D, L):
    """Directions (T,12,3) and lengths (12,) or (T,12) -> positions (T,13,3)."""
    D = np.asarray(D, float)
    L = np.asarray(L, float)
    if L.ndim == 1:
        L = np.broadcast_to(L, (len(D), N_BONES))
    n = np.linalg.norm(D, axis=2, keepdims=True)
    D = np.divide(D, n, out=np.zeros_like(D), where=n > 1e-9)
    P = np.zeros((len(D), N_JOINTS, 3))
    for i in range(1, N_JOINTS):
        P[:, i] = P[:, PARENTS[i]] + D[:, i - 1] * L[:, i - 1, None]
    return P


def canonical_from_recordings(rec_poses, robust=True):
    """-> canonical bone lengths (12,) and per-recording lengths {name: (12,)}."""
    per = {}
    for k, P in rec_poses.items():
        _, L = bone_dirs_and_lengths(P)
        per[k] = np.median(L, axis=0) if robust else L.mean(0)
    canon = np.median(np.stack(list(per.values())), axis=0)
    return canon, per


def save_skeleton(path, canon, per):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({
        "parents": PARENTS,
        "joint_names": JOINT_NAMES,
        "canonical_bone_lengths_m": np.asarray(canon).tolist(),
        "per_recording_bone_lengths_m": {k: np.asarray(v).tolist() for k, v in per.items()},
    }, indent=2))


def load_skeleton(path):
    d = json.loads(Path(path).read_text())
    if d["parents"] != PARENTS:
        raise ValueError("Skeleton file does not match config.PARENTS.")
    canon = np.array(d["canonical_bone_lengths_m"], float)
    if canon.shape != (N_BONES,) or not np.all(canon > 0.01):
        raise ValueError(f"Implausible bone lengths: {canon}")
    per = {k: np.array(v, float) for k, v in d["per_recording_bone_lengths_m"].items()}
    return canon, per


def canonicalize(P13, L_target):
    """(T,13,3) -> same pose rebuilt with bone lengths L_target."""
    D, _ = bone_dirs_and_lengths(P13)
    return forward(D, L_target)


def body_frame(P13):
    """(T,13,3) -> pose with the hip axis on +x, and the rotations used."""
    P13 = np.asarray(P13, float)
    up = np.broadcast_to(UP, (len(P13), 3))
    h = P13[:, 4] - P13[:, 1]
    x = h - (h * up).sum(1, keepdims=True) * up
    n = np.linalg.norm(x, axis=1, keepdims=True)
    x = np.divide(x, n, out=np.tile([1.0, 0.0, 0.0], (len(P13), 1)), where=n > 1e-9)
    z = np.cross(x, up)
    R = np.stack([x, up, z], axis=2)
    return np.einsum("tji,tkj->tki", R, P13), R


def mpjpe(A, B):
    """Mean per-joint position error in mm."""
    return float(np.linalg.norm(np.asarray(A) - np.asarray(B), axis=-1).mean() * 1000)
