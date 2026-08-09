"""Scale an AMASS sequence to the body proportions of the real recordings.

SMPL and MediaPipe do not mean the same thing by "hip": SMPL places the joint in
the femoral head, MediaPipe further out. Measured on video1 the MediaPipe hip
points are about 20 cm apart, the SMPL ones only 12. Left unchanged, synthetic
recordings would carry a differently built skeleton than the real ones.

Every bone is therefore scaled to the length it has in the real recordings.
Directions, and hence the motion, stay unchanged. Targets are averaged between
left and right because MediaPipe systematically underestimates the side facing
away from the camera.
"""
from pathlib import Path

import numpy as np
import pandas as pd

TREE = [
    (-1, 23, "pelvis_hip"), (-1, 24, "pelvis_hip"),
    (-1, 11, "pelvis_shoulder"), (-1, 12, "pelvis_shoulder"),
    (23, 25, "hip_knee"), (24, 26, "hip_knee"),
    (25, 27, "knee_ankle"), (26, 28, "knee_ankle"),
    (11, 13, "shoulder_elbow"), (12, 14, "shoulder_elbow"),
    (13, 15, "elbow_wrist"), (14, 16, "elbow_wrist"),
    (-2, 0, "neck_nose"),
]
FOOT = [(27, 29, "ankle_heel"), (28, 30, "ankle_heel"),
        (27, 31, "ankle_toe"), (28, 32, "ankle_toe")]

HEAD_COPIES = range(1, 11)          # Gesichtspunkte folgen der Nase
HAND_L, HAND_R = (17, 19, 21), (18, 20, 22)


def _pel(P):
    return (P[:, 23] + P[:, 24]) / 2.0


def _mid_sh(P):
    return (P[:, 11] + P[:, 12]) / 2.0


def measure(P: np.ndarray) -> dict:
    acc = {}
    for par, ch, name in TREE + FOOT:
        a = _pel(P) if par == -1 else _mid_sh(P) if par == -2 else P[:, par]
        L = float(np.median(np.linalg.norm(P[:, ch] - a, axis=1)))
        acc.setdefault(name, []).append(L)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def targets_from_recording(folder) -> dict:
    folder = Path(folder)
    files = sorted(folder.glob("*_gt_3d.csv"))
    if not files:
        raise SystemExit(f"Keine *_gt_3d.csv in {folder}")
    df = pd.read_csv(files[0])
    df.columns = [c.strip().lower() for c in df.columns]
    need = [f"j{j}_{a}" for j in range(33) for a in "xyz"]
    if any(c not in df.columns for c in need):
        raise SystemExit(f"{files[0].name} does not contain all 33 joints")
    P = np.stack([df[[f"j{j}_x", f"j{j}_y", f"j{j}_z"]].to_numpy(float)
                  for j in range(33)], axis=1)
    t = measure(P)
    t["_source"] = files[0].name
    return t


def apply(P: np.ndarray, targets: dict) -> np.ndarray:
    out = np.array(P, dtype=float, copy=True)

    def stretch(parent_old, parent_new, ch, name):
        v = P[:, ch] - parent_old
        L = np.linalg.norm(v, axis=1, keepdims=True)
        out[:, ch] = parent_new + v * (targets[name] / np.maximum(L, 1e-9))

    pel_old, pel_new = _pel(P), _pel(out)          # the pelvis stays where it is
    for par, ch, name in TREE:
        if par == -1:
            stretch(pel_old, pel_new, ch, name)
    for chain in ((23, 25), (24, 26), (11, 13), (12, 14),
                  (25, 27), (26, 28), (13, 15), (14, 16)):
        par, ch = chain
        name = dict((c, n) for p, c, n in TREE)[ch]
        stretch(P[:, par], out[:, par], ch, name)
    stretch(_mid_sh(P), _mid_sh(out), 0, "neck_nose")
    for par, ch, name in FOOT:
        stretch(P[:, par], out[:, par], ch, name)

    for m in HEAD_COPIES:
        out[:, m] = out[:, 0]
    for m in HAND_L:
        out[:, m] = out[:, 15]
    for m in HAND_R:
        out[:, m] = out[:, 16]
    return out


def report(before: dict, after: dict, targets: dict) -> str:
    rows = ["  bone                    AMASS  target   after",
            "  " + "-" * 44]
    for name in ["pelvis_hip", "pelvis_shoulder", "hip_knee", "knee_ankle",
                 "shoulder_elbow", "elbow_wrist", "neck_nose"]:
        if name in targets:
            rows.append(f"  {name:20s} {before.get(name,0)*100:6.1f} "
                        f"{targets[name]*100:6.1f} {after.get(name,0)*100:7.1f}")
    return "\n".join(rows)
