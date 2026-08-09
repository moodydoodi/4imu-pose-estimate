"""SMPL-H joint positions from pose parameters, in numpy only.

The usual route is smplx and therefore PyTorch, which is not needed here: joint
positions require neither the surface mesh nor the blend weights, only
J_regressor (52 joints from the 6890 rest vertices) and kintree_table. What
remains is the standard chain - rest pose, per-joint rotation from the axis-angle
vector, then propagation outward from the pelvis.
"""
from pathlib import Path

import numpy as np

_LAYOUTS = [
    "{g}/model.npz",
    "smplh/{g}/model.npz",
    "smplh/SMPLH_{G}.npz",
    "SMPLH_{G}.npz",
    "smplh/model_{g}.npz",
]
_GENDERS = ["neutral", "male", "female"]


def find_model(root) -> tuple:
    root = Path(root)
    for g in _GENDERS:
        for pat in _LAYOUTS:
            p = root / pat.format(g=g, G=g.upper())
            if p.exists():
                return p, g
    for p in sorted(root.rglob("*.npz")):
        try:
            keys = set(np.load(p, allow_pickle=True).files)
        except Exception:
            continue
        if {"J_regressor", "v_template", "kintree_table"} <= keys:
            g = next((x for x in _GENDERS if x in p.as_posix().lower()), "neutral")
            return p, g
    raise SystemExit(
        f"No SMPL-H body model found under {root}.\n"
        f"Expected a model.npz with J_regressor, v_template and "
        f"kintree_table, z.B. unter {root}/neutral/model.npz")


def _rodrigues(rv: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(rv, axis=-1, keepdims=True)
    k = rv / np.maximum(theta, 1e-12)
    K = np.zeros(rv.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    th = theta[..., None]
    I = np.broadcast_to(np.eye(3), K.shape)
    return I + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


class BodyModel:

    def __init__(self, path):
        d = np.load(str(path), allow_pickle=True)
        missing = {"J_regressor", "v_template", "kintree_table"} - set(d.files)
        if missing:
            raise SystemExit(f"{path} fehlen die Felder {sorted(missing)} - "
                             f"this is not an SMPL-H model file.")
        self.J_regressor = np.asarray(d["J_regressor"], dtype=np.float64)
        self.v_template = np.asarray(d["v_template"], dtype=np.float64)
        self.shapedirs = np.asarray(d["shapedirs"], dtype=np.float64) \
            if "shapedirs" in d.files else None
        kt = np.asarray(d["kintree_table"])
        self.parents = kt[0].astype(int).copy()
        self.parents[0] = -1
        self.n_joints = self.J_regressor.shape[0]
        self.path = Path(path)

    def rest_joints(self, betas=None) -> np.ndarray:
        v = self.v_template
        if betas is not None and self.shapedirs is not None:
            n = min(len(betas), self.shapedirs.shape[2])
            if n:
                v = v + self.shapedirs[:, :, :n] @ np.asarray(betas[:n], float)
        return self.J_regressor @ v

    def segment_align(self, child_of: dict, betas=None) -> dict:
        J = self.rest_joints(betas)
        out = {}
        for j, child in child_of.items():
            u = J[child] - J[j]
            u = u / (np.linalg.norm(u) + 1e-12)
            a = np.zeros(3)
            a[int(np.argmin(np.abs(u)))] = 1.0
            x = np.cross(u, a)
            x /= np.linalg.norm(x) + 1e-12
            out[j] = np.stack([x, u, np.cross(x, u)], axis=1)   # rechtshaendig
        return out

    def joints(self, poses: np.ndarray, trans=None, betas=None,
               return_rot=False):
        poses = np.asarray(poses, dtype=np.float64)
        T = len(poses)
        n = self.n_joints
        need = n * 3
        if poses.shape[1] < need:                 # e.g. SMPL with 24 joints
            poses = np.concatenate(
                [poses, np.zeros((T, need - poses.shape[1]))], axis=1)
        R = _rodrigues(poses[:, :need].reshape(T, n, 3))       # (T, n, 3, 3)

        J = self.rest_joints(betas)                            # (n, 3)
        out = np.empty((T, n, 3))
        Rg = np.empty((T, n, 3, 3))
        Rg[:, 0] = R[:, 0]
        out[:, 0] = J[0]
        for i in range(1, n):
            p = self.parents[i]
            Rg[:, i] = Rg[:, p] @ R[:, i]
            out[:, i] = out[:, p] + np.einsum("tij,j->ti", Rg[:, p], J[i] - J[p])

        if trans is not None:
            out = out + np.asarray(trans, float)[:, None, :]
        return (out, Rg) if return_rot else out
