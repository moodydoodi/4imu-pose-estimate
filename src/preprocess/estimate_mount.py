"""Mounting rotation per recording and sensor, by Kabsch fit.
Pairs the gravity direction in the anatomical frame (from the pose) with the
one in the device frame (from the accelerometer) over the whole recording. The
residual says how reliable the fit is.

    python estimate_mount.py data/processed/video1 [--lag 0.10]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SENSORS = ["left_wrist", "right_wrist", "left_ankle", "right_ankle"]
# sensor -> (proximal, middle, distal joint) in MediaPipe indices
CHAIN = {"left_wrist": (11, 13, 15), "right_wrist": (12, 14, 16),
         "left_ankle": (23, 25, 27), "right_ankle": (24, 26, 28)}
UP_WORLD = np.array([0.0, -1.0, 0.0])       # MediaPipe: +y is down


def lowpass(x, fs, cutoff):
    sigma = max(1.0, fs / (2 * np.pi * cutoff))
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(x, ((r, r), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, i], k, mode="valid")
                     for i in range(x.shape[1])], axis=1)


def anatomical_frames(P, trip):
    prox, mid, dist = trip
    upper = P[:, mid] - P[:, prox]
    lower = P[:, dist] - P[:, mid]
    y = lower / (np.linalg.norm(lower, axis=1, keepdims=True) + 1e-9)
    x = np.cross(upper, lower)
    q = np.linalg.norm(x, axis=1) / (np.linalg.norm(upper, axis=1)
                                     * np.linalg.norm(lower, axis=1) + 1e-9)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    x = x - (x * y).sum(1, keepdims=True) * y
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=2), q       # q = sin(joint angle)


def kabsch(Aset, Bset, w=None):
    if w is None:
        w = np.ones(len(Aset))
    H = (Bset * w[:, None]).T @ Aset
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, d])
    return U @ D @ Vt


def analyse(folder: Path, lag=0.0, suffix="_aligned"):
    gts = sorted(folder.glob("*_gt_3d.csv"))
    if not gts:
        raise SystemExit(f"no *_gt_3d.csv in {folder}")
    g = pd.read_csv(gts[0])
    g.columns = [c.strip().lower() for c in g.columns]
    tg = g["time"].to_numpy(float)
    P = np.stack([g[[f"j{j}_x", f"j{j}_y", f"j{j}_z"]].to_numpy(float)
                  for j in range(33)], axis=1)

    out = {}
    for s in SENSORS:
        p = folder / f"{s}{suffix}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p, usecols=["t", "acc_x", "acc_y", "acc_z"])
        t = d["t"].to_numpy(float)
        acc = d[["acc_x", "acc_y", "acc_z"]].to_numpy(float)
        fs = 1.0 / np.median(np.diff(t[:2000]))

        a_lp = lowpass(acc, fs, 1.0)
        mag = np.linalg.norm(a_lp, axis=1)
        u_dev_all = a_lp / (mag[:, None] + 1e-9)
        u_dev = np.stack([np.interp(tg + lag, t, u_dev_all[:, k]) for k in range(3)], axis=1)
        m_ok = np.interp(tg + lag, t, mag)

        S, q = anatomical_frames(P, CHAIN[s])
        up_anat = np.einsum("tji,j->ti", S, UP_WORLD)      # S^T @ up

        keep = (np.abs(m_ok - 9.80665) < 1.5) & (q > 0.35) & np.isfinite(up_anat).all(1)
        n = int(keep.sum())
        if n < 200:
            out[s] = {"n": n, "note": "too few usable frames"}
            continue
        A, B = up_anat[keep], u_dev[keep]
        A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-9
        B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-9
        R = kabsch(A, B)
        res = np.degrees(np.arccos(np.clip(((R @ A.T).T * B).sum(1), -1, 1)))
        h = n // 2
        R1, R2 = kabsch(A[:h], B[:h]), kabsch(A[h:], B[h:])
        dR = R1.T @ R2
        split = float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
        out[s] = {"R": R.tolist(), "n": n,
                  "residual_deg": float(np.median(res)),
                  "halves_deg": split}
    return out


def axis_angle(R):
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    w, v = np.linalg.eig(R)
    ax = np.real(v[:, np.argmin(np.abs(w - 1))])
    return ax / (np.linalg.norm(ax) + 1e-12), ang


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--lag", type=float, default=0.10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    allres = {}
    for f in a.folders:
        name = Path(f).name
        allres[name] = analyse(Path(f), a.lag)
        print(f"\n=== {name} ===")
        print(f"{'sensor':13s} {'frames':>7s} {'residual':>11s} {'halves':>9s}")
        for s, v in allres[name].items():
            if "R" not in v:
                print(f"{s:13s} {v['n']:7d}   {v.get('note','')}")
                continue
            print(f"{s:13s} {v['n']:7d} {v['residual_deg']:9.1f} deg "
                  f"{v['halves_deg']:7.1f} deg")
    if a.out:
        Path(a.out).write_text(json.dumps(allres, indent=2))
        print(f"\nwritten: {a.out}")


if __name__ == "__main__":
    main()
