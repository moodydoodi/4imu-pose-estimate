"""Reading a recording, resampling onto a common time grid, building windows.

Deliberately tolerant of what the recording folders actually contain: sensor
files start before the video (negative timestamps) and end after it, sampling
rates differ per file, and the suffix is _aligned, _mp_spatial or _segment
depending on the processing stage. Only the interval covered by all five
sources is used.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import skeleton as SK
from config import ACC_COLS, FPS, GYR_COLS, HOP, N_FEAT, SENSORS, WIN
from features import sensor_features


def _rate(t):
    d = np.diff(t[:5000])
    d = d[d > 0]
    return float(1.0 / np.median(d)) if len(d) else 50.0


def read_pose(folder: Path):
    gts = sorted(Path(folder).glob("*_gt_3d.csv"))
    if not gts:
        return None, None
    g = pd.read_csv(gts[0])
    g.columns = [c.strip().lower() for c in g.columns]
    tcol = "time" if "time" in g.columns else ("t" if "t" in g.columns else None)
    if tcol is None:
        return None, None
    need = [f"j{j}_{a}" for j in range(33) for a in "xyz"]
    if any(c not in g.columns for c in need):
        return None, None
    g = g.sort_values(tcol)
    t = g[tcol].to_numpy(float)
    P33 = np.stack([g[[f"j{j}_x", f"j{j}_y", f"j{j}_z"]].to_numpy(float)
                    for j in range(33)], axis=1)
    return t, SK.mp33_to_13(P33)


def load_recording(folder, suffix="_segment", fps=FPS, canon_L=None,
                   frame="world", verbose=False):
    """-> dict with t, X (T,N_FEAT), Y (T,13,3), name; or None."""
    folder = Path(folder)
    tg, P13 = read_pose(folder)
    if tg is None:
        return None

    raw = {}
    for s in SENSORS:
        p = folder / f"{s}{suffix}.csv"
        if not p.exists():
            return None
        d = pd.read_csv(p)
        d.columns = [c.strip().lower() for c in d.columns]
        if any(c not in d.columns for c in ["t"] + ACC_COLS + GYR_COLS):
            return None
        d = d.sort_values("t")
        raw[s] = (d["t"].to_numpy(float),
                  d[ACC_COLS].to_numpy(float), d[GYR_COLS].to_numpy(float))

    t0 = max([tg[0]] + [v[0][0] for v in raw.values()])
    t1 = min([tg[-1]] + [v[0][-1] for v in raw.values()])
    if t1 - t0 < WIN / fps * 2:
        return None
    t_dst = np.arange(t0, t1, 1.0 / fps)

    X = np.empty((len(t_dst), N_FEAT))
    for i, s in enumerate(SENSORS):
        ts, acc, gyr = raw[s]
        fs = _rate(ts)
        if verbose:
            print(f"    {s:13s} {len(ts):8d} samples, {fs:6.1f} Hz")
        X[:, i * 12:(i + 1) * 12] = sensor_features(ts, acc, gyr, t_dst, fs, fps)

    Y = np.stack([np.interp(t_dst, tg, P13[:, j, k])
                  for j in range(13) for k in range(3)], axis=1).reshape(-1, 13, 3)
    Y = Y - Y[:, 0:1]
    if canon_L is not None:
        Y = SK.canonicalize(Y, canon_L)
    if frame == "body":
        Y, _ = SK.body_frame(Y)
    elif frame != "world":
        raise ValueError(f"Unknown frame: {frame}")

    if not np.isfinite(X).all():
        raise ValueError(f"{folder.name}: non-finite features")
    return {"name": folder.name, "t": t_dst, "X": X, "Y": Y}


def windows(rec, win=WIN, hop=HOP):
    T = len(rec["t"])
    st = np.arange(0, max(T - win + 1, 0), hop)
    if len(st) == 0:
        return None, None
    return (np.stack([rec["X"][s:s + win] for s in st]),
            np.stack([rec["Y"][s:s + win] for s in st]))


def find_recordings(root, exclude=()):
    root = Path(root)
    return [d for d in sorted(x for x in root.iterdir() if x.is_dir())
            if d.name not in exclude and sorted(d.glob("*_gt_3d.csv"))]
