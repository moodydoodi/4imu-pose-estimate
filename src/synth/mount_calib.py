"""Measure how a sensor band sits on a limb.
Per sensor, two body-fixed directions in the device frame: the bone axis b from
the specific force at rest, and the flexion axis f from the principal axis of
the angular rate covariance. The rotation between the real and the synthetic
result is the mounting rotation; feeds synth_imu.py --mount.

    python mount_calib.py <recording folder> [--suffix _aligned] [--out mount.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def lowpass(x, fs, cutoff):
    """Zero-phase Gaussian low pass, without scipy."""
    sigma = max(1.0, fs / (2 * np.pi * cutoff))
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(x, ((r, r), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, i], k, mode="valid")
                     for i in range(x.shape[1])], axis=1)

SENSORS = ["left_wrist", "right_wrist", "left_ankle", "right_ankle"]
# sensor -> (proximal, joint, distal) in MediaPipe indices
ANGLE_JOINTS = {"left_wrist": (11, 13, 15), "right_wrist": (12, 14, 16),
                "left_ankle": (23, 25, 27), "right_ankle": (24, 26, 28)}


def resting_gravity(acc, gyr, fs=200.0, frac=0.05):
    n = int(fs)
    nb = len(acc) // n
    A = acc[:nb * n].reshape(nb, n, 3)
    G = gyr[:nb * n].reshape(nb, n, 3)
    score = np.linalg.norm(G, axis=2).std(1) + 10 * np.linalg.norm(A, axis=2).std(1)
    idx = np.argsort(score)[:max(3, int(nb * frac))]
    return np.median(A[idx].reshape(-1, 3), axis=0)


def joint_angle(P, trip):
    a, b, c = trip
    u = P[:, a] - P[:, b]
    v = P[:, c] - P[:, b]
    u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-9
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return np.arccos(np.clip((u * v).sum(1), -1, 1))


def analyse(folder: Path, suffix="_aligned", lag=0.0):
    gts = sorted(folder.glob("*_gt_3d.csv"))
    g = pd.read_csv(gts[0])
    g.columns = [c.strip().lower() for c in g.columns]
    tg = g["time"].to_numpy(float)
    P = np.stack([g[[f"j{j}_x", f"j{j}_y", f"j{j}_z"]].to_numpy(float)
                  for j in range(33)], axis=1)

    out = {}
    for s in SENSORS:
        d = pd.read_csv(folder / f"{s}{suffix}.csv")
        d.columns = [c.strip().lower() for c in d.columns]
        t = d["t"].to_numpy(float)
        acc = d[["acc_x", "acc_y", "acc_z"]].to_numpy(float)
        gyr = d[["gyr_x", "gyr_y", "gyr_z"]].to_numpy(float)
        fs = 1.0 / np.median(np.diff(t[:2000]))

        # 1) bone axis: opposite the specific force at rest
        gv = resting_gravity(acc, gyr, fs)
        b = -gv / np.linalg.norm(gv)

        # 2) flexion axis: the limb moves in the sagittal plane, so gravity
        #    sweeps out a plane whose normal is the medio-lateral axis.
        gl = lowpass(acc, fs, 2.0)
        gl = gl / (np.linalg.norm(gl, axis=1, keepdims=True) + 1e-9)
        Cg = np.cov((gl - gl.mean(0)).T)
        val, vec = np.linalg.eigh(Cg)          # ascending
        f = vec[:, 0]                          # smallest spread = the normal
        sharp = float(np.sqrt(val[1] / max(val[0], 1e-12)))
        f = f - np.dot(f, b) * b
        f /= np.linalg.norm(f)

        # stability: first half against second half
        fh = []
        for sl in (slice(0, len(gl) // 2), slice(len(gl) // 2, None)):
            c = np.linalg.eigh(np.cov((gl[sl] - gl[sl].mean(0)).T))[1][:, 0]
            c = c - np.dot(c, b) * b
            fh.append(c / np.linalg.norm(c))
        stab = float(np.degrees(np.arccos(np.clip(abs(fh[0] @ fh[1]), -1, 1))))

        # 3) sign: flexion counts positive. The direction is already fixed, only

        #    the sign is open - a weak but, over tens of thousands of samples,

        #    stable correlation is enough for that.
        th = joint_angle(P, ANGLE_JOINTS[s])
        dth = np.gradient(th, tg)
        W = np.stack([np.interp(tg + lag, t, gyr[:, k]) for k in range(3)], axis=1)
        m = np.isfinite(dth) & np.isfinite(W).all(1)
        r = float(np.corrcoef(np.deg2rad(W[m]) @ f, dth[m])[0, 1])
        if r < 0:
            f, r = -f, -r
        r2 = sharp
        third = np.cross(f, b)
        B = np.stack([f, b, third], axis=1)          # anatomical -> device
        out[s] = {"bone_in_device": b.tolist(), "flex_in_device": f.tolist(),
                  "basis": B.tolist(), "flex_corr": r, "sharpness": float(r2),
                  "halves_disagree_deg": stab, "resting_acc": gv.tolist()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--suffix", default="_aligned")
    ap.add_argument("--lag", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = analyse(Path(a.folder), a.suffix, a.lag)
    print(f"{'sensor':13s} {'bone axis b':>26s} {'flexion axis f':>26s} "
          f"{'r':>6s} {'sharpness':>9s} {'halves':>9s}")
    for s, v in res.items():
        b, f = np.array(v["bone_in_device"]), np.array(v["flex_in_device"])
        print(f"{s:13s} ({b[0]:6.2f},{b[1]:6.2f},{b[2]:6.2f})     "
              f"({f[0]:6.2f},{f[1]:6.2f},{f[2]:6.2f})  {v['flex_corr']:6.2f} "
              f"{v['sharpness']:9.1f} {v['halves_disagree_deg']:7.1f} deg")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))
        print(f"\nwritten: {a.out}")


if __name__ == "__main__":
    main()
