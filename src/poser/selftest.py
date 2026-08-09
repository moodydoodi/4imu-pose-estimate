"""Data checks before training. Does not require torch.

Verifies that the skeleton is complete and plausible, that the parameterisation
reproduces the pose exactly, what the rigid-skeleton floor is, that the features
are finite and correctly scaled, and that sensors and pose are time aligned.

    python selftest.py --cache cache/real_body --skeleton config/skeleton.json
"""
import argparse
from pathlib import Path

import numpy as np

import skeleton as SK
from config import FEAT_PER_SENSOR, JOINT_NAMES, LEAF_JOINTS, SENSORS


def lag_check(X, Y, fps, max_lag=1.0):
    """Coarse check on the time offset: impact energy at the ankle must
    coincide with the acceleration of the same joint in the pose. More than
    about 0.1 s indicates a synchronisation problem."""
    best = {}
    for i, s in enumerate(SENSORS):
        hb = X[:, i * FEAT_PER_SENSOR + 11]
        j = LEAF_JOINTS[i]
        a = np.linalg.norm(np.diff(Y[:, j], n=2, axis=0), axis=1)
        a = np.r_[a, a[-1], a[-1]]
        n = min(len(hb), len(a))
        hb, a = hb[:n] - hb[:n].mean(), a[:n] - a[:n].mean()
        L = int(max_lag * fps)
        cs = [np.corrcoef(hb[L:n-L], a[L+k:n-L+k])[0, 1] for k in range(-L, L + 1)]
        k = int(np.nanargmax(cs))
        best[s] = ((k - L) / fps, float(cs[k]))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/real")
    ap.add_argument("--skeleton", default="config/skeleton.json")
    ap.add_argument("--fps", type=float, default=50.0)
    a = ap.parse_args()

    canon, per = SK.load_skeleton(a.skeleton)
    print("1. skeleton")
    bad = [i for i in range(12) if not (0.05 < canon[i] < 0.8)]
    for i in range(12):
        print(f"   {JOINT_NAMES[i+1]:14s}{canon[i]*1000:7.0f} mm"
              + ("   <-- implausible" if i in bad else ""))
    print("   " + ("FAILED" if bad else "ok"))

    files = sorted(Path(a.cache).glob("*.npz"))
    if not files:
        raise SystemExit(f"No cache under {a.cache}. Run prepare.py first.")
    print(f"\n{len(files)} recordings in the cache\n")

    print(f"{'recording':14s}{'Frames':>8s}{'exact':>9s}{'floor':>9s}"
          f"{'mean pose':>12s}{'|g|':>7s}{'ankle/wrist':>15s}{'lag s':>11s}")
    for p in files[:12]:
        z = np.load(p, allow_pickle=True)
        X, Y, D = z["X"].astype(float), z["Y"].astype(float), z["D"].astype(float)
        _, Ltrue = SK.bone_dirs_and_lengths(Y)
        exact = SK.mpjpe(SK.forward(D, Ltrue), Y)
        floor = SK.mpjpe(SK.forward(D, canon), Y)
        mean = SK.mpjpe(np.repeat(Y.mean(0)[None], len(Y), 0), Y)
        gn = np.mean([np.linalg.norm(X[:, i*FEAT_PER_SENSOR:i*FEAT_PER_SENSOR+3], axis=1).mean()
                      for i in range(4)])
        hb = [np.percentile(X[:, i*FEAT_PER_SENSOR+11], 95) for i in range(4)]
        ratio = (hb[2] + hb[3]) / max(hb[0] + hb[1], 1e-9)
        lags = lag_check(X, Y, a.fps)
        lag = np.median([v[0] for v in lags.values()])
        print(f"{str(z['name'])[:14]:14s}{len(X):8d}{exact:9.2f}{floor:9.1f}"
              f"{mean:12.1f}{gn:7.3f}{ratio:15.1f}{lag:11.2f}")
        if not np.isfinite(X).all():
            print("   WARNING: non-finite features")

    print("\nHow to read this:")
    print("  exact       must be ~0, otherwise the parameterisation does not fit the pose")
    print("  floor       best a rigid canonical skeleton allows")
    print("  mean pose   reachable without any model; anything above it is worthless")
    print("  |g|         must be 1.000")
    print("  ankle/wrist impact energy ratio; well above 1, otherwise the sensor")
    print("              assignment is wrong")
    print("  lag         seconds between sensor and pose; above 0.1 s indicates a")
    print("              synchronisation problem")


if __name__ == "__main__":
    main()
