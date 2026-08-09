"""Time offset between sensor and pose, per sensor.

Two independent signals are correlated against the pose and the peak correlation
is reported alongside the lag. Without it there is no way to tell a real offset
from a spurious maximum: what must coincide at touchdown is the impact-band
energy and the magnitude of linear acceleration on the sensor side, and the
second derivative of the corresponding joint on the pose side.

    python checklag.py --cache cache/real_body
"""
import argparse
from pathlib import Path

import numpy as np

from config import FEAT_PER_SENSOR, LEAF_JOINTS, SENSORS


def xcorr_lag(a, b, fps, max_lag=1.2):
    """Best k such that a[i] coincides with b[i+k]. Returns seconds, peak
    correlation and the correlation at zero offset."""
    n = min(len(a), len(b))
    a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    sa, sb = a.std(), b.std()
    if sa < 1e-9 or sb < 1e-9:
        return 0.0, 0.0, 0.0
    L = int(max_lag * fps)
    cs = []
    for k in range(-L, L + 1):
        x, y = a[L:n - L], b[L + k:n - L + k]
        cs.append(float(np.dot(x - x.mean(), y - y.mean()) /
                        (len(x) * x.std() * y.std() + 1e-12)))
    cs = np.array(cs)
    i = int(np.argmax(cs))
    return (i - L) / fps, float(cs[i]), float(cs[L])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/seg")
    ap.add_argument("--fps", type=float, default=50.0)
    a = ap.parse_args()

    print(f"{'recording':10s}{'Sensor':13s}{'lag':>9s}{'peak':>8s}"
          f"{'at 0':>8s}   verdict")
    for p in sorted(Path(a.cache).glob("*.npz")):
        z = np.load(p, allow_pickle=True)
        X, Y = z["X"].astype(float), z["Y"].astype(float)
        name = str(z["name"])
        rows = []
        for i, s in enumerate(SENSORS):
            o = i * FEAT_PER_SENSOR
            hb = X[:, o + 11]
            lin = np.linalg.norm(X[:, o + 3:o + 6], axis=1)
            j = LEAF_JOINTS[i]
            acc = np.linalg.norm(np.diff(Y[:, j], n=2, axis=0), axis=1)
            acc = np.r_[acc, acc[-1], acc[-1]]
            r = [xcorr_lag(hb, acc, a.fps), xcorr_lag(lin, acc, a.fps)]
            best = max(r, key=lambda t: t[1])
            rows.append((s, best))
        med = float(np.median([b[0] for _, b in rows]))
        peak = float(np.median([b[1] for _, b in rows]))
        for k, (s, (lag, pk, z0)) in enumerate(rows):
            if pk < 0.12:
                note = "correlation too weak, value meaningless"
            elif abs(lag) <= 0.06:
                note = "ok"
            elif pk - z0 < 0.05:
                note = "shifting gains little, probably fine"
            else:
                note = f"real offset, at 0 only {z0:+.2f} instead of {pk:+.2f}"
            print(f"{name if k == 0 else '':10s}{s:13s}{lag:+9.2f}{pk:8.2f}"
                  f"{z0:8.2f}   {note}")
        v = ("ok" if abs(med) <= 0.06 or peak < 0.12
             else f"all four shifted by {med:+.2f} s - check synchronisation")
        print(f"{'':10s}{'--> overall':13s}{med:+9.2f}{peak:8.2f}{'':8s}   {v}\n")

    print("How to read this: 'peak' is the highest correlation, 'at 0' the one")
    print("without any shift. Below a peak of 0.12 the search only found noise and")
    print("the lag says nothing. If all four sensors of a recording are shifted by")
    print("the same amount, the cause is video-to-sensor synchronisation, not a")
    print("single band.")


if __name__ == "__main__":
    main()
