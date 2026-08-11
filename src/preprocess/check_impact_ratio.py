"""Test 2: ankle-to-wrist impact energy, to rule out a swapped sensor mapping.

This reads only the aligned IMU files, never the pose, so it is independent of
MediaPipe. That is the point: if the pose is bad, test 1 and the calibration
residual are both compromised, but this one still works.

In jumping, the ankles take the landing impact and the wrists do not. The ratio

    ankle impact energy / wrist impact energy

is therefore well above 1 for every correctly mapped recording. If a wrist and
an ankle sensor were swapped in the file naming, the ratio for that side drops
below 1 while the other side stays high -- a swap shows up as a side asymmetry,
not just a low overall number.

Impact energy is the RMS of the high-frequency part of |acc|, which isolates
the landing transient and is insensitive to orientation and to gravity.

Usage
    python src/preprocess/check_impact_ratio.py data/processed/video*
    python src/preprocess/check_impact_ratio.py <folders> --skip 120
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SENSORS = ["left_ankle", "right_ankle", "left_wrist", "right_wrist"]
HP_CUTOFF = 5.0        # Hz; below this is body motion, above is impact


def highpass_rms(mag, fs, cutoff=HP_CUTOFF):
    """RMS of |acc| after removing everything below `cutoff` with a Gaussian.

    Subtracting a low-passed copy is a zero-phase high pass and needs no filter
    design, so there is nothing to go unstable on an odd sample rate.
    """
    sigma = max(1.0, fs / (2 * np.pi * cutoff))
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(mag, (r, r), mode="edge")
    low = np.convolve(pad, k, mode="valid")
    hp = mag - low
    return float(np.sqrt(np.mean(hp ** 2))), hp


def analyse(folder: Path, skip_s=0.0, suffix="_aligned"):
    out = {}
    for s in SENSORS:
        p = folder / f"{s}{suffix}.csv"
        if not p.exists():
            out[s] = None
            continue
        d = pd.read_csv(p, usecols=["t", "acc_x", "acc_y", "acc_z"])
        t = d["t"].to_numpy(float)
        acc = d[["acc_x", "acc_y", "acc_z"]].to_numpy(float)
        ok = np.isfinite(acc).all(axis=1) & np.isfinite(t)
        t, acc = t[ok], acc[ok]
        if len(t) < 100:
            out[s] = None
            continue
        if skip_s > 0:
            m = t >= (t[0] + skip_s)
            if m.sum() > 100:
                t, acc = t[m], acc[m]
        fs = 1.0 / np.median(np.diff(t[:2000]))
        mag = np.linalg.norm(acc, axis=1)
        rms, hp = highpass_rms(mag, fs)
        out[s] = {
            "fs": round(float(fs), 2),
            "n": int(len(t)),
            "impact_rms": rms,
            "peak": float(np.percentile(np.abs(hp), 99.9)),
            "mean_mag": float(np.mean(mag)),
        }
    return out


def ratios(m):
    def g(s):
        return m[s]["impact_rms"] if m.get(s) else None

    la, ra, lw, rw = g("left_ankle"), g("right_ankle"), g("left_wrist"), g("right_wrist")
    r = {}
    if la and ra and lw and rw:
        r["ankle_over_wrist"] = (la + ra) / max(lw + rw, 1e-12)
    if la and lw:
        r["left_ankle_over_left_wrist"] = la / max(lw, 1e-12)
    if ra and rw:
        r["right_ankle_over_right_wrist"] = ra / max(rw, 1e-12)
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--skip", type=float, default=0.0,
                    help="drop this many seconds from the start of each sensor")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    res = {}
    for f in a.folders:
        p = Path(f)
        if not p.is_dir():
            continue
        m = analyse(p, a.skip)
        res[p.name] = {"sensors": m, "ratios": ratios(m)}

    print(f"{'recording':12s} {'LA':>8s} {'RA':>8s} {'LW':>8s} {'RW':>8s} "
          f"{'A/W':>7s} {'L A/W':>7s} {'R A/W':>7s}")
    print("-" * 72)
    for name, v in res.items():
        m, r = v["sensors"], v["ratios"]

        def c(s):
            return f"{m[s]['impact_rms']:8.3f}" if m.get(s) else f"{'-':>8s}"

        def rr(k):
            return f"{r[k]:7.2f}" if k in r else f"{'-':>7s}"

        print(f"{name:12s} {c('left_ankle')} {c('right_ankle')} "
              f"{c('left_wrist')} {c('right_wrist')} "
              f"{rr('ankle_over_wrist')} {rr('left_ankle_over_left_wrist')} "
              f"{rr('right_ankle_over_right_wrist')}")

    print("\nA/W well above 1 is correct. A/W below 1 on one side only points to "
          "a wrist/ankle swap\non that side. A/W near 1 on both sides means the "
          "sensors cannot be told apart,\nwhich is a different problem from a swap.")

    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))
        print(f"\nwritten: {a.out}")


if __name__ == "__main__":
    main()
