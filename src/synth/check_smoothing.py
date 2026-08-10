"""Compare the impact band of synthetic against real recordings.

The smoothing cutoff in synth_imu.py decides how much energy ends up in the
20-90 Hz band that the model reads as impact energy. Too high and the synthetic
data invents impacts, too low and it has none. The current default was measured
against six recordings; with a new dataset it should be measured again.

    python src/synth/check_smoothing.py --real data/processed \\
        --synth synthdata/output/recordings

Prints, per sensor, the ratio synthetic/real of the 95th percentile of the
impact-band energy. Around 1 is what you want; the wrists are the sensitive
ones because real wrists carry almost nothing in that band.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "poser"))
from features import high_band_energy          # noqa: E402
from config import ACC_SCALE, SENSORS          # noqa: E402


def band_q95(folder: Path, sensor: str, suffix: str) -> float:
    p = folder / f"{sensor}{suffix}.csv"
    if not p.exists():
        return float("nan")
    d = pd.read_csv(p, usecols=["t", "acc_x", "acc_y", "acc_z"])
    t = d["t"].to_numpy(float)
    acc = d[["acc_x", "acc_y", "acc_z"]].to_numpy(float)
    if len(t) < 500:
        return float("nan")
    fs = 1.0 / np.median(np.diff(t[:5000]))
    t_dst = np.arange(t[0], t[-1], 1.0 / 50.0)
    hb = high_band_energy(np.linalg.norm(acc, axis=1), fs, t, t_dst)
    return float(np.percentile(hb / ACC_SCALE, 95))


def collect(root: Path, suffix: str, limit: int) -> dict:
    out = {s: [] for s in SENSORS}
    folders = [d for d in sorted(root.iterdir()) if d.is_dir()][:limit] if root.is_dir() else []
    for d in folders:
        for s in SENSORS:
            v = band_q95(d, s, suffix)
            if np.isfinite(v):
                out[s].append(v)
    return out, len(folders)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True, help="folder of real recordings")
    ap.add_argument("--synth", required=True, help="folder of synthetic recordings")
    ap.add_argument("--suffix", default="_aligned")
    ap.add_argument("--limit", type=int, default=40, help="recordings per side")
    a = ap.parse_args()

    real, n_real = collect(Path(a.real), a.suffix, a.limit)
    syn, n_syn = collect(Path(a.synth), a.suffix, a.limit)
    if not any(real.values()):
        raise SystemExit(f"no readable recordings under {a.real}")
    if not any(syn.values()):
        raise SystemExit(f"no readable recordings under {a.synth}")

    print(f"impact-band energy (q95, feature units), {a.suffix} files")
    print(f"{n_real} real, {n_syn} synthetic recordings\n")
    print(f"{'sensor':14s}{'real':>10s}{'synthetic':>12s}{'ratio':>9s}")
    worst = 0.0
    for s in SENSORS:
        if not real[s] or not syn[s]:
            continue
        r, y = float(np.median(real[s])), float(np.median(syn[s]))
        ratio = y / max(r, 1e-9)
        worst = max(worst, ratio if ratio > 1 else 1 / max(ratio, 1e-9))
        print(f"{s:14s}{r:10.4f}{y:12.4f}{ratio:8.1f}x")

    print()
    if worst <= 2.0:
        print("Within a factor of two of the real data - the smoothing fits.")
    else:
        print(f"Off by up to {worst:.1f}x. Too much energy means lowering --smooth-hz,\n"
              f"too little means raising it. Regenerate a handful of recordings per\n"
              f"value rather than the whole set.")
    return 0 if worst <= 2.0 else 1


if __name__ == "__main__":
    sys.exit(main())
