"""Rotate sensor signals from the device frame into the segment frame.
The rotation comes from estimate_mount.py. Writes <sensor>_segment.csv next to
the input.

    python to_segment.py data/processed/video1 --suffix _aligned
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import estimate_mount as EM


def convert(folder: Path, suffix="_aligned", lag=0.0, out_suffix="_segment"):
    res = EM.analyse(folder, lag=lag, suffix=suffix)
    info = {}
    for s in EM.SENSORS:
        v = res.get(s, {})
        if "R" not in v:
            continue
        R = np.array(v["R"])          # anatomical -> device
        d = pd.read_csv(folder / f"{s}{suffix}.csv")
        acc = d[["acc_x", "acc_y", "acc_z"]].to_numpy(float)
        gyr = d[["gyr_x", "gyr_y", "gyr_z"]].to_numpy(float)
        a2, g2 = acc @ R, gyr @ R     # same as R.T @ v
        pd.DataFrame({"t": d["t"].to_numpy(float),
                      "acc_x": a2[:, 0], "acc_y": a2[:, 1], "acc_z": a2[:, 2],
                      "gyr_x": g2[:, 0], "gyr_y": g2[:, 1], "gyr_z": g2[:, 2]}
                     ).to_csv(folder / f"{s}{out_suffix}.csv", index=False,
                              float_format="%.6g")
        info[s] = (v["residual_deg"], v["halves_deg"], v["n"])
    return info


MAX_RESIDUAL_DEG = 25.0     # median angular residual of the Kabsch fit
MAX_HALVES_DEG = 45.0       # disagreement between the first and second half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--suffix", default="_aligned")
    ap.add_argument("--lag", type=float, default=0.0)
    ap.add_argument("--max-residual", type=float, default=MAX_RESIDUAL_DEG)
    ap.add_argument("--max-halves", type=float, default=MAX_HALVES_DEG)
    ap.add_argument("--allow-uncertain", action="store_true",
                    help="write the files even when the calibration is weak")
    a = ap.parse_args()
    weak = []
    for f in a.folders:
        info = convert(Path(f), a.suffix, a.lag)
        bad = [s for s, (r, h, n) in info.items()
               if r > a.max_residual or h > a.max_halves]
        print(f"{Path(f).name:26s} {len(info)}/4 sensors"
              + (f"   uncertain: {', '.join(bad)}" if bad else ""))
        for s in bad:
            r, h, n = info[s]
            weak.append((Path(f).name, s, r, h))
        if len(info) < 4:
            weak.append((Path(f).name, f"only {len(info)}/4 sensors", 0.0, 0.0))

    if not weak:
        return 0
    print(f"\nWEAK SENSOR-TO-SEGMENT CALIBRATION "
          f"(residual > {a.max_residual:.0f} deg or halves > {a.max_halves:.0f} deg):")
    for name, s, r, h in weak:
        print(f"  {name:16s} {s:14s} residual {r:5.1f} deg   halves {h:5.1f} deg")
    print("\nThe segment frame is the basis of every reported number, so a weak fit\n"
          "propagates into everything. The fit needs frames where the limb is\n"
          "nearly still and the joint is bent; a recording with too few of those\n"
          "cannot be calibrated from the video alone.")
    if a.allow_uncertain:
        print("\n--allow-uncertain given, files were written anyway.")
        return 0
    print("\nFiles were written, but treat these recordings as unverified.\n"
          "Re-run with --allow-uncertain to acknowledge, or exclude them.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
