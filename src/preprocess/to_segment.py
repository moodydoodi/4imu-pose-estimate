"""Rotate sensor signals from the device frame into the segment frame.
The rotation comes from estimate_mount.py. Writes <sensor>_segment.csv next to
the input.

    python to_segment.py data/processed/video1 --suffix _aligned
"""
import argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--suffix", default="_aligned")
    ap.add_argument("--lag", type=float, default=0.0)
    a = ap.parse_args()
    for f in a.folders:
        info = convert(Path(f), a.suffix, a.lag)
        bad = [s for s, (r, h, n) in info.items() if r > 25 or h > 45]
        print(f"{Path(f).name:26s} {len(info)}/4 sensors"
              + (f"   uncertain: {', '.join(bad)}" if bad else ""))


if __name__ == "__main__":
    main()
