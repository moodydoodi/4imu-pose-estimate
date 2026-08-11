"""Test 1: bone-length stability, plus detection of fabricated (frozen) pose blocks.

Compares one or more recordings against each other. Two things are measured:

  1. FROZEN BLOCKS. 01_extract_skeleton.py writes NaN for every frame in which
     MediaPipe finds no person, then fills the gaps with

         df.interpolate(limit=30, ...)          # bounded
         df[col].bfill().ffill().fillna(0)      # UNBOUNDED

     The bfill has no limit, so a leading stretch with nobody in frame is not
     dropped -- it is back-filled with the first real pose and written out as a
     perfectly still skeleton. Those frames look like clean data downstream.

  2. BONE-LENGTH STABILITY, per window. A frozen block has a bone-length
     variance of exactly zero, so a single number over the whole recording is
     *improved* by the corruption and hides it. Everything here is therefore
     reported per window, and the frozen frames are reported separately.

Usage
    python src/preprocess/check_pose_quality.py data/processed/video*
    python src/preprocess/check_pose_quality.py <folders> --out pose_quality.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# MediaPipe Pose (33 landmarks). Bones chosen to be rigid in a real human.
BONES = {
    "l_upperarm":  (11, 13),
    "l_forearm":   (13, 15),
    "r_upperarm":  (12, 14),
    "r_forearm":   (14, 16),
    "l_thigh":     (23, 25),
    "l_shank":     (25, 27),
    "r_thigh":     (24, 26),
    "r_shank":     (26, 28),
    "shoulders":   (11, 12),
    "hips":        (23, 24),
    "l_torso":     (11, 23),
    "r_torso":     (12, 24),
}

# Left/right pairs, for the symmetry check.
SYMMETRY = [
    ("l_upperarm", "r_upperarm"),
    ("l_forearm",  "r_forearm"),
    ("l_thigh",    "r_thigh"),
    ("l_shank",    "r_shank"),
    ("l_torso",    "r_torso"),
]

JOINTS = sorted({j for pair in BONES.values() for j in pair})


def load_joints(folder: Path):
    """Return (time, P) with P of shape (frames, joint, 3). Only needed joints."""
    gts = sorted(folder.glob("*_gt_3d.csv"))
    if not gts:
        raise FileNotFoundError(f"no *_gt_3d.csv in {folder}")
    cols = ["time"] + [f"j{j}_{ax}" for j in JOINTS for ax in "xyz"]
    df = pd.read_csv(gts[0], usecols=cols)
    df.columns = [c.strip().lower() for c in df.columns]
    t = df["time"].to_numpy(float)
    P = np.stack(
        [df[[f"j{j}_x", f"j{j}_y", f"j{j}_z"]].to_numpy(float) for j in JOINTS],
        axis=1,
    )
    return t, P, gts[0].name


def frozen_mask(P, eps):
    """True where a frame is identical to the previous one across all joints.

    A real MediaPipe pose always jitters, even for someone standing still, so a
    run of exact repeats means the frames were manufactured by bfill/ffill.
    """
    step = np.linalg.norm(np.diff(P, axis=0), axis=2).max(axis=1)
    step = np.concatenate([[np.inf], step])          # frame 0 has no predecessor
    return step < eps, step


def leading_run(mask):
    """Length of the run of True at the start of mask (index 0 is skipped)."""
    n = 0
    for i in range(1, len(mask)):
        if mask[i]:
            n += 1
        else:
            break
    return n


def analyse(folder: Path, window_s=10.0, eps=1e-6):
    t, P, src = load_joints(folder)
    n_frames = len(t)
    if n_frames < 10:
        return {"error": "too few frames"}
    fps = 1.0 / np.median(np.diff(t[:2000])) if n_frames > 1 else 0.0

    fmask, step = frozen_mask(P, eps)
    lead = leading_run(fmask)

    idx = {j: k for k, j in enumerate(JOINTS)}
    lengths = {
        name: np.linalg.norm(P[:, idx[a]] - P[:, idx[b]], axis=1)
        for name, (a, b) in BONES.items()
    }

    # Live frames only: everything after the fabricated leading block.
    live = np.zeros(n_frames, bool)
    live[lead:] = True
    live &= ~fmask
    n_live = int(live.sum())

    # Per-window coefficient of variation, over live frames only.
    win = max(1, int(round(window_s * fps))) if fps > 0 else 300
    per_bone = {}
    for name, L in lengths.items():
        cvs = []
        for s in range(0, n_frames - win + 1, win):
            sl = slice(s, s + win)
            if live[sl].sum() < win * 0.5:
                continue
            seg = L[sl][live[sl]]
            m = np.mean(seg)
            if m > 1e-9:
                cvs.append(float(np.std(seg) / m))
        if not cvs:
            per_bone[name] = None
            continue
        cvs = np.array(cvs)
        per_bone[name] = {
            "median_cv": float(np.median(cvs)),
            "p90_cv": float(np.percentile(cvs, 90)),
            "mean_len": float(np.mean(L[live])) if n_live else None,
        }

    valid = [v["median_cv"] for v in per_bone.values() if v]
    overall = float(np.median(valid)) if valid else None
    worst = float(np.max(valid)) if valid else None

    sym = {}
    for a, b in SYMMETRY:
        va, vb = per_bone.get(a), per_bone.get(b)
        if va and vb and va["mean_len"] and vb["mean_len"]:
            la, lb = va["mean_len"], vb["mean_len"]
            sym[f"{a}/{b}"] = float(max(la, lb) / max(min(la, lb), 1e-9))
    sym_worst = float(max(sym.values())) if sym else None

    return {
        "source": src,
        "frames": n_frames,
        "fps": round(float(fps), 3),
        "duration_s": round(float(t[-1] - t[0]), 1),
        "frozen_leading_frames": int(lead),
        "frozen_leading_s": round(float(lead / fps), 1) if fps > 0 else None,
        "frozen_total_frames": int(fmask[1:].sum()),
        "frozen_total_pct": round(float(100.0 * fmask[1:].sum() / max(n_frames - 1, 1)), 2),
        "live_frames": n_live,
        "window_s": window_s,
        "median_bone_cv": overall,
        "worst_bone_cv": worst,
        "worst_symmetry_ratio": sym_worst,
        "per_bone": per_bone,
        "symmetry": sym,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+", help="recording folders, e.g. data/processed/video1")
    ap.add_argument("--window", type=float, default=10.0, help="window length in seconds")
    ap.add_argument("--eps", type=float, default=1e-6,
                    help="per-frame motion below this counts as frozen")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    res = {}
    for f in a.folders:
        p = Path(f)
        if not p.is_dir():
            continue
        try:
            res[p.name] = analyse(p, a.window, a.eps)
        except Exception as e:
            res[p.name] = {"error": str(e)}

    print(f"{'recording':12s} {'dur_s':>7s} {'frozen_lead':>12s} {'frozen%':>8s} "
          f"{'med_cv':>8s} {'worst_cv':>9s} {'sym':>6s}")
    print("-" * 70)
    for name, v in res.items():
        if "error" in v:
            print(f"{name:12s}   {v['error']}")
            continue
        lead = f"{v['frozen_leading_s']}s" if v["frozen_leading_s"] is not None else "-"
        mc = f"{v['median_bone_cv']:.4f}" if v["median_bone_cv"] is not None else "-"
        wc = f"{v['worst_bone_cv']:.4f}" if v["worst_bone_cv"] is not None else "-"
        sy = f"{v['worst_symmetry_ratio']:.2f}" if v["worst_symmetry_ratio"] else "-"
        print(f"{name:12s} {v['duration_s']:7.1f} {lead:>12s} "
              f"{v['frozen_total_pct']:7.2f}% {mc:>8s} {wc:>9s} {sy:>6s}")

    print("\nfrozen_lead > 0 means frames were manufactured by bfill and are not "
          "real pose.\nmed_cv is measured on live frames only, so it is comparable "
          "across recordings.")

    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))
        print(f"\nwritten: {a.out}")


if __name__ == "__main__":
    main()
