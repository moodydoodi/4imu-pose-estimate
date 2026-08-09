"""Feature and skeleton preparation. Does not require torch.

  1. read every recording under --data
  2. derive the canonical skeleton (median bone length across recordings) and
     write it to --skeleton
  3. compute features and store one .npz per recording in the cache

Synthetic recordings take the same skeleton via --skeleton-in. Only then are
pre-training and fine-tuning comparable; otherwise the body-size difference
between AMASS subjects and ours becomes an error the network can only read as a
pose error.

    python prepare.py --data data/processed --exclude video7 \
        --suffix _segment --frame body --cache cache/real_body \
        --skeleton config/skeleton.json
"""
import argparse
import time
from pathlib import Path

import numpy as np

import dataio
import skeleton as SK
from config import FPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--suffix", default="_segment")
    ap.add_argument("--fps", type=float, default=FPS)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--skeleton", default=None, help="write a new skeleton file here")
    ap.add_argument("--skeleton-in", default=None, help="reuse an existing skeleton file")
    ap.add_argument("--frame", choices=["world", "body"], default="world",
                    help="body rotates the target pose so the hip axis points along "
                         "+x, removing the unobservable rotation about the vertical")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    dirs = dataio.find_recordings(a.data, exclude=a.exclude)
    if a.limit:
        dirs = dirs[:a.limit]
    if not dirs:
        raise SystemExit(f"No recordings with *_gt_3d.csv under {a.data}.")
    print(f"reference frame: {a.frame}")
    print(f"{len(dirs)} recordings under {a.data}"
          + (f", excluded: {', '.join(a.exclude)}" if a.exclude else ""))

    # ---- skeleton
    if a.skeleton_in:
        canon, per = SK.load_skeleton(a.skeleton_in)
        print(f"skeleton taken from {a.skeleton_in}")
    else:
        poses = {}
        for d in dirs:
            t, P = dataio.read_pose(d)
            if P is not None:
                poses[d.name] = P
        canon, per = SK.canonical_from_recordings(poses)
        out = a.skeleton or "config/skeleton.json"
        SK.save_skeleton(out, canon, per)
        print(f"skeleton written to {out}")

    print("\ncanonical bone lengths (mm):")
    for i in range(1, 13):
        spread = (max(v[i-1] for v in per.values()) -
                  min(v[i-1] for v in per.values())) * 1000 if per else 0.0
        print(f"  {i:2d} {SK.JOINT_NAMES[i] if hasattr(SK,'JOINT_NAMES') else '':14s}"
              f"{canon[i-1]*1000:7.0f}   spread across recordings {spread:5.0f}")

    # ---- cache
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    ok, skip = 0, []
    for i, d in enumerate(dirs, 1):
        dst = cache / f"{d.name}.npz"
        if dst.exists():
            ok += 1
            continue
        t0 = time.time()
        r = dataio.load_recording(d, suffix=a.suffix, fps=a.fps, canon_L=canon,
                                  frame=a.frame)
        if r is None:
            skip.append(d.name)
            print(f"[{i}/{len(dirs)}] {d.name}: skipped "
                  f"(missing {a.suffix} files or too short)")
            continue
        D, _ = SK.bone_dirs_and_lengths(r["Y"])
        np.savez_compressed(dst, X=r["X"].astype(np.float32),
                            Y=r["Y"].astype(np.float32), D=D.astype(np.float32),
                            t=r["t"].astype(np.float32), name=d.name,
                            frame=a.frame, suffix=a.suffix, fps=a.fps)
        ok += 1
        print(f"[{i}/{len(dirs)}] {d.name}: {len(r['t'])} frames "
              f"({r['t'][-1]-r['t'][0]:.0f} s) in {time.time()-t0:.1f}s")

    print(f"\n{ok} recordings cached in {cache}"
          + (f", {len(skip)} skipped: {', '.join(skip)}" if skip else ""))

    # ---- reference values so later numbers can be judged
    print("\nreference value (MPJPE of the trivial mean pose):")
    for p in sorted(cache.glob("*.npz"))[:8]:
        z = np.load(p, allow_pickle=True)
        Y = z["Y"]
        mean_pose = np.repeat(Y.mean(0)[None], len(Y), 0)
        print(f"  {str(z['name']):14s} mean-pose baseline {SK.mpjpe(mean_pose, Y):6.1f} mm")


if __name__ == "__main__":
    main()
