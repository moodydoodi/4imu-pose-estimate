"""Build the canonical skeleton and the feature cache.
Writes one .npz per recording to --cache. --skeleton derives a new skeleton,
--skeleton-in reuses an existing one; synthetic data must reuse the real one.

    python prepare.py --data data/processed --suffix _segment --frame body \
        --cache cache/real_body --skeleton config/skeleton.json
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
    ap.add_argument("--skeleton", default=None,
                    help="derive a NEW skeleton from --data and write it here. "
                         "Refuses to overwrite an existing file unless --force.")
    ap.add_argument("--skeleton-in", default=None,
                    help="reuse an existing skeleton file instead of deriving one")
    ap.add_argument("--force", action="store_true",
                    help="allow --skeleton to overwrite an existing file")
    ap.add_argument("--frame", choices=["world", "body"], default="world",
                    help="body rotates the target pose so the hip axis points along "
                         "+x, removing the unobservable rotation about the vertical")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    if a.skeleton and a.skeleton_in:
        raise SystemExit("--skeleton and --skeleton-in are mutually exclusive: "
                         "--skeleton writes a new file, --skeleton-in reads one.")
    out_skel = Path(a.skeleton or "config/skeleton.json")
    if not a.skeleton_in and out_skel.exists() and not a.force:
        raise SystemExit(
            f"{out_skel} already exists and would be overwritten.\n"
            f"  To reuse it (the usual case):  --skeleton-in {out_skel}\n"
            f"  To deliberately rebuild it:    --skeleton {out_skel} --force")

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
        out = out_skel
        SK.save_skeleton(out, canon, per)
        print(f"skeleton written to {out}")

    print("\ncanonical bone lengths (mm):")
    for i in range(1, 13):
        spread = (max(v[i-1] for v in per.values()) -
                  min(v[i-1] for v in per.values())) * 1000 if per else 0.0
        print(f"  {i:2d} {SK.JOINT_NAMES[i] if hasattr(SK,'JOINT_NAMES') else '':14s}"
              f"{canon[i-1]*1000:7.0f}   spread {spread:5.0f}")

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

    # ---- reference values
    print("\nMPJPE of the trivial mean pose:")
    for p in sorted(cache.glob("*.npz"))[:8]:
        z = np.load(p, allow_pickle=True)
        Y = z["Y"]
        mean_pose = np.repeat(Y.mean(0)[None], len(Y), 0)
        print(f"  {str(z['name']):14s} {SK.mpjpe(mean_pose, Y):6.1f} mm")


if __name__ == "__main__":
    main()
