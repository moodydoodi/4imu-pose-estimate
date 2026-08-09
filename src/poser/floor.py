"""Cost of forcing every subject onto one shared skeleton.

This cannot be measured from the cache, because prepare.py already stores poses
on the canonical skeleton, so the answer there is zero by construction. Measured
here against the raw MediaPipe pose: true joint directions, canonical bone
lengths.

    python floor.py --data data/processed --skeleton config/skeleton.json --exclude video7
"""
import argparse
from pathlib import Path

import numpy as np

import dataio
import skeleton as SK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--skeleton", default="config/skeleton.json")
    ap.add_argument("--exclude", nargs="*", default=[])
    a = ap.parse_args()

    canon, per = SK.load_skeleton(a.skeleton)
    print(f"{'recording':12s}{'own lengths':>16s}{'canonical':>13s}"
          f"{'surcharge':>11s}   largest length deviation")
    rows = []
    for d in dataio.find_recordings(a.data, exclude=a.exclude):
        t, P = dataio.read_pose(d)
        if P is None:
            continue
        D, L = SK.bone_dirs_and_lengths(P)
        own = SK.mpjpe(SK.forward(D, np.median(L, axis=0)), P)
        can = SK.mpjpe(SK.forward(D, canon), P)
        dev = np.abs(np.median(L, axis=0) - canon) * 1000
        i = int(np.argmax(dev))
        rows.append((own, can))
        print(f"{d.name:12s}{own:13.1f} mm{can:10.1f} mm{can-own:8.1f} mm"
              f"   {SK.JOINT_NAMES[i+1]} {dev[i]:.0f} mm")
    if rows:
        o = np.mean([r[0] for r in rows]); c = np.mean([r[1] for r in rows])
        print(f"{'mean':12s}{o:13.1f} mm{c:10.1f} mm{c-o:8.1f} mm")
    print("\n'own lengths' is MediaPipe's own jitter: measured bone lengths vary")
    print("from frame to frame and a rigid skeleton can never match that.")
    print("'surcharge' is the price of giving every subject the same skeleton.")
    print("Training and evaluation both use the canonical skeleton, so neither")
    print("value appears in the model MPJPE; they only say how good the targets")
    print("are.")


if __name__ == "__main__":
    main()
