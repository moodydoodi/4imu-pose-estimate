"""Filter an existing selection manifest towards leg-dominant motion.

The pilot showed that the synthetic set sits at an ankle/wrist impact ratio of
about 2, while the real jump recordings sit at about 6, with single sequences
below 1 -- there the wrists carry more impact energy than the ankles, which is
the opposite of the target activity.

The cause is the selection, not the synthesis: build_amass_manifest.py weights
for diversity across datasets and motion profiles, so martial arts, gestures
and crawling end up in the set. Every entry already carries `leg_p90` and
`arm_p90`, so the mismatch can be filtered out without rescanning AMASS.

Keeps the manifest structure intact, so run_pipeline.py --selection accepts the
result unchanged. Nothing is re-selected and no new sequences are added; this
only drops entries, which keeps the set a strict subset of the original.

    python filter_manifest.py IN.json OUT.json                  # ratio >= 1.0
    python filter_manifest.py IN.json OUT.json --min-ratio 1.3
    python filter_manifest.py IN.json OUT.json --dry-run
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def ratio(entry) -> float:
    """leg_p90 / arm_p90; >1 means the legs move more than the arms."""
    arm = float(entry.get("arm_p90", 0.0))
    leg = float(entry.get("leg_p90", 0.0))
    return leg / arm if arm > 1e-9 else float("inf")


def usable_seconds(entry, max_seconds=60.0) -> float:
    """Seconds that actually survive the crop in amass_to_pose.py.

    The clip is cut at suggested_crop_start_s and capped at --max-seconds, so
    duration_s alone overstates what reaches prepare.py. Anything near the
    200-frame training window (4 s at 50 Hz) is dropped there as too short.
    """
    dur = float(entry.get("duration_s", 0.0))
    start = float(entry.get("suggested_crop_start_s", 0.0))
    return min(max(dur - start, 0.0), max_seconds)


def quantiles(values):
    if not values:
        return None
    v = sorted(values)
    def q(p):
        return v[min(len(v) - 1, int(p * (len(v) - 1)))]
    return q(0.0), q(0.25), q(0.5), q(0.75), q(1.0)


def describe(label, entries):
    rs = [ratio(e) for e in entries if ratio(e) != float("inf")]
    qs = quantiles(rs)
    print(f"\n{label}: {len(entries)} sequences")
    if qs:
        print(f"  leg/arm  min {qs[0]:.2f}  q25 {qs[1]:.2f}  median {qs[2]:.2f}  "
              f"q75 {qs[3]:.2f}  max {qs[4]:.2f}")
        print(f"  below 1.0 (arm dominant): {sum(r < 1.0 for r in rs)}")
    us = [usable_seconds(e) for e in entries]
    qu = quantiles(us)
    if qu:
        print(f"  usable s min {qu[0]:.1f}  q25 {qu[1]:.1f}  median {qu[2]:.1f}  "
              f"q75 {qu[3]:.1f}  max {qu[4]:.1f}")
        print(f"  total {sum(us)/60.0:.0f} min of motion")
    prof = Counter(e.get("motion_profile", "?") for e in entries)
    print("  profiles: " + ", ".join(f"{k}={v}" for k, v in sorted(prof.items())))
    ds = Counter(e.get("dataset", "?") for e in entries)
    print("  datasets: " + ", ".join(f"{k}={v}" for k, v in sorted(ds.items())))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("infile")
    ap.add_argument("outfile")
    ap.add_argument("--min-ratio", type=float, default=1.0,
                    help="keep entries with leg_p90/arm_p90 at or above this")
    ap.add_argument("--min-seconds", type=float, default=9.0,
                    help="keep entries with at least this many usable seconds "
                         "after the crop; shorter ones are dropped by prepare.py")
    ap.add_argument("--min-per-dataset", type=int, default=0,
                    help="if a dataset would drop below this, keep its best "
                         "entries by ratio so the set stays diverse")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    a = ap.parse_args()

    src = Path(a.infile)
    data = json.loads(src.read_text(encoding="utf-8"))
    entries = data["selected"]
    describe("before", entries)

    n_arm = sum(ratio(e) < a.min_ratio for e in entries)
    n_short = sum(usable_seconds(e) < a.min_seconds for e in entries)
    n_both = sum(ratio(e) < a.min_ratio and usable_seconds(e) < a.min_seconds
                 for e in entries)
    print(f"\ndropped for arm dominance: {n_arm}")
    print(f"dropped for length:        {n_short}")
    print(f"  of which both reasons:   {n_both}")

    keep = [e for e in entries
            if ratio(e) >= a.min_ratio and usable_seconds(e) >= a.min_seconds]

    # Optionally top a dataset back up so the filter cannot wipe one out.
    if a.min_per_dataset > 0:
        kept_ds = Counter(e.get("dataset", "?") for e in keep)
        for ds in sorted({e.get("dataset", "?") for e in entries}):
            need = a.min_per_dataset - kept_ds[ds]
            if need <= 0:
                continue
            pool = [e for e in entries
                    if e.get("dataset", "?") == ds and e not in keep
                    and usable_seconds(e) >= a.min_seconds]
            pool.sort(key=ratio, reverse=True)
            added = pool[:need]
            keep.extend(added)
            if added:
                print(f"  {ds}: {len(added)} entries kept below the threshold "
                      f"to hold the dataset quota")

    order = {id(e): i for i, e in enumerate(entries)}
    keep.sort(key=lambda e: order[id(e)])
    describe(f"after (min-ratio {a.min_ratio})", keep)

    dropped = len(entries) - len(keep)
    print(f"\n{len(keep)} kept, {dropped} dropped "
          f"({100.0 * dropped / max(len(entries), 1):.0f} %)")

    if a.dry_run:
        print("\n--dry-run, nothing written")
        return

    out = dict(data)
    out["selected"] = keep
    out["requested"] = len(keep)
    out["filtered_from"] = src.name
    out["filter_min_leg_arm_ratio"] = a.min_ratio
    out["filter_min_usable_seconds"] = a.min_seconds
    out["dataset_counts"] = dict(Counter(e.get("dataset", "?") for e in keep))
    out["motion_profile_counts"] = dict(Counter(e.get("motion_profile", "?") for e in keep))
    Path(a.outfile).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"written: {a.outfile}")


if __name__ == "__main__":
    main()
