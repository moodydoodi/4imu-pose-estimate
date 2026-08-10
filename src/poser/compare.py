"""Quick paired comparison of two or more runs: mean, spread, sign test.
Differences are taken per test recording. For the reported statistic with
bootstrap intervals use compare_metrics_to_pdf.py.

    python compare.py models/base_s0 models/finetune_s0
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load(path):
    p = Path(path)
    f = p / "metrics.json" if p.is_dir() else p
    d = json.loads(Path(f).read_text())
    folds = d["folds"] if isinstance(d, dict) and "folds" in d else d
    return {m["test"]: m for m in folds}, (d.get("args", {}) if isinstance(d, dict) else {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="directories containing metrics.json")
    ap.add_argument("--metric", default="mpjpe", choices=["mpjpe", "pa_mpjpe", "pck50", "pck100"])
    a = ap.parse_args()

    runs = []
    for r in a.runs:
        try:
            folds, args = load(r)
            runs.append((Path(r).name, folds, args))
        except Exception as e:
            print(f"[skipped] {r}: {e}")
    if not runs:
        raise SystemExit("Nothing found.")

    vids = sorted(set.intersection(*[set(f) for _, f, _ in runs]))
    if not vids:
        raise SystemExit("The runs share no common test recording.")

    print(f"metric: {a.metric}   common test recordings: {len(vids)}\n")
    print(f"{'recording':10s}" + "".join(f"{n[:14]:>15s}" for n, _, _ in runs))
    M = np.array([[runs[j][1][v][a.metric] for j in range(len(runs))] for v in vids])
    for i, v in enumerate(vids):
        print(f"{v:10s}" + "".join(f"{M[i, j]:15.1f}" for j in range(len(runs))))
    print(f"{'mean':10s}" + "".join(f"{M[:, j].mean():15.1f}" for j in range(len(runs))))

    seeds = [ar.get("seed", "?") for _, _, ar in runs]
    print("\nseeds: " + ", ".join(f"{n}={s}" for (n, _, _), s in zip(runs, seeds)))

    if len(runs) < 2:
        return
    print("\npaired comparison (negative means the second variant is better)")
    print("-" * 72)
    better = "pck" not in a.metric      # lower is better for MPJPE
    for j in range(1, len(runs)):
        d = M[:, j] - M[:, 0]
        wins = int((d < 0).sum()) if better else int((d > 0).sum())
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        t = d.mean() / se if se and se > 0 else float("nan")
        verdict = ("clear signal" if wins in (0, len(d)) and abs(t) > 2.5 else
                   "signal" if abs(t) > 2.0 and wins >= len(d) - 1 else
                   "within noise")
        print(f"{runs[j][0][:22]:24s} vs {runs[0][0][:16]:18s} "
              f"{d.mean():+7.2f} mm  spread {d.std(ddof=1):5.2f}  "
              f"{wins}/{len(d)} better  t={t:+.2f}  -> {verdict}")
    print("\n't' is the mean divided by its own standard error. Below about 2 the "
          "difference\ncannot be told from chance. Comparing two runs of the SAME "
          "variant with\ndifferent seeds gives exactly the noise level of the method.")


if __name__ == "__main__":
    main()
