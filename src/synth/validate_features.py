"""Compare synthetic and real recordings in the 48 model input features.

Reports, per feature, the quantile distance between the synthetic distribution
and each real recording, expressed in standard deviations.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

POSER = Path(__file__).resolve().parents[1] / "poser"
sys.path.insert(0, str(POSER))
import dataio  # noqa: E402
from config import SENSORS  # noqa: E402

NAMES = ["grav_x", "grav_y", "grav_z", "lin_x", "lin_y", "lin_z",
         "gyr_x", "gyr_y", "gyr_z", "acc_mag", "gyr_mag", "highband"]


def take(folder: Path, suffix: str, max_frames: int) -> np.ndarray | None:
    rec = dataio.load_recording(folder, suffix=suffix, canon_L=None)
    if rec is None:
        return None
    x = rec["X"]
    if len(x) > max_frames:
        x = x[np.linspace(0, len(x) - 1, max_frames).astype(int)]
    return x


def summary(x: np.ndarray) -> dict:
    return {"mean": np.mean(x, 0).tolist(), "std": np.std(x, 0).tolist(),
            "q05": np.quantile(x, .05, 0).tolist(), "q50": np.quantile(x, .50, 0).tolist(),
            "q95": np.quantile(x, .95, 0).tolist()}


def distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    qa = np.quantile(a, [.05, .25, .50, .75, .95], axis=0)
    qb = np.quantile(b, [.05, .25, .50, .75, .95], axis=0)
    return np.mean(np.abs(qa - qb), axis=0) / np.maximum(np.std(b, axis=0), 1e-4)


def main():
    ap = argparse.ArgumentParser(description="Finale 48-Feature-Synth-vs-Real-Validierung")
    ap.add_argument("--real", nargs="+", required=True); ap.add_argument("--synth", nargs="+", required=True)
    ap.add_argument("--suffix", default="_segment"); ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=20000)
    args = ap.parse_args()
    synth = [x for p in map(Path, args.synth) if (x := take(p, args.suffix, args.max_frames)) is not None]
    if not synth: raise SystemExit("No readable synthetic recordings.")
    sx = np.concatenate(synth)
    report = {"suffix": args.suffix, "synth_recordings": len(synth), "synth_frames": len(sx),
              "synth": summary(sx), "real": {}, "feature_names": NAMES}
    rows = []
    for rp in map(Path, args.real):
        rx = take(rp, args.suffix, args.max_frames)
        if rx is None:
            print(f"WARN: {rp} not readable, skipped"); continue
        d = distance(sx, rx)
        report["real"][rp.name] = {"frames": len(rx), "stats": summary(rx), "quantile_distance": d.tolist()}
        for si, sensor in enumerate(SENSORS):
            for fi, feature in enumerate(NAMES):
                rows.append({"recording": rp.name, "sensor": sensor, "feature": feature,
                             "quantile_distance_std": float(d[si * 12 + fi])})
    if not rows: raise SystemExit("Keine lesbaren Realrecordings.")
    report["median_distance_per_feature"] = np.median(
        [report["real"][n]["quantile_distance"] for n in report["real"]], axis=0).tolist()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    med = np.asarray(report["median_distance_per_feature"]).reshape(4, 12)
    print(f"{len(synth)} synthetic recordings, {len(report['real'])} real recordings")
    for i, s in enumerate(SENSORS):
        print(f"{s:13s} median Q-Dist: " + ", ".join(f"{NAMES[j]}={med[i,j]:.2f}" for j in range(12)))
    print(f"report: {out} and {out.with_suffix('.csv')}")


if __name__ == "__main__": main()
