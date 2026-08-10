"""Index AMASS and build a reproducible, diversity-weighted selection manifest.
Categories are movement profiles derived from the raw parameters, not action
labels. Fixing source file and crop here is what makes a run repeatable; the
manifest is passed to run_pipeline.py --selection.

    python build_amass_manifest.py --amass synthdata/input/amass_raw \\
        --index synthdata/output/amass_index.json \\
        --selection synthdata/output/manifest.json --count 800
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def stem_for(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    dataset = rel.parts[0] if len(rel.parts) else path.parent.name
    subject = rel.parts[-2] if len(rel.parts) >= 2 else path.parent.name
    raw = f"{dataset}__{subject}__{path.stem}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def metric(path: Path, root: Path, crop_s: float, min_seconds: float) -> dict | None:
    try:
        z = np.load(path, allow_pickle=False)
        poses = np.asarray(z["poses"], float)
        fps = float(np.asarray(z.get("mocap_framerate", 60.0)).reshape(-1)[0])
        trans = np.asarray(z["trans"], float) if "trans" in z.files else None
    except Exception as exc:
        return {"source": str(path), "error": str(exc)}
    # dataio needs two 4 s windows; nine seconds leaves slack for the common
    # time axis of pose and sensors.
    if poses.ndim != 2 or len(poses) < max(20, int(min_seconds * fps)) or fps <= 1:
        return None
    n_joints = min(22, poses.shape[1] // 3)
    rv = poses[:, :n_joints * 3].reshape(len(poses), n_joints, 3)
    # Purely kinematic activity measure, not an action label.
    omega = np.linalg.norm(np.diff(rv, axis=0), axis=2) * fps
    arm_idx = [i for i in (16, 17, 18, 19, 20, 21) if i < n_joints]
    leg_idx = [i for i in (1, 2, 4, 5, 7, 8, 10, 11) if i < n_joints]
    arm = np.mean(omega[:, arm_idx], axis=1) if arm_idx else np.zeros(len(omega))
    leg = np.mean(omega[:, leg_idx], axis=1) if leg_idx else np.zeros(len(omega))
    whole = np.mean(omega, axis=1)
    if trans is not None and len(trans) == len(poses):
        root_speed = np.linalg.norm(np.diff(trans[:, [0, 2]], axis=0), axis=1) * fps
    else:
        root_speed = np.zeros(len(omega))
    root_turn = np.linalg.norm(np.diff(rv[:, 0], axis=0), axis=1) * fps
    win = max(1, int(crop_s * fps))
    starts = np.arange(0, max(1, len(whole) - win), max(1, int(fps)))
    # Prefers lively motion over pure locomotion.
    energy = whole + 0.30 * root_speed + 0.15 * root_turn
    best = int(starts[np.argmax([energy[s:min(s + win, len(energy))].mean() for s in starts])])
    rel = path.relative_to(root)
    return {
        "source": str(path.resolve()), "output_stem": stem_for(path, root),
        "dataset": rel.parts[0], "subject": rel.parts[-2] if len(rel.parts) >= 2 else path.parent.name,
        "frames": int(len(poses)), "fps": fps, "duration_s": float(len(poses) / fps),
        "suggested_crop_start_s": float(best / fps), "crop_seconds": crop_s,
        "whole_p90": float(np.quantile(whole, .90)), "arm_p90": float(np.quantile(arm, .90)),
        "leg_p90": float(np.quantile(leg, .90)), "root_speed_p90": float(np.quantile(root_speed, .90)),
        "turn_p90": float(np.quantile(root_turn, .90)),
    }


def classify(rows: list[dict]) -> None:
    whole_cut = float(np.quantile([r["whole_p90"] for r in rows], .65))
    speed_cut = float(np.quantile([r["root_speed_p90"] for r in rows], .65))
    turn_cut = float(np.quantile([r["turn_p90"] for r in rows], .70))
    for r in rows:
        arm, leg = r["arm_p90"], r["leg_p90"]
        if r["whole_p90"] >= whole_cut and arm > 0.72 * leg and leg > 0.72 * arm:
            group = "full_body_dynamic"
        elif r["root_speed_p90"] >= speed_cut or r["turn_p90"] >= turn_cut:
            group = "locomotion_turn"
        elif leg > 1.30 * max(arm, 1e-6):
            group = "lower_body_dynamic"
        elif arm > 1.30 * max(leg, 1e-6):
            group = "upper_body_dynamic"
        else:
            group = "mixed_everyday"
        r["motion_profile"] = group
        r["activity_score"] = r["whole_p90"] + .3*r["root_speed_p90"] + .15*r["turn_p90"]


def select(rows: list[dict], n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    datasets = sorted({r["dataset"] for r in rows})
    targets = {d: n // len(datasets) + (i < n % len(datasets)) for i, d in enumerate(datasets)}
    weights = {"full_body_dynamic": .30, "locomotion_turn": .22,
               "lower_body_dynamic": .20, "upper_body_dynamic": .15, "mixed_everyday": .13}
    chosen, seen = [], Counter()
    for dataset in datasets:
        pool = [r for r in rows if r["dataset"] == dataset]
        for group, weight in weights.items():
            want = int(round(targets[dataset] * weight))
            candidates = [r for r in pool if r["motion_profile"] == group]
            rng.shuffle(candidates)
            candidates.sort(key=lambda r: r["activity_score"], reverse=True)
            for r in candidates:
                key = (dataset, r["subject"])
                if seen[key] >= 2 or len([x for x in chosen if x["dataset"] == dataset]) >= targets[dataset]:
                    continue
                chosen.append(r); seen[key] += 1
                if sum(x["dataset"] == dataset and x["motion_profile"] == group for x in chosen) >= want:
                    break
    # Fill remaining dataset quotas, still limiting repeats per subject.
    for dataset in datasets:
        pool = [r for r in rows if r["dataset"] == dataset and r not in chosen]
        rng.shuffle(pool); pool.sort(key=lambda r: r["activity_score"], reverse=True)
        for r in pool:
            if len([x for x in chosen if x["dataset"] == dataset]) >= targets[dataset]: break
            key = (dataset, r["subject"])
            if seen[key] < 3:
                chosen.append(r); seen[key] += 1
    # Small datasets may have too few subjects for the limit of three per
    # subject; top them up within their own quota.
    for dataset in datasets:
        remaining = [r for r in rows if r["dataset"] == dataset and r not in chosen]
        rng.shuffle(remaining); remaining.sort(key=lambda r: r["activity_score"], reverse=True)
        need = targets[dataset] - sum(x["dataset"] == dataset for x in chosen)
        chosen.extend(remaining[:max(0, need)])
    if len(chosen) < n:
        remaining = [r for r in rows if r not in chosen]
        rng.shuffle(remaining); remaining.sort(key=lambda r: r["activity_score"], reverse=True)
        chosen.extend(remaining[:n - len(chosen)])
    return chosen[:n]


def main():
    ap = argparse.ArgumentParser(description="AMASS candidate index and diversity-weighted selection manifest")
    ap.add_argument("--amass", required=True); ap.add_argument("--index", required=True)
    ap.add_argument("--selection", default=None); ap.add_argument("--count", type=int, default=0)
    ap.add_argument("--crop-seconds", type=float, default=25.0); ap.add_argument("--min-seconds", type=float, default=9.0)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args(); root = Path(args.amass).resolve()
    paths = sorted(p for p in root.rglob("*.npz") if "shape" not in p.name.lower() and "stagei" not in p.name.lower())
    rows = [r for p in paths if (r := metric(p, root, args.crop_seconds, args.min_seconds)) and "error" not in r]
    classify(rows)
    Path(args.index).parent.mkdir(parents=True, exist_ok=True)
    Path(args.index).write_text(json.dumps({"amass_root": str(root), "candidates": rows}, indent=2), encoding="utf-8")
    print(f"Index: {len(rows)}/{len(paths)} nutzbare Sequenzen; Datasets: {dict(Counter(r['dataset'] for r in rows))}")
    if args.selection:
        selected = select(rows, args.count, args.seed)
        out = {"amass_root": str(root), "seed": args.seed, "requested": args.count,
               "minimum_usable_seconds": args.min_seconds, "selected": selected,
               "dataset_counts": dict(Counter(r["dataset"] for r in selected)),
               "motion_profile_counts": dict(Counter(r["motion_profile"] for r in selected))}
        Path(args.selection).parent.mkdir(parents=True, exist_ok=True)
        Path(args.selection).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"Manifest: {len(selected)} Clips; datasets={out['dataset_counts']}; profiles={out['motion_profile_counts']}")


if __name__ == "__main__":
    main()
