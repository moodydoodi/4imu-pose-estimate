"""Run the full synthetic data generation in one pass.

    1. measure the noise profile from the real recordings
    2. convert AMASS sequences into the pose format
    3. generate virtual AX6 signals for every pose
    4. compare the result against a real recording

    python run_pipeline.py
    python run_pipeline.py --sample 600 --max-seconds 60 --foot-impacts
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import common as C

PY = sys.executable


def banner(n, title):
    print()
    print("=" * 74)
    print(f"Step {n}: {title}")
    print("=" * 74)


def run(script, args, label):
    cmd = [PY, str(C.HERE / script)] + [str(a) for a in args]
    print("$ " + " ".join(f'"{c}"' if " " in c else c for c in cmd) + "\n")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(C.HERE))
    if r.returncode != 0:
        raise SystemExit(f"\n{label} abgebrochen (Rueckgabewert {r.returncode}). "
                         f"Die Meldung darueber sagt, woran es lag.")
    print(f"\n{label} fertig in {time.time()-t0:.1f} s")


# --------------------------------------------------------------------- Schritte
def step1_profile(recs: list[Path]):
    banner(1, "measure the noise profile from the real recordings")
    print("Vorlagen: " + ", ".join(r.name for r in recs) + "\n")
    run("sensor_noise_profile.py",
        ["--subjects", *recs, "--out", C.PROFILE], "noise profile")


def step2_amass(limit, sample, max_seconds, fps, rec, selection, overwrite):
    banner(2, "convert AMASS sequences into the pose format")
    files = C.amass_files()
    if not files:
        raise SystemExit(f"No AMASS files under {C.AMASS_DIR}. "
                         f"Erst check_setup.py lesen.")
    args = ["--amass", C.AMASS_DIR, "--body-model", C.MODEL_DIR,
            "--out", C.POSE_DIR, "--fps", fps, "--max-seconds", max_seconds]
    if rec is not None:
        args += ["--bone-target", rec]
    if sample:
        args += ["--sample", sample]
    if limit:
        args += ["--limit", limit]
    if selection:
        args += ["--selection", Path(selection).resolve()]
    if overwrite:
        args += ["--overwrite"]
    run("amass_to_pose.py", args, "conversion")


def step3_synth(variants, seed0, jitter, root_amp, selection, overwrite,
                foot_impacts=False):
    banner(3, "Virtuelle AX6-Signale erzeugen")
    if selection:
        entries = json.loads(Path(selection).read_text(encoding="utf-8"))["selected"]
        poses = [C.POSE_DIR / f"{e['output_stem']}.npz" for e in entries]
        missing = [p for p in poses if not p.exists()]
        if missing:
            raise SystemExit(f"{len(missing)} manifest poses are missing, e.g. {missing[0]}")
    else:
        poses = sorted(p for p in C.POSE_DIR.glob("*.npz"))
    if not poses:
        raise SystemExit(f"No poses under {C.POSE_DIR}. Run step 2 first.")
    if not C.PROFILE.exists():
        raise SystemExit(f"Noise profile missing ({C.PROFILE}). Run step 1 first.")

    print(f"{len(poses)} pose sequences x {variants} variant(s) "
          f"= {len(poses)*variants} recordings\n")
    made = []
    def complete_synth(folder: Path) -> bool:
        return (folder / "synthesis_info.json").exists() and bool(list(folder.glob("*_gt_3d.csv"))) \
            and all((folder / f"{s}_segment.csv").exists() and
                    (folder / f"{s}_segment.csv").stat().st_size > 0 for s in C.SENSORS)
    for i, p in enumerate(poses):
        for v in range(variants):
            name = f"{p.stem}_s{v}" if variants > 1 else p.stem
            dest = C.REC_DIR / name
            if dest.exists() and not overwrite and complete_synth(dest):
                print(f"[{i*variants+v+1}/{len(poses)*variants}] {name} already present, skipped")
                continue
            if dest.exists() and not overwrite:
                print(f"[{i*variants+v+1}/{len(poses)*variants}] {name} incomplete, regenerating")
            print(f"[{i*variants+v+1}/{len(poses)*variants}] {name}")
            run("synth_imu.py",
                ["--npz", p, "--profile", C.PROFILE, "--out", dest,
                 "--seed", seed0 + v + 1000 * i,
                 "--jitter-scale", jitter, "--root-amp", root_amp]
                + (["--foot-impacts"] if foot_impacts else []),
                "synthesis")
            made.append(dest)
    print(f"\n{len(made)} synthetic recordings under {C.REC_DIR}")
    return made


def step4_validate(rec: Path):
    banner(4, "compare synthetic against a real recording")
    made = sorted(d for d in C.REC_DIR.iterdir() if C.is_recording(d))
    if not made:
        raise SystemExit(f"No synthetic recordings under {C.REC_DIR}.")
    target = made[0]
    print(f"echt: {rec.name}   synthetisch: {target.name}\n")
    run("validate_synthetic.py",
        ["--real", rec, "--synth", target, "--out", C.REPORT_DIR], "validation")
    print(f"\nDiagramme: {C.REPORT_DIR}")
    print("Wichtig ist, dass Histogramm und Leistungsdichte grob uebereinander "
          "liegen. Weichen sie stark ab, hilft --jitter-scale oder --root-amp.")


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Synthetische AX6-Trainingsdaten erzeugen")
    ap.add_argument("--recording", default=None,
                    help="name or path of the real recording (default: the first found)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N AMASS sequences")
    ap.add_argument("--sample", type=int, default=0,
                    help="N sequences spread evenly over all subjects")
    ap.add_argument("--selection", default=None,
                    help="Manifest aus build_amass_manifest.py; selektiert auch Schritt 3 exakt")
    ap.add_argument("--profile-recordings", nargs="*", default=None,
                    help="recordings used for the noise profile bank")
    ap.add_argument("--overwrite", action="store_true", help="bestehende Posen/Recordings ersetzen")
    ap.add_argument("--variants", type=int, default=1, help="noise variants per sequence")
    ap.add_argument("--fps", type=float, default=120.0, help="Bildrate der Posen (AMASS nativ)")
    ap.add_argument("--max-seconds", type=float, default=60.0, help="maximum length per sequence in seconds")
    ap.add_argument("--jitter-scale", type=float, default=1.0)
    ap.add_argument("--root-amp", type=float, default=0.10)
    ap.add_argument("--foot-impacts", action="store_true",
                    help="add foot impact transients at the ankles. Without this the "
                         "synthesis falls about 20 percent short of the measured "
                         "peak acceleration at the ankle.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--from", dest="start", type=int, default=1, help="ab Schritt N")
    ap.add_argument("--only", type=int, default=0, help="nur Schritt N")
    ap.add_argument("--skip-check", action="store_true")
    args = ap.parse_args()

    C.ensure_dirs()
    steps = {args.only} if args.only else set(range(args.start, 5))

    if not args.skip_check and 1 in steps:
        import check_setup
        if check_setup.main() != 0:
            raise SystemExit("\nAbgebrochen. Erst die Punkte oben erledigen, "
                             "dann noch einmal starten.")

    # pick the reference recording
    rec = None
    profile_recs = []
    if steps & {1, 2, 4}:
        recs = C.find_recordings()
        has_gt = lambda r: bool(list(r.glob("*_gt_3d.csv")))
        if args.recording:
            p = Path(args.recording)
            matches = [r for r in recs if r.name.lower() == args.recording.lower()]
            rec = p.resolve() if p.is_absolute() and C.is_recording(p) and has_gt(p) else next(
                (r for r in matches if has_gt(r)), None)
            if rec is None:
                raise SystemExit(f"Recording '{args.recording}' not found. "
                                 f"Available: {', '.join(r.name for r in recs) or 'none'}")
        elif recs:
            rec = next((r for r in recs if has_gt(r)), recs[0])
        else:
            raise SystemExit("No real recording found; one is needed as the noise "
                             "reference. See check_setup.py.")
        by_name = {r.name.lower(): r for r in recs}
        if args.profile_recordings:
            profile_recs = [Path(x) if C.is_recording(Path(x)) else by_name.get(x.lower())
                            for x in args.profile_recordings]
            if any(r is None for r in profile_recs):
                raise SystemExit("At least one --profile-recordings entry was not found.")
        else:
            profile_recs = [r for r in recs if r.name.lower() in {f"video{i}" for i in range(1, 7)}]
            profile_recs = profile_recs or [rec]

    t0 = time.time()
    if 1 in steps:
        step1_profile(profile_recs)
    if 2 in steps:
        step2_amass(args.limit, args.sample, args.max_seconds, args.fps, rec,
                    args.selection, args.overwrite)
    if 3 in steps:
        step3_synth(args.variants, args.seed, args.jitter_scale, args.root_amp,
                    args.selection, args.overwrite, args.foot_impacts)
    if 4 in steps:
        step4_validate(rec)

    print()
    print("=" * 74)
    print(f"Durchlauf beendet in {(time.time()-t0)/60:.1f} min")
    print(f"  noise profile : {C.PROFILE}")
    print(f"  Posen        : {C.POSE_DIR}")
    print(f"  recordings    : {C.REC_DIR}")
    print(f"  report        : {C.REPORT_DIR}")
    print()
    print("Folders under output/recordings/ use the same layout as the real")
    print("recordings (four sensor files plus *_gt_3d.csv) and can be dropped")
    print("direkt neben data/processed/ ins Training legen.")


if __name__ == "__main__":
    main()
