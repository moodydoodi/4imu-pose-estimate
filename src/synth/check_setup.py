"""Check what a synthesis run needs: packages, body model, AMASS, one recording."""
import sys
from pathlib import Path

import common as C

OK, WARN, BAD = "[ok]  ", "[!]   ", "[--]  "


def line(state, text):
    print(f"{state} {text}")


def main():
    C.ensure_dirs()
    print("=" * 74)
    print("Setup check for synthetic data generation")
    print("=" * 74)
    print(f"project root : {C.PROJECT}")
    print(f"synth data   : {C.SYNTH_DIR}")
    print(f"python       : {sys.version.split()[0]}  ({sys.executable})")
    print()

    blocking, optional = [], []

    # ------------------------------------------------------------- packages
    print("1) Python packages")
    for name, have, ver, why, is_optional in C.check_packages():
        if have:
            line(OK, f"{name:12s} {ver:10s} ({why})")
        elif is_optional:
            line(WARN, f"{name:12s} missing - optional, needed for: {why}")
            optional.append(name)
        else:
            line(BAD, f"{name:12s} missing - needed for: {why}")
            blocking.append(name)
    if blocking or optional:
        print(f'      -> pip install -r "{C.REQUIREMENTS}"')
    print()

    # --------------------------------------------------------- recordings
    print("2) real recording for the noise profile")
    recs = C.find_recordings()
    if recs:
        for r in recs:
            line(OK, f"{r.name}  ({r})")
    else:
        line(BAD, "no recording with left_wrist_aligned.csv etc. found")
        print(f"      looked under {C.PROJECT / 'data'} (processed, raw, sample) "
              f"and in {C.PROJECT}")
        blocking.append("recording")
    print()

    print("3) SMPL-H body model")
    try:
        from smpl_joints import BodyModel, find_model
        path, gender = find_model(C.MODEL_DIR)
        bm = BodyModel(path)
        line(OK, f"{gender}, {bm.n_joints} joints  ->  {path}")
    except SystemExit as e:
        line(BAD, "no body model found")
        for l in str(e).splitlines():
            print("      " + l)
        print(f"      Download: https://amass.is.tue.mpg.de  ->  Download  ->  "
              f"\"Extended SMPL+H model\"")
        print(f"      unpack into: {C.MODEL_DIR}")
        blocking.append("body model")
    print()

    print("4) AMASS motion data")
    files = C.amass_files()
    if files:
        line(OK, f"{len(files)} sequence files under {C.AMASS_DIR}")
        for f in files[:5]:
            print(f"      {f.relative_to(C.AMASS_DIR)}")
        if len(files) > 5:
            print(f"      ... and {len(files)-5} more")
        bad = _check_amass_content(files[0])
        if bad:
            line(BAD, bad)
            blocking.append("AMASS format")
    else:
        line(BAD, f"no .npz files under {C.AMASS_DIR}")
        print("      download: https://amass.is.tue.mpg.de -> Download -> any dataset")
        print("      IMPORTANT: choose the \"SMPL+H G\" variant, not SMPL-X.")
        print(f"      unpack into: {C.AMASS_DIR}")
        blocking.append("AMASS data")
    print()

    # -------------------------------------------------------------- verdict
    print("=" * 74)
    if blocking:
        print("Not ready yet. Missing: " + ", ".join(sorted(set(blocking))))
        print("Work through the points above, then run check_setup.py again.")
        return 1
    print("Everything present. Continue with:  python run_pipeline.py")
    if optional:
        print("Optional, still missing: " + ", ".join(optional) + " (works without)")
    return 0


def _check_amass_content(path: Path):
    try:
        import numpy as np
        d = np.load(path, allow_pickle=True)
    except Exception as e:
        return f"{path.name} cannot be read: {e}"
    keys = list(d.keys())
    if "poses" not in keys:
        return (f"{path.name} has no 'poses' (found: {keys[:6]}). "
                f"This is probably a model file, not a motion sequence.")
    n = int(d["poses"].shape[1])
    if n < 66:
        return (f"{path.name} has only {n} pose parameters, at least 66 are "
                f"required; the torso would be missing.")
    if n not in (156, 165, 72):
        print(f"      note: {n} pose parameters (156 is usual for SMPL+H). "
              f"Processing it anyway.")
    return None


if __name__ == "__main__":
    sys.exit(main())
