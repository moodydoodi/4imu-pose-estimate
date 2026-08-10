"""Paths and helpers for check_setup.py and run_pipeline.py, relative to the
repository root."""
from pathlib import Path

HERE = Path(__file__).resolve().parent      # src/synth
PROJECT = HERE.parents[1]                   # repository root

SYNTH_DIR = PROJECT / "synthdata"
IN_DIR = SYNTH_DIR / "input"
AMASS_DIR = IN_DIR / "amass_raw"
MODEL_DIR = IN_DIR / "body_models"

OUT_DIR = SYNTH_DIR / "output"
PROFILE = OUT_DIR / "ax6_noise_profile.json"
POSE_DIR = OUT_DIR / "poses"
REC_DIR = OUT_DIR / "recordings"
REPORT_DIR = OUT_DIR / "reports"

REQUIREMENTS = PROJECT / "requirements.txt"

SENSORS = ["left_wrist", "right_wrist", "left_ankle", "right_ankle"]


def ensure_dirs():
    for d in (AMASS_DIR, MODEL_DIR, OUT_DIR, POSE_DIR, REC_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def is_recording(p: Path) -> bool:
    """A recording folder holds the four aligned sensor files."""
    return p.is_dir() and all((p / f"{s}_aligned.csv").exists() for s in SENSORS)


def find_recordings(root: Path = PROJECT):
    """-> real recording folders found under data/."""
    seen, out = set(), []
    candidates = [root / "data" / "processed", root / "data" / "raw",
                  root / "data" / "sample", root / "data", root]
    for base in candidates:
        if not base.is_dir():
            continue
        for p in sorted(base.iterdir()):
            if is_recording(p) and p.resolve() not in seen:
                seen.add(p.resolve())
                out.append(p)
    return out


def amass_files(folder: Path = AMASS_DIR):
    if not folder.is_dir():
        return []
    skip = ("shape", "stagei")
    return sorted(p for p in folder.rglob("*.npz")
                  if p.is_file() and not any(s in p.name.lower() for s in skip))


def body_model_files(folder: Path = MODEL_DIR):
    return sorted(folder.rglob("*.npz")) if folder.is_dir() else []


# Checked by check_setup.py. Optional means the pipeline still runs without it.
PACKAGES = [
    ("numpy", "everything", False),
    ("pandas", "everything", False),
    ("scipy", "feature filters", True),
    ("matplotlib", "comparison plots", True),
]


def check_packages():
    """-> list of (name, installed, version, needed_for, optional)."""
    rows = []
    for name, why, optional in PACKAGES:
        try:
            m = __import__(name)
            rows.append((name, True, getattr(m, "__version__", "?"), why, optional))
        except Exception:
            rows.append((name, False, "-", why, optional))
    return rows
