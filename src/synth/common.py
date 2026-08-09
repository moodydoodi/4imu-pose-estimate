"""Shared helpers for check_setup.py and run_pipeline.py."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent                       # TechnicalColl

IN_DIR = HERE / "input"
AMASS_DIR = IN_DIR / "amass_raw"
MODEL_DIR = IN_DIR / "body_models"

OUT_DIR = HERE / "output"
PROFILE = OUT_DIR / "ax6_noise_profile.json"
POSE_DIR = OUT_DIR / "poses"
REC_DIR = OUT_DIR / "recordings"
REPORT_DIR = OUT_DIR / "reports"

SENSORS = ["left_wrist", "right_wrist", "left_ankle", "right_ankle"]


def ensure_dirs():
    for d in (AMASS_DIR, MODEL_DIR, OUT_DIR, POSE_DIR, REC_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def is_recording(p: Path) -> bool:
    return p.is_dir() and all((p / f"{s}_aligned.csv").exists() for s in SENSORS)


def find_recordings(root: Path = PROJECT):
    seen, out = set(), []
    candidates = [root, root / "data" / "processed", root / "data"]
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


def check_packages():
    """-> Liste (Name, vorhanden, Version, benoetigt_fuer)"""
    wanted = [("numpy", "alles"), ("pandas", "alles"),
              ("matplotlib", "plots"), ("scipy", "filters (optional)")]
    rows = []
    for name, why in wanted:
        try:
            m = __import__(name)
            rows.append((name, True, getattr(m, "__version__", "?"), why))
        except Exception:
            rows.append((name, False, "-", why))
    return rows
