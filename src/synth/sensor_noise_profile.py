"""Measure a noise profile from real AX6 recordings.

Finds quiet blocks in each recording and measures, per sensor and axis: noise
standard deviation, AR(1) coefficient, bias instability, gyro offset,
quantisation step and the accelerometer scale error relative to 9.81 m/s2. One
profile per recording is stored so the synthesis can draw a different device
behaviour for each generated recording.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SENSORS = ["left_wrist", "right_wrist", "left_ankle", "right_ankle"]
ACC = ["acc_x", "acc_y", "acc_z"]
GYR = ["gyr_x", "gyr_y", "gyr_z"]
G_REF = 9.80665


# ---------------------------------------------------------------------------
def read_sensor(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in ["t"] + ACC + GYR if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: fehlende Spalten {missing}")
    return df


def sample_rate(t: np.ndarray) -> float:
    dt = np.diff(t[:5000])
    dt = dt[dt > 0]
    return float(1.0 / np.median(dt)) if dt.size else 0.0


def quantisation_step(x: np.ndarray, limit: int = 300_000) -> float:
    """Kleinster Abstand zwischen zwei tatsaechlich vorkommenden Werten."""
    u = np.unique(np.round(x[:limit].astype(float), 9))
    if u.size < 3:
        return 0.0
    d = np.diff(u)
    d = d[d > 1e-12]
    return float(np.min(d)) if d.size else 0.0


def blocks(x: np.ndarray, n: int) -> np.ndarray:
    nb = len(x) // n
    return x[: nb * n].reshape(nb, n, -1)


def find_static_blocks(acc: np.ndarray, gyr: np.ndarray, fs: float,
                       block_s: float = 1.0, keep_frac: float = 0.05):
    n = max(8, int(block_s * fs))
    ab, gb = blocks(acc, n), blocks(gyr, n)
    if len(gb) < 4:
        return ab, gb, np.arange(len(gb))
    gyr_std = np.linalg.norm(gb, axis=2).std(axis=1)
    acc_std = np.linalg.norm(ab, axis=2).std(axis=1)
    score = gyr_std + 10.0 * acc_std          # both must be small
    k = max(3, int(len(score) * keep_frac))
    idx = np.argsort(score)[:k]
    return ab[idx], gb[idx], idx


def ar1_coefficient(resid_blocks: np.ndarray) -> np.ndarray:
    out = []
    for a in range(resid_blocks.shape[2]):
        vals = []
        for b in resid_blocks[:, :, a]:
            v = b - b.mean()
            denom = np.sum(v * v)
            if denom > 1e-20 and len(v) > 3:
                vals.append(float(np.sum(v[1:] * v[:-1]) / denom))
        out.append(float(np.clip(np.mean(vals), -0.95, 0.95)) if vals else 0.0)
    return np.array(out)


def channel_stats(bl: np.ndarray, kind: str) -> dict:
    means = bl.mean(axis=1)                            # (n_blocks, 3)
    resid = bl - means[:, None, :]
    out = {
        "bias": means.mean(axis=0).tolist(),
        "noise_std": resid.reshape(-1, 3).std(axis=0).tolist(),
        "ar1": ar1_coefficient(resid).tolist(),
    }
    if kind == "acc":
        mag_drift = float(np.std(np.linalg.norm(means, axis=1)))
        out["bias_instability"] = [mag_drift] * 3
        out["bias_instability_note"] = "from the spread of |g| across the quiet blocks"
    else:
        out["bias_instability"] = means.std(axis=0).tolist()
    return out


def gap_stats(t: np.ndarray, fs: float) -> dict:
    dt = np.diff(t)
    nominal = 1.0 / fs
    gaps = dt[dt > 3 * nominal]
    return {"n_gaps": int(gaps.size),
            "gap_seconds_total": float(gaps.sum()) if gaps.size else 0.0,
            "max_gap_s": float(gaps.max()) if gaps.size else 0.0}


# ---------------------------------------------------------------------------
def profile_sensor(path: Path, block_s: float, keep_frac: float) -> dict:
    df = read_sensor(path)
    t = df["t"].to_numpy(float)
    acc = df[ACC].to_numpy(float)
    gyr = df[GYR].to_numpy(float)
    fs = sample_rate(t)

    ab, gb, idx = find_static_blocks(acc, gyr, fs, block_s, keep_frac)
    a_stats, g_stats = channel_stats(ab, "acc"), channel_stats(gb, "gyr")

    gmag = float(np.linalg.norm(ab.reshape(-1, 3), axis=1).mean())
    a_stats["lsb"] = quantisation_step(acc[:, 0])
    g_stats["lsb"] = quantisation_step(gyr[:, 0])
    a_stats.pop("bias")

    return {
        "file": path.name,
        "n_samples": int(len(df)),
        "duration_s": float(t[-1] - t[0]),
        "fs_hz": fs,
        "static_blocks_used": int(len(idx)),
        "gravity_mag_ms2": gmag,
        "acc_scale_vs_9_81": gmag / G_REF,
        "acc": a_stats,
        "gyr": g_stats,
        "gaps": gap_stats(t, fs),
    }


def profile_recording(subj: Path, suffix: str, block_seconds: float,
                      quiet_fraction: float, verbose: bool = True) -> dict:
    out = {"source_subject": str(subj), "suffix": suffix,
           "gravity_reference_ms2": G_REF, "sensors": {}}
    if verbose:
        print(f"\nAufnahme: {subj}")
        print(f"{'sensor':14s} {'fs':>6s} {'|g| rest':>9s} {'scale':>7s} "
              f"{'acc noise':>13s} {'gyr noise':>13s}")
    for s in SENSORS:
        p = subj / f"{s}{suffix}.csv"
        if not p.exists():
            if verbose:
                print(f"{s:14s} -- file missing, skipped ({p.name})")
            continue
        prof = profile_sensor(p, block_seconds, quiet_fraction)
        out["sensors"][s] = prof
        if verbose:
            print(f"{s:14s} {prof['fs_hz']:6.1f} {prof['gravity_mag_ms2']:9.3f} "
                  f"{prof['acc_scale_vs_9_81']:7.3f} "
                  f"{np.mean(prof['acc']['noise_std']):13.4f} "
                  f"{np.mean(prof['gyr']['noise_std']):13.4f}")
    if not out["sensors"]:
        raise ValueError("Keine Sensordateien gefunden.")
    scales = [v["acc_scale_vs_9_81"] for v in out["sensors"].values()]
    out["aggregate"] = {
        "fs_hz": float(np.median([v["fs_hz"] for v in out["sensors"].values()])),
        "acc_noise_std": float(np.mean([np.mean(v["acc"]["noise_std"]) for v in out["sensors"].values()])),
        "gyr_noise_std": float(np.mean([np.mean(v["gyr"]["noise_std"]) for v in out["sensors"].values()])),
        "acc_lsb": float(np.median([v["acc"]["lsb"] for v in out["sensors"].values()])),
        "gyr_lsb": float(np.median([v["gyr"]["lsb"] for v in out["sensors"].values()])),
        "acc_scale_min": float(np.min(scales)), "acc_scale_max": float(np.max(scales)),
    }
    return out


def aggregate_profiles(profiles: list[dict]) -> dict:
    keys = ["fs_hz", "acc_noise_std", "gyr_noise_std", "acc_lsb", "gyr_lsb"]
    return {k: float(np.median([p["aggregate"][k] for p in profiles])) for k in keys}


def main():
    ap = argparse.ArgumentParser(description="measure an AX6 noise profile from real recordings")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--subject",
                    help="a recording folder containing <sensor>_aligned.csv")
    src.add_argument("--subjects", nargs="+",
                     help="several real recordings: builds a profile bank, one entry drawn per synthetic clips davon")
    ap.add_argument("--suffix", default="_aligned",
                    help="_aligned (Standard) oder _mp_spatial")
    ap.add_argument("--out", default=None, help="Ziel-JSON (Standard: config/ax6_noise_profile.json)")
    ap.add_argument("--block-seconds", type=float, default=1.0)
    ap.add_argument("--quiet-fraction", type=float, default=0.05,
                    help="fraction of the quietest blocks treated as rest")
    args = ap.parse_args()

    subjects = [Path(args.subject)] if args.subject else [Path(p) for p in args.subjects]
    for subj in subjects:
        if not subj.exists():
            raise SystemExit(f"directory not found: {subj}")
    profiles = [profile_recording(s, args.suffix, args.block_seconds,
                                  args.quiet_fraction) for s in subjects]
    out = profiles[0] if len(profiles) == 1 else {
        "source_subjects": [p["source_subject"] for p in profiles],
        "suffix": args.suffix, "gravity_reference_ms2": G_REF,
        "profiles": profiles, "aggregate": aggregate_profiles(profiles),
    }

    dest = Path(args.out) if args.out else Path("config/ax6_noise_profile.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nProfil geschrieben: {dest}")

    if len(profiles) == 1:
        spread = (out["aggregate"]["acc_scale_max"] - out["aggregate"]["acc_scale_min"]) * 100
        if spread > 2.0:
            print(f"Hinweis: die Sensoren unterscheiden sich in der Skalierung um {spread:.1f} %. "
                  f"Eine Kalibrierung wuerde diesen systematischen Fehler entfernen.")


if __name__ == "__main__":
    main()
