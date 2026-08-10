"""Plot histograms and power spectra of a synthetic recording against a real one."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SENSORS = ["left_wrist", "right_wrist", "left_ankle", "right_ankle"]
ACC = ["acc_x", "acc_y", "acc_z"]
GYR = ["gyr_x", "gyr_y", "gyr_z"]


def load(folder: Path, sensor: str, suffix: str):
    p = folder / f"{sensor}{suffix}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def welch_psd(x: np.ndarray, fs: float, nseg: int = 2048):
    """Welch power spectral density, without scipy."""
    x = np.asarray(x, float)
    x = x - x.mean()
    if len(x) < nseg:
        nseg = max(64, len(x) // 2 * 2)
    step = nseg // 2
    win = np.hanning(nseg)
    segs = [x[i:i + nseg] * win for i in range(0, len(x) - nseg + 1, step)]
    if not segs:
        return np.array([]), np.array([])
    P = np.mean([np.abs(np.fft.rfft(s)) ** 2 for s in segs], axis=0)
    P /= (fs * (win ** 2).sum())
    return np.fft.rfftfreq(nseg, 1 / fs), P


def quiet_noise(acc, gyr, fs, frac=0.05):
    n = max(8, int(fs))
    nb = min(len(acc), len(gyr)) // n
    if nb < 4:
        return np.nan, np.nan, False
    a = acc[:nb * n].reshape(nb, n, 3)
    g = gyr[:nb * n].reshape(nb, n, 3)
    score = np.linalg.norm(g, axis=2).std(axis=1) + 10 * np.linalg.norm(a, axis=2).std(axis=1)
    k = max(3, int(nb * frac))
    idx = np.argsort(score)[:k]
    aa, gg = a[idx], g[idx]
    an = float((aa - aa.mean(axis=1, keepdims=True)).std())
    gn = float((gg - gg.mean(axis=1, keepdims=True)).std())
    return an, gn, bool(an < 0.5 and gn < 5.0)


def stats(df):
    t = df["t"].to_numpy(float)
    fs = 1.0 / np.median(np.diff(t[:5000]))
    acc, gyr = df[ACC].to_numpy(float), df[GYR].to_numpy(float)
    an, gn, has_rest = quiet_noise(acc, gyr, fs)
    return {
        "fs_hz": float(fs), "duration_s": float(t[-1] - t[0]), "n": int(len(df)),
        "has_rest_phase": has_rest,
        "acc_mag_median": float(np.median(np.linalg.norm(acc, axis=1))),
        "acc_mag_p95": float(np.percentile(np.linalg.norm(acc, axis=1), 95)),
        "gyr_mag_median": float(np.median(np.linalg.norm(gyr, axis=1))),
        "gyr_mag_p95": float(np.percentile(np.linalg.norm(gyr, axis=1), 95)),
        "acc_noise_quiet": an, "gyr_noise_quiet": gn,
        "acc": acc, "gyr": gyr,
    }


def make_plots(real, synth, sensor, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("  (matplotlib missing - plots skipped)")
        return
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f"echt vs. synthetisch - {sensor}")

    for i, (key, lab, unit) in enumerate(((("acc"), "Beschleunigung", "m/s^2"),
                                          (("gyr"), "angular rate", "deg/s"))):
        rm = np.linalg.norm(real[key], axis=1)
        sm = np.linalg.norm(synth[key], axis=1)
        bins = np.linspace(0, np.percentile(np.r_[rm, sm], 99), 80)
        ax[i, 0].hist(rm, bins=bins, alpha=.55, label="echt", density=True)
        ax[i, 0].hist(sm, bins=bins, alpha=.55, label="synthetisch", density=True)
        ax[i, 0].set_title(f"{lab}: magnitude"); ax[i, 0].set_xlabel(unit)
        ax[i, 0].legend(); ax[i, 0].grid(alpha=.3)

        f1, p1 = welch_psd(real[key][:, 0], real["fs_hz"])
        f2, p2 = welch_psd(synth[key][:, 0], synth["fs_hz"])
        ax[i, 1].loglog(f1[1:], p1[1:], label="echt", lw=1)
        ax[i, 1].loglog(f2[1:], p2[1:], label="synthetisch", lw=1)
        ax[i, 1].set_title(f"{lab}: Leistungsdichte (x-Achse)")
        ax[i, 1].set_xlabel("Hz"); ax[i, 1].legend(); ax[i, 1].grid(alpha=.3, which="both")

    fig.tight_layout()
    fig.savefig(out_dir / f"compare_{sensor}.png", dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="compare synthetic against real recordings")
    ap.add_argument("--real", required=True)
    ap.add_argument("--synth", required=True)
    ap.add_argument("--suffix", default="_aligned")
    ap.add_argument("--out", default="reports/synth_check")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"real": args.real, "synth": args.synth, "sensors": {}}

    hdr = f"{'sensor':13s} {'source':12s} {'|acc| med':>10s} {'|acc| p95':>10s} " \
          f"{'|gyr| med':>10s} {'|gyr| p95':>10s} {'acc noise':>11s} {'gyr noise':>11s}"
    print(hdr); print("-" * len(hdr))
    no_rest = set()

    for s in SENSORS:
        dr, ds = load(Path(args.real), s, args.suffix), load(Path(args.synth), s, args.suffix)
        if dr is None or ds is None:
            print(f"{s:13s} -- missing in one of the two recordings")
            continue
        sr, ss = stats(dr), stats(ds)
        for lab, st in (("echt", sr), ("synthetisch", ss)):
            if st["has_rest_phase"]:
                noise = f"{st['acc_noise_quiet']:11.4f} {st['gyr_noise_quiet']:11.4f}"
            else:
                noise = f"{'-':>11s} {'-':>11s}"     # no rest phase present
            print(f"{s:13s} {lab:12s} {st['acc_mag_median']:10.2f} {st['acc_mag_p95']:10.2f} "
                  f"{st['gyr_mag_median']:10.2f} {st['gyr_mag_p95']:10.2f} {noise}")
        if not ss["has_rest_phase"]:
            no_rest.add(s)
        make_plots(sr, ss, s, out_dir)
        report["sensors"][s] = {
            "real": {k: v for k, v in sr.items() if not isinstance(v, np.ndarray)},
            "synth": {k: v for k, v in ss.items() if not isinstance(v, np.ndarray)},
            "ratio_acc_mag": ss["acc_mag_median"] / max(sr["acc_mag_median"], 1e-9),
            "ratio_gyr_p95": ss["gyr_mag_p95"] / max(sr["gyr_mag_p95"], 1e-9),
        }
        print()

    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report and plots in {out_dir}")

    if no_rest:
        print("\nThe synthetic recording contains no rest phase, so the noise columns "
              "show a dash. That is expected: AMASS subjects move continuously, and "
              "the noise comes straight from the measured profile anyway.")

    bad = [s for s, v in report["sensors"].items()
           if not (0.7 < v["ratio_gyr_p95"] < 1.4)]
    if bad:
        print("\nnote: motion strength differs at " + ", ".join(bad) + ". A factor of "
              "two is harmless when the recordings show different motions. Larger "
              "deviations can be adjusted with --root-amp or --smooth-hz.")


if __name__ == "__main__":
    main()
