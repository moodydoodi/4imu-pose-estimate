import pandas as pd
import argparse
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scipy.signal import butter, filtfilt, correlate, correlation_lags
from scipy.interpolate import interp1d

COMMON_HZ = 500.0
DT = 1.0 / COMMON_HZ
LPF_CUTOFF = 3.0

def apply_lpf(data, fps):
    nyq = 0.5 * fps
    normal_cutoff = LPF_CUTOFF / nyq
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def resample_signal(t, y):
    t_rel = t - t[0]
    t_uniform = np.arange(0, t_rel[-1], DT)
    f = interp1d(t_rel, y, kind='cubic', bounds_error=False, fill_value=0)
    y_uniform = f(t_uniform)
    return t_uniform, y_uniform, t[0]

PROJECT = Path(__file__).resolve().parents[2]     # repository root


def _dir(p, default):
    """Resolve a path against the repository root, not the working directory."""
    q = Path(p) if p else Path(default)
    return q if q.is_absolute() else (PROJECT / q)


MIN_SYNC_CONFIDENCE = 0.30     # below this the offset is not trustworthy


def extract_video_envelope(df_gt, loc):
    t_vid = df_gt['time'].values
    fps = 1.0 / np.mean(np.diff(t_vid[:100]))
    
    if loc == "LW":
        cols = [c for c in ['j15_x', 'j15_y', 'j15_z'] if c in df_gt.columns]
        if len(cols) == 3:
            pos = df_gt[cols].values
            vel = np.linalg.norm(np.diff(pos, axis=0, prepend=pos[0:1, :]), axis=1)
        else:
            vel = np.zeros(len(t_vid))
    elif loc == "RW":
        cols = [c for c in ['j16_x', 'j16_y', 'j16_z'] if c in df_gt.columns]
        if len(cols) == 3:
            pos = df_gt[cols].values
            vel = np.linalg.norm(np.diff(pos, axis=0, prepend=pos[0:1, :]), axis=1)
        else:
            vel = np.zeros(len(t_vid))
    else:
        y_cols = [c for c in ['j27_y', 'j28_y', 'j29_y', 'j30_y'] if c in df_gt.columns]
        if y_cols:
            pos = df_gt[y_cols].mean(axis=1).values
            vel = np.abs(np.diff(pos, prepend=pos[0]))
        else:
            vel = np.zeros(len(t_vid))
            
    y_env = apply_lpf(vel, fps)
    return resample_signal(t_vid, y_env)

def load_and_extract_imu_envelope(csv_path):
    df_head = pd.read_csv(csv_path, header=None, nrows=5)
    has_header = isinstance(df_head.iloc[0, 1], str) and any(c in df_head.iloc[0,1] for c in [':','-'])
    df = pd.read_csv(csv_path, header=0 if has_header else None)
    
    raw_time = df.iloc[:, 0]
    acc = df.iloc[:, 1:4].values.astype(float)
    gyro = df.iloc[:, 4:7].values.astype(float)
    
    
    t_series = pd.to_datetime(raw_time, errors='coerce')
    valid_times = t_series.dropna()
    
    if len(valid_times) < len(df) * 0.9 or (len(valid_times) > 0 and (valid_times.iloc[-1] - valid_times.iloc[0]).total_seconds() < 60 and len(df) > 10000):
        t = np.arange(len(df)) / 200.0
    else:
        t = t_series.values.astype(np.int64) / 1e9
        
    valid_mask = ~np.isnan(acc).any(axis=1)
    if np.sum(valid_mask) == 0:
        return None, None, None, None
        
    t = t[valid_mask]
    acc = acc[valid_mask]
    gyro = gyro[valid_mask]
    
    mag = np.linalg.norm(acc, axis=1)
    if 0.5 < np.median(mag) < 1.5: 
        acc *= 9.81
        mag *= 9.81
        
    fps = 1.0 / np.mean(np.diff(t[:100])) if len(t) > 100 else 200.0
    
    mag_dynamic = np.abs(mag - np.median(mag))
    y_env = apply_lpf(mag_dynamic, fps)
    
    t_uniform, y_uniform, t_start = resample_signal(t, y_env)
    return t_uniform, y_uniform, t_start, acc, gyro, t

def compute_rigid_offset(y_vid, y_imu):
    y_vid_norm = (y_vid - np.mean(y_vid)) / (np.std(y_vid) + 1e-8)
    y_imu_norm = (y_imu - np.mean(y_imu)) / (np.std(y_imu) + 1e-8)
    
    corr = correlate(y_vid_norm, y_imu_norm, mode='full')
    lags = correlation_lags(len(y_vid_norm), len(y_imu_norm), mode='full')
    
    best_lag_idx = np.argmax(corr)
    best_lag = lags[best_lag_idx]
    max_corr = corr[best_lag_idx] / min(len(y_vid_norm), len(y_imu_norm))
    
    return best_lag * DT, max_corr

def process_and_plot_subject(subj_folder, raw_root):
    subj_id = subj_folder.name
    print(f"\n[TARGET] {subj_id}")
    
    gt_files = list(subj_folder.glob(f"{subj_id}*_gt_3d.csv"))
    if not gt_files:
        print(f"  [!] Missing GT.")
        return {}

    gt_path = gt_files[0]
    df_gt = pd.read_csv(gt_path)
    t_vid_raw = df_gt['time'].values
    
    sensors = {"LW": "left_wrist", "RW": "right_wrist", "LA": "left_ankle", "RA": "right_ankle"}
    sync_metrics = {}
    plot_data = {}
    vid_envelopes = {}
    
    for loc, name in sensors.items():
        files = list((raw_root / subj_id).glob(f"*{subj_id}*_{loc}*.csv"))
        if not files:
            print(f"  [{loc}] Missing source CSV.")
            continue
            
        t_imu_uni, y_imu_uni, t_imu_start, acc_raw, gyro_raw, t_imu_raw = load_and_extract_imu_envelope(files[0])
        if t_imu_uni is None:
            print(f"  [{loc}] Parse failure.")
            continue
            
        t_vid_uni, y_vid_uni, t_vid_start = extract_video_envelope(df_gt, loc)
        vid_envelopes[loc] = (t_vid_uni, y_vid_uni, t_vid_start)
        
        lag_sec, confidence = compute_rigid_offset(y_vid_uni, y_imu_uni)
        
        global_offset = lag_sec + t_vid_start - t_imu_start
        t_aligned = t_imu_raw + global_offset
        
        print(f"  [{loc}] Shift: {global_offset:+.3f}s | Confidence: {confidence:.3f}")
        
        data_combined = np.hstack((acc_raw, gyro_raw))
        df_out = pd.DataFrame(data_combined, columns=["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"])
        df_out["t"] = t_aligned
        out_filename = subj_folder / f"{name}_aligned.csv"
        df_out[["t", "acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]].to_csv(out_filename, index=False)
        
        sync_metrics[loc] = {"offset_seconds": global_offset, "correlation_confidence": confidence}
        plot_data[loc] = {"t": t_aligned, "mag": np.linalg.norm(acc_raw, axis=1)}

    if sync_metrics:
        with open(subj_folder / "sync_manifest.json", "w") as f:
            json.dump({"source_gt": gt_path.name, "metrics": sync_metrics}, f, indent=4)
            
    fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
    vid_start, vid_end = t_vid_raw.min(), t_vid_raw.max()
    
    for i, (loc, name) in enumerate(sensors.items()):
        ax = axes[i]
        
        if loc in vid_envelopes:
            t_vid_uni, y_vid_uni, t_vid_start = vid_envelopes[loc]
            y_vid_plot = y_vid_uni / (np.max(y_vid_uni) + 1e-8)
            t_vid_plot = t_vid_uni + t_vid_start
            ax.plot(t_vid_plot, y_vid_plot, color='#1f77b4', alpha=0.7, label=f'Video {loc} LPF Env')
            
        if loc in plot_data:
            t_a = plot_data[loc]["t"]
            mag_a = plot_data[loc]["mag"]
            
            mask = (t_a >= vid_start - 2.0) & (t_a <= vid_end + 2.0)
            t_a_crop = t_a[mask]
            mag_a_crop = mag_a[mask]
            
            if len(t_a_crop) > 0:
                mag_dyn = np.abs(mag_a_crop - np.median(mag_a_crop))
                fps_plot = 1.0 / np.mean(np.diff(t_a_crop[:100])) if len(t_a_crop) > 100 else 200.0
                y_imu_plot = apply_lpf(mag_dyn, fps_plot)
                y_imu_plot = y_imu_plot / (np.max(y_imu_plot) + 1e-8)
                
                ax.plot(t_a_crop, y_imu_plot, color='#ff7f0e', alpha=0.7, label=f'IMU {loc} LPF Env')
                ax.set_title(f"Sensor: {loc} (Conf: {sync_metrics[loc]['correlation_confidence']:.2f})")
            else:
                ax.set_title(f"Sensor: {loc} (TIMING ERROR)")
        else:
            ax.set_title(f"Sensor: {loc} (DATA MISSING)")
            
        ax.grid(True, alpha=0.3)
        ax.set_xlim(vid_start, vid_end)
        if i == 0: ax.legend(loc='upper right')

    plt.xlabel("Time (seconds)")
    plt.suptitle(f"Continuous Sync Verification: {subj_id}", fontsize=16)
    plt.tight_layout()
    plt.savefig(subj_folder / "sync_verification_plot.png", dpi=150)
    plt.close()
    return sync_metrics

def main():
    ap = argparse.ArgumentParser(description="Synchronise sensors to the video")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--only", nargs="*", default=None,
                    help="recording folder names; default is all of them")
    ap.add_argument("--min-confidence", type=float, default=MIN_SYNC_CONFIDENCE,
                    help="reject a sensor below this cross-correlation confidence")
    a = ap.parse_args()

    raw_root, processed_root = _dir(a.raw, "data/raw"), _dir(a.processed, "data/processed")
    if not (processed_root.exists() and raw_root.exists()):
        raise SystemExit(f"need both {raw_root} and {processed_root}")
    subs = [d for d in sorted(processed_root.iterdir()) if d.is_dir()
            and (a.only is None or d.name in a.only)]
    if not subs:
        raise SystemExit(f"no recordings under {processed_root}")

    weak = []
    for subj in subs:
        m = process_and_plot_subject(subj, raw_root)
        for loc, v in (m or {}).items():
            if v["correlation_confidence"] < a.min_confidence:
                weak.append((subj.name, loc, v["correlation_confidence"]))

    print(f"\n{len(subs)} recording(s) synchronised.")
    if weak:
        print(f"\nWEAK SYNCHRONISATION, below {a.min_confidence}:")
        for name, loc, c in weak:
            print(f"  {name:14s} {loc:13s} confidence {c:.2f}")
        print("A weak fit means the offset is a guess. Check the plot in the "
              "recording folder before using it, and verify afterwards with\n"
              "  python src/poser/checklag.py --cache <cache>")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())