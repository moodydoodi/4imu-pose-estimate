import pandas as pd
import numpy as np
from pathlib import Path
import concurrent.futures
import os

# MediaPipe Target Bases
# Rows correspond to target X, Y, Z. Columns correspond to local X, Y, Z.
TARGET_BASIS_RIGHT = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, -1]
])

# Left sensor basis rotated 180 deg (det +1) to avoid reflection issues.
TARGET_BASIS_LEFT = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
])

# ---------------------------------------------------------------------------
# Mount Correction
# ---------------------------------------------------------------------------
# Fixed known mount misalignments based on pose vs. gravity calibration.
# E.g., video3 right_wrist was worn upside down (~180 deg around Z).
def rot180(axis: str) -> np.ndarray:
    """180-degree rotation around a coordinate axis (det +1)."""
    d = {"x": np.diag([1.0, -1.0, -1.0]),
         "y": np.diag([-1.0, 1.0, -1.0]),
         "z": np.diag([-1.0, -1.0, 1.0])}
    return d[axis.lower()]


MOUNT_CORRECTION = {
    ("video3", "right_wrist"): rot180("z"),
}


def mount_correction(subj_id: str, loc: str) -> np.ndarray:
    return MOUNT_CORRECTION.get((subj_id, loc), np.eye(3))


def extract_calibration_window(dfs: dict, fps: float, subj_id: str,
                               duration_sec: int = 5, t_range=None) -> dict:
    """
    Finds the most static window per sensor within the video timeframe.
    """
    print(f"[DEBUG] [PID:{os.getpid()}] [{subj_id}] Selecting quiet window per sensor"
          + (f" ({t_range[0]:.1f}..{t_range[1]:.1f}s)." 
             if t_range else " (WARNING: full file)."))

    block = max(2, int(fps * duration_sec))
    indices = {}
    
    for loc, df in dfs.items():
        t = df['t'].values.astype(float)
        if t_range is not None:
            inside = np.where((t >= t_range[0]) & (t <= t_range[1]))[0]
            off = int(inside[0]) if len(inside) else 0
        else:
            inside, off = np.arange(len(t)), 0
            
        acc = df[['acc_x', 'acc_y', 'acc_z']].values.astype(float)[inside]
        gyr = df[['gyr_x', 'gyr_y', 'gyr_z']].values.astype(float)[inside]
        n = len(acc) // block
        
        if n < 1:
            indices[loc] = (0, len(acc))
            print(f"[DEBUG] [PID:{os.getpid()}] [{subj_id}] {loc}: recording too short, using all.")
            continue

        A = acc[:n * block].reshape(n, block, 3)
        G = gyr[:n * block].reshape(n, block, 3)
        
        # Score based on low gyro variance and constant 1g acceleration
        score = (np.linalg.norm(G, axis=2).std(axis=1)
                 + 10.0 * np.linalg.norm(A, axis=2).std(axis=1))
        k = int(np.argmin(score))
        indices[loc] = (off + k * block, off + (k + 1) * block)

        mag = float(np.linalg.norm(A[k].mean(axis=0)))
        print(f"[DEBUG] [PID:{os.getpid()}] [{subj_id}] {loc}: window at t="
              f"{t[off + k * block]:7.1f}s, |acc|={mag:.2f} m/s^2, score={score[k]:.3f}")
        
        if abs(mag - 9.80665) > 1.0:
            print(f"[DEBUG] [PID:{os.getpid()}] [{subj_id}] WARNING: {loc} lacks a true static window.")

    return indices


def compute_calibration_matrix(acc_calib: np.ndarray, loc: str, subj_id: str) -> tuple:
    """
    Constructs static basis transformation from local to MediaPipe coordinates.
    """
    mean_acc = np.mean(acc_calib, axis=0)
    norm = np.linalg.norm(mean_acc)
    if norm < 1e-6:
        raise ValueError(f"[{subj_id}] {loc}: no gravity signal.")
    u_X = mean_acc / norm

    # Z is fixed in sensor frame. Alignment assumes corrected mount orientation.
    v_Z = np.array([0.0, 0.0, 1.0])
    u_Y = np.cross(u_X, v_Z)
    u_Y_norm = np.linalg.norm(u_Y)
    
    if u_Y_norm < 1e-6:
        print(f"[DEBUG] [PID:{os.getpid()}] [{subj_id}] WARNING: {loc} normal collinear with Z. Using Y fallback.")
        u_Y = np.array([0.0, 1.0, 0.0])
    else:
        u_Y = u_Y / u_Y_norm

    u_Z = np.cross(u_X, u_Y)
    u_Z = u_Z / np.linalg.norm(u_Z)

    R_imu = np.column_stack((u_X, u_Y, u_Z))
    R_target = TARGET_BASIS_LEFT if "left" in loc else TARGET_BASIS_RIGHT
    R_calib = R_target @ R_imu.T

    det = float(np.linalg.det(R_calib))
    if det < 0:
        raise ValueError(f"[{subj_id}] {loc}: R_calib is a reflection (det {det:+.3f}).")
        
    print(f"[DEBUG] [PID:{os.getpid()}] [{subj_id}] R_calib computed for {loc} (det {det:+.3f}).")
    return R_calib


def process_subject(subject_dir: Path):
    """Reads IMU files, aligns axes, and saves output."""
    subj_id = subject_dir.name
    print(f"\n[DEBUG] [PID:{os.getpid()}] [{subj_id}] Processing directory.")
    
    csv_files = list(subject_dir.glob("*_aligned.csv"))
    imu_files = {f.stem.replace('_aligned', ''): f for f in csv_files if "video" not in f.name}

    if len(imu_files) != 4:
        print(f"[DEBUG] [PID:{os.getpid()}] [{subj_id}] Missing sensors ({len(imu_files)}/4). Skipping.")
        return

    dfs = {}
    fps = 200.0
    for loc, path in imu_files.items():
        df = pd.read_csv(path)

        # Apply predefined mount corrections
        M = mount_correction(subj_id, loc)
        if not np.allclose(M, np.eye(3)):
            print(f"[DEBUG] [PID:{os.getpid()}] [{subj_id}] Applying mount correction for {loc}.")
            df[['acc_x', 'acc_y', 'acc_z']] = (M @ df[['acc_x', 'acc_y', 'acc_z']].values.T).T
            df[['gyr_x', 'gyr_y', 'gyr_z']] = (M @ df[['gyr_x', 'gyr_y', 'gyr_z']].values.T).T

        dfs[loc] = df
        if len(df) > 1:
            fps = 1.0 / np.mean(np.diff(df['t'].values[:100]))

    # Restrict calibration to actual video recording time
    t_range = None
    gt_files = sorted(subject_dir.glob("*_gt_3d.csv"))
    if gt_files:
        tg = pd.read_csv(gt_files[0], usecols=['time'])['time'].values.astype(float)
        t_range = (float(tg[0]), float(tg[-1]))
    else:
        print(f"[DEBUG] [PID:{os.getpid()}] [{subj_id}] WARNING: missing *_gt_3d.csv, no time restriction.")

    try:
        window_indices = extract_calibration_window(dfs, fps, subj_id, t_range=t_range)
    except ValueError as e:
        print(f"[ERROR] [PID:{os.getpid()}] [{subj_id}] Extraction failed: {e}")
        return

    for loc, df in dfs.items():
        start_idx, end_idx = window_indices[loc]
        
        acc_calib = df.iloc[start_idx:end_idx][['acc_x', 'acc_y', 'acc_z']].values
        R_calib = compute_calibration_matrix(acc_calib, loc, subj_id)

        acc = df[['acc_x', 'acc_y', 'acc_z']].values
        gyr = df[['gyr_x', 'gyr_y', 'gyr_z']].values

        acc_transformed = (R_calib @ acc.T).T
        gyr_transformed = (R_calib @ gyr.T).T

        df[['acc_x', 'acc_y', 'acc_z']] = acc_transformed
        df[['gyr_x', 'gyr_y', 'gyr_z']] = gyr_transformed

        # Self-test: Ensure resting gravity vector aligns with (0, +9.81, 0)
        check = acc_transformed[start_idx:end_idx].mean(axis=0)
        ang = np.degrees(np.arccos(np.clip(
            check @ np.array([0.0, 1.0, 0.0]) / (np.linalg.norm(check) + 1e-9), -1, 1)))
        status = "OK" if ang < 5.0 else "FAILED"
        
        print(f"[DEBUG] [PID:{os.getpid()}] [{subj_id}] {loc}: gravity error {ang:.1f} deg -> {status}")

        out_path = subject_dir / f"{loc}_mp_spatial.csv"
        df.to_csv(out_path, index=False)
        print(f"[DEBUG] [PID:{os.getpid()}] [{subj_id}] Saved {out_path.name}")


if __name__ == "__main__":
    processed_root = Path("./data/processed")
    if not processed_root.exists():
        print("[DEBUG] [MAIN] data/processed not found. Aborting.")
        exit(1)

    subject_dirs = [d for d in processed_root.iterdir() if d.is_dir()]
    print(f"[DEBUG] [MAIN] Found {len(subject_dirs)} subjects. Dispatching workers.")

    MAX_CORES = 10

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_CORES) as executor:
        futures = {executor.submit(process_subject, subj): subj for subj in subject_dirs}

        for future in concurrent.futures.as_completed(futures):
            subj = futures[future]
            try:
                future.result()
                print(f"[DEBUG] [MAIN] Completed: {subj.name}")
            except Exception as exc:
                print(f"[ERROR] [MAIN] {subj.name} failed: {exc}")