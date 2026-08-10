"""Virtual AX6 signals from a pose sequence (--npz from AMASS, or --pose).

Places a sensor on each limb, derives specific force and angular rate, then
degrades the clean signal with a noise profile measured from real recordings.
Mounting is drawn per recording.

Writes <sensor>_{aligned,mp_spatial,segment}.csv, *_gt_3d.csv and
synthesis_info.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import mounting

G_REF = 9.80665
ACC = ["acc_x", "acc_y", "acc_z"]
GYR = ["gyr_x", "gyr_y", "gyr_z"]

SEGMENTS = {
    "left_wrist":  (13, 15, 11),   # elbow -> wrist, reference shoulder
    "right_wrist": (14, 16, 12),
    "left_ankle":  (25, 27, 23),   # knee -> ankle, reference hip
    "right_ankle": (26, 28, 24),
}
# (proximal, joint, distal) for the anatomical frame, as in estimate_mount.py
CHAIN = {"left_wrist": (11, 13, 15), "right_wrist": (12, 14, 16),
         "left_ankle": (23, 25, 27), "right_ankle": (24, 26, 28)}
# sensor offset in the segment frame (x, along the bone, z), in metres
SENSOR_OFFSET = np.array([0.015, -0.030, 0.010])

# Frequencies inside the impact band that cannot be an image of a 60 or 120 Hz
# source. They give the baseline the image frequencies are compared against.
CONTROL_FREQUENCIES = [27.0, 33.0, 47.0, 53.0, 67.0, 73.0, 87.0]

# Calibrated on 24 real sensor streams (no images possible, they are measured),
# 8 corrected synthetic ones and 25 recordings of the faulty 800-set:
#   real           0.87 - 1.51
#   corrected      0.93 - 1.94
#   faulty set     4.55 - 249      (per recording, worst of its four sensors)
# At 3.0 every faulty recording is rejected and nothing clean is.
IMAGE_TONE_LIMIT = 3.0

ACC_RANGE = 16.0 * G_REF     # AX6: +-16 g
GYR_RANGE = 2000.0           # AX6: +-2000 dps


# --------------------------------------------------------------------------- IO
def load_pose(path: Path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    js = sorted({int(c[1:].split("_")[0]) for c in df.columns
                 if c.startswith("j") and c.endswith(("_x", "_y", "_z"))})
    if not js:
        raise ValueError(f"{path.name}: no jN_x/jN_y/jN_z columns found")
    P = np.stack([df[[f"j{j}_x", f"j{j}_y", f"j{j}_z"]].to_numpy(float) for j in js], axis=1)
    t = df["time"].to_numpy(float) if "time" in df.columns else np.arange(len(df)) / 50.0
    return P, t, df


def load_npz(path: Path):
    d = np.load(path)
    P = d["joints"].astype(float)
    fps = float(d["fps"]) if "fps" in d else 50.0
    if not np.isfinite(fps) or fps <= 0:
        raise SystemExit(f"{path.name}: frame rate is {fps}. Regenerate the pose "
                         f"with amass_to_pose.py.")
    root = d["root"].astype(float) if "root" in d.files else None
    seg = d["seg_rot"].astype(float) if "seg_rot" in d.files else None
    return P, np.arange(len(P)) / fps, root, seg


# ------------------------------------------------------------------- signals
def lowpass(x: np.ndarray, fs: float, cutoff: float) -> np.ndarray:
    """Zero-phase low pass. scipy if available, otherwise a Gaussian kernel."""
    if cutoff <= 0 or cutoff >= fs / 2:
        return x
    try:
        from scipy.signal import butter, filtfilt
        b, a = butter(4, cutoff / (fs / 2), btype="low")
        return filtfilt(b, a, x, axis=0)
    except Exception:
        sigma = max(1.0, fs / (2 * np.pi * cutoff))
        r = int(np.ceil(3 * sigma))
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
        k /= k.sum()
        pad = np.pad(x, ((r, r), (0, 0)), mode="edge")
        return np.stack([np.convolve(pad[:, i], k, mode="valid")
                         for i in range(x.shape[1])], axis=1)


def lowpass_rotations(R: np.ndarray, fs: float, cutoff: float) -> np.ndarray:
    """Low-pass (T,...,3,3) rotations, re-orthonormalised by polar decomposition."""
    R = np.asarray(R, float)
    if cutoff <= 0 or cutoff >= fs / 2:
        return R
    shape = R.shape
    sm = lowpass(R.reshape(len(R), -1), fs, cutoff).reshape(shape)
    U, _, Vt = np.linalg.svd(sm)
    out = U @ Vt
    flip = np.linalg.det(out) < 0            # undo reflections
    if np.any(flip):
        U = U.copy()
        U[flip, ..., -1] *= -1
        out = U @ Vt
    return out


def image_frequencies(fs: float, fs_src: float, harmonics: int = 4):
    """-> source-rate images folded into [0, fs/2].

    Interpolation repeats the baseband at every multiple of the source rate.
    On the sensor grid those repeats fold back below Nyquist: a 120 Hz source
    on a 200 Hz grid gives 40 and 80 Hz, a 60 Hz source gives 20, 40, 60 and
    80 Hz - so checking only |fs - fs_src| misses most of them.
    """
    found = []
    for k in range(1, harmonics + 1):
        f = (k * fs_src) % fs
        if f > fs / 2:
            f = fs - f
        if 1.0 < f < fs / 2 - 1.0:
            found.append(f)
    out = []                              # merge near-duplicates: fs is not integer,
    for f in sorted(found):               # so 80.00 and 79.99 are the same image
        if not out or f - out[-1] > 1.0:
            out.append(f)
    return [round(f, 1) for f in out]


def image_tone_ratio(sig: np.ndarray, fs: float, images,
                     lo: float = 20.0, hi: float = 90.0):
    """-> (worst image relative to the control frequencies, where it sits).

    Two normalisations, both needed:

    A tone is measured against its LOCAL surroundings, not against the whole
    band. At the edge of the low pass the band median lies far below the local
    level, so every image there would look like a tone.

    That local peak-to-median is then divided by the same quantity at
    CONTROL_FREQUENCIES, where no image can be. Peak-to-median is biased
    upwards for any structured spectrum - real recordings score 2.5 to 5.8
    unnormalised - and dividing by the control value removes that bias. Clean
    data then sits at 1, whatever the spectrum looks like otherwise.
    """
    x = np.linalg.norm(np.asarray(sig, float), axis=1)
    x = x - x.mean()
    nseg = int(2 ** int(np.log2(max(len(x) // 4, 256))))
    inside = [f for f in np.atleast_1d(images) if lo <= f <= hi]
    if nseg < 256 or nseg > len(x) or not inside:
        return float("nan"), float("nan")
    w = np.hanning(nseg)
    P = np.mean([np.abs(np.fft.rfft(x[i:i + nseg] * w)) ** 2
                 for i in range(0, len(x) - nseg + 1, nseg // 2)], axis=0)
    f = np.fft.rfftfreq(nseg, 1.0 / fs)

    def peak_over_ring(f0):
        near = (f >= f0 - 1.5) & (f <= f0 + 1.5)
        ring = (f >= f0 - 8.0) & (f <= f0 + 8.0) & ~near
        if not near.any() or not ring.any():
            return float("nan")
        return float(P[near].max() / max(np.median(P[ring]), 1e-30))

    control = [v for v in (peak_over_ring(c) for c in CONTROL_FREQUENCIES)
               if np.isfinite(v)]
    if not control:
        return float("nan"), float("nan")
    base = max(float(np.median(control)), 1e-12)

    worst, at = float("nan"), float("nan")
    for f0 in inside:
        r = peak_over_ring(f0)
        if not np.isfinite(r):
            continue
        r /= base
        if not np.isfinite(worst) or r > worst:
            worst, at = r, f0
    return worst, at


def resample_uniform(P: np.ndarray, t: np.ndarray, fs_out: float):
    """Interpolate onto a uniform time grid."""
    t0, t1 = float(t[0]), float(t[-1])
    tn = np.arange(t0, t1, 1.0 / fs_out)
    T, J, _ = P.shape
    out = np.empty((len(tn), J, 3))
    for j in range(J):
        for a in range(3):
            out[:, j, a] = np.interp(tn, t, P[:, j, a])
    return out, tn


def _matrix_to_quat(R: np.ndarray) -> np.ndarray:
    M = np.asarray(R, float).reshape(-1, 3, 3)
    q = np.empty((len(M), 4), float)
    for i, A in enumerate(M):
        tr = float(np.trace(A))
        if tr > 0:
            s = 2.0 * np.sqrt(tr + 1.0)
            q[i] = [(0.25 * s), (A[2, 1] - A[1, 2]) / s,
                    (A[0, 2] - A[2, 0]) / s, (A[1, 0] - A[0, 1]) / s]
        else:
            j = int(np.argmax(np.diag(A)))
            if j == 0:
                s = 2.0 * np.sqrt(max(1.0 + A[0, 0] - A[1, 1] - A[2, 2], 1e-12))
                q[i] = [(A[2, 1] - A[1, 2]) / s, 0.25 * s,
                        (A[0, 1] + A[1, 0]) / s, (A[0, 2] + A[2, 0]) / s]
            elif j == 1:
                s = 2.0 * np.sqrt(max(1.0 + A[1, 1] - A[0, 0] - A[2, 2], 1e-12))
                q[i] = [(A[0, 2] - A[2, 0]) / s, (A[0, 1] + A[1, 0]) / s,
                        0.25 * s, (A[1, 2] + A[2, 1]) / s]
            else:
                s = 2.0 * np.sqrt(max(1.0 + A[2, 2] - A[0, 0] - A[1, 1], 1e-12))
                q[i] = [(A[1, 0] - A[0, 1]) / s, (A[0, 2] + A[2, 0]) / s,
                        (A[1, 2] + A[2, 1]) / s, 0.25 * s]
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    return q


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, float)
    q /= np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    w, x, y, z = [q[..., i] for i in range(4)]
    return np.stack([
        1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w),
        2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w),
        2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y),
    ], axis=-1).reshape(q.shape[:-1] + (3, 3))


def resample_rotations(R: np.ndarray, t_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    R = np.asarray(R, float)
    if len(R) != len(t_src):
        raise ValueError("rotations and source time have different lengths")
    flat = R.reshape(len(R), -1, 3, 3)
    q = _matrix_to_quat(flat).reshape(len(R), -1, 4)
    for i in range(1, len(q)):
        flip = (q[i - 1] * q[i]).sum(-1) < 0
        q[i, flip] *= -1
    hi = np.clip(np.searchsorted(t_src, t_dst, side="right"), 1, len(t_src) - 1)
    lo = hi - 1
    u = ((t_dst - t_src[lo]) / np.maximum(t_src[hi] - t_src[lo], 1e-12))[:, None]
    q0, q1 = q[lo], q[hi]
    dot = np.clip((q0 * q1).sum(-1, keepdims=True), -1.0, 1.0)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    linear = (1.0 - u[..., None]) * q0 + u[..., None] * q1
    s0 = np.sin((1.0 - u[..., None]) * theta) / np.maximum(sin_theta, 1e-12)
    s1 = np.sin(u[..., None] * theta) / np.maximum(sin_theta, 1e-12)
    qi = np.where(sin_theta < 1e-7, linear, s0 * q0 + s1 * q1)
    return _quat_to_matrix(qi).reshape((len(t_dst),) + R.shape[1:])


def so3_log(R: np.ndarray) -> np.ndarray:
    tr = np.clip((np.trace(R, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    th = np.arccos(tr)
    v = np.stack([R[:, 2, 1] - R[:, 1, 2],
                  R[:, 0, 2] - R[:, 2, 0],
                  R[:, 1, 0] - R[:, 0, 1]], axis=1)
    small = th < 1e-6
    scale = np.where(small, 0.5, th / (2 * np.sin(np.where(small, 1.0, th))))
    return v * scale[:, None]


def rotvec_to_matrix(r: np.ndarray) -> np.ndarray:
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def _perpendicular(u: np.ndarray) -> np.ndarray:
    """Any unit vector perpendicular to u."""
    a = np.zeros(3)
    a[int(np.argmin(np.abs(u)))] = 1.0
    v = np.cross(u, a)
    return v / (np.linalg.norm(v) + 1e-12)


def segment_frames(P: np.ndarray, prox: int, dist: int, ref: int = None) -> np.ndarray:
    u = P[:, dist] - P[:, prox]
    u = u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-12)
    T = len(u)
    R = np.empty((T, 3, 3))
    x = _perpendicular(u[0])
    R[0] = np.stack([x, u[0], np.cross(x, u[0])], axis=1)

    for t in range(1, T):
        a, b = u[t - 1], u[t]
        v = np.cross(a, b)
        s = float(np.linalg.norm(v))
        c = float(np.dot(a, b))
        if s < 1e-9:                                   # direction unchanged
            x = R[t - 1][:, 0] if c > 0 else -R[t - 1][:, 0]
        else:
            x = rotvec_to_matrix(v / s * np.arctan2(s, c)) @ R[t - 1][:, 0]
        x = x - np.dot(x, b) * b                       # keep exactly perpendicular
        n = np.linalg.norm(x)
        x = x / n if n > 1e-9 else _perpendicular(b)
        R[t] = np.stack([x, b, np.cross(x, b)], axis=1)
    return R


def anatomical_frame(P, trip):
    prox, mid, dist = trip
    upper = P[:, mid] - P[:, prox]
    lower = P[:, dist] - P[:, mid]
    y = lower / (np.linalg.norm(lower, axis=1, keepdims=True) + 1e-9)
    x = np.cross(upper, lower)
    q = np.linalg.norm(x, axis=1) / (np.linalg.norm(upper, axis=1)
                                     * np.linalg.norm(lower, axis=1) + 1e-9)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    x = x - (x * y).sum(1, keepdims=True) * y
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    return np.stack([x, y, np.cross(x, y)], axis=2), q


def mean_rotation(Rs, w):
    M = np.einsum("t,tij->ij", w, Rs)
    U, _, Vt = np.linalg.svd(M)
    Q = U @ Vt
    if np.linalg.det(Q) < 0:
        Q = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
    return Q


SEGMENT_ERROR_DEG = {
    "left_wrist": 27.0,
    "right_wrist": 16.0,
    "left_ankle": 10.0,
    "right_ankle": 10.0,
}


def draw_segment_errors(rng, scale=1.0):
    """-> constant residual rotation per sensor and its angle in degrees."""
    out, angles = {}, {}
    for name in SEGMENTS:
        target = SEGMENT_ERROR_DEG[name] * max(float(scale), 0.0)
        rv = rng.normal(0.0, np.deg2rad(target) / 1.596, 3)
        out[name] = rotvec_to_matrix(rv)
        angles[name] = float(np.rad2deg(np.linalg.norm(rv)))
    return out, angles


def synth_root_motion(T: int, fs: float, rng, amp_horiz=0.25, amp_vert=0.02):
    def band(amp, lo, hi):
        w = rng.normal(0, 1, T)
        w = lowpass(w[:, None], fs, hi)[:, 0]
        w = w - lowpass(w[:, None], fs, lo)[:, 0]
        s = w.std()
        return w / s * amp if s > 1e-9 else w
    x = band(amp_horiz, 0.15, 1.5)
    z = band(amp_horiz, 0.15, 1.5)
    y = band(amp_vert, 0.8, 3.0)        # slight bob while walking
    return np.stack([x, y, z], axis=1)


# ------------------------------------------------------------------ degrade
def coloured_noise(n: int, sigma: float, ar1: float, rng) -> np.ndarray:
    ar1 = float(np.clip(ar1, -0.95, 0.95))
    if abs(ar1) < 1e-6:
        return rng.normal(0, sigma, n)
    e = rng.normal(0, sigma * np.sqrt(1 - ar1 ** 2), n)
    out = np.empty(n)
    out[0] = rng.normal(0, sigma)
    for i in range(1, n):
        out[i] = ar1 * out[i - 1] + e[i]
    return out


def slow_drift(n: int, sigma: float, fs: float, rng, tau_s: float = 60.0) -> np.ndarray:
    if sigma <= 0:
        return np.zeros(n)
    phi = float(np.exp(-1.0 / (max(tau_s, 1e-3) * fs)))
    e = rng.normal(0, sigma * np.sqrt(1 - phi ** 2), n)
    out = np.empty(n)
    out[0] = rng.normal(0, sigma)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + e[i]
    return out


def sanitise_profiles(bank: dict, factor: float = 5.0) -> dict:
    profs = bank.get("profiles")
    if not profs:
        return bank
    vals = {"acc": [], "gyr": []}
    for pr in profs:
        for sd in pr.get("sensors", {}).values():
            for k in ("acc", "gyr"):
                ns = sd.get(k, {}).get("noise_std")
                if ns:
                    vals[k].append(float(np.median(ns)))
    med = {k: float(np.median(v)) if v else 0.0 for k, v in vals.items()}
    n_fixed = 0
    for pr in profs:
        for name, sd in pr.get("sensors", {}).items():
            for k in ("acc", "gyr"):
                d = sd.get(k)
                if not d or not d.get("noise_std"):
                    continue
                if float(np.median(d["noise_std"])) > factor * med[k] > 0:
                    d["noise_std"] = [med[k]] * 3
                    if d.get("bias_instability"):
                        d["bias_instability"] = [med[k] * 0.2] * 3
                    if d.get("ar1"):
                        d["ar1"] = [0.25] * 3
                    n_fixed += 1
                    print(f"  [profile] {pr.get('source_subject','?')} / {name} / {k}: "
                          f"unusable quiet block, replaced by the median")
    if n_fixed:
        print(f"  [profile] {n_fixed} entries replaced "
              f"(Median acc {med['acc']:.4f}, gyr {med['gyr']:.4f})")
    return bank


def degrade(sig: np.ndarray, kind: str, sp: dict, rng, fs: float,
            misalign_deg: float, jitter_scale: float):
    """sig: (T,3) clean signal in the sensor frame."""
    T = len(sig)
    if misalign_deg > 0:
        r = rng.normal(0, np.deg2rad(misalign_deg) / 2, 3)
        sig = sig @ rotvec_to_matrix(r).T
    scale = sp.get("scale", 1.0) * (1.0 + rng.normal(0, 0.005, 3))
    sig = sig * scale
    bias = np.array(sp.get("bias", [0, 0, 0]), float)
    inst = np.array(sp.get("bias_instability", [0, 0, 0]), float)
    for a in range(3):
        sig[:, a] += bias[a] + slow_drift(T, inst[a], fs, rng)
    ns = np.array(sp.get("noise_std", [0, 0, 0]), float) * jitter_scale
    ar = np.array(sp.get("ar1", [0, 0, 0]), float)
    for a in range(3):
        if ns[a] > 0:
            sig[:, a] += coloured_noise(T, ns[a], ar[a], rng)
    lsb = float(sp.get("lsb", 0.0))
    if lsb > 0:
        sig = np.round(sig / lsb) * lsb
    lim = ACC_RANGE if kind == "acc" else GYR_RANGE
    return np.clip(sig, -lim, lim)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="virtual AX6 signals from a pose sequence")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pose", help="*_gt_3d.csv with 33 MediaPipe joints")
    src.add_argument("--npz", help="npz with joints (T,J,3) and fps")
    ap.add_argument("--profile", required=True, help="JSON from sensor_noise_profile.py")
    ap.add_argument("--out", required=True, help="output directory for the recording")
    ap.add_argument("--fs", type=float, default=None,
                    help="target sample rate (default: from the profile)")
    ap.add_argument("--smooth-hz", type=float, default=None,
                    help="low pass before differentiating; default 5 Hz for --pose, "
                         "14 Hz for --npz")
    ap.add_argument("--gt-fps", type=float, default=50.0,
                    help="frame rate of the written ground truth")
    ap.add_argument("--root-motion", choices=["auto", "file", "none", "random"], default="auto",
                    help="auto: real trajectory from the npz, otherwise synthesise one")
    ap.add_argument("--root-amp", type=float, default=0.10,
                    help="amplitude of the synthetic root motion in m; 0.10 is about "
                         "normal walking, much more drowns out the limb motion")
    ap.add_argument("--misalign-deg", type=float, default=8.0)
    ap.add_argument("--segment-error-scale", type=float, default=1.0,
                    help="scale of the residual per-recording segment frame error")
    ap.add_argument("--jitter-scale", type=float, default=1.0,
                    help="factor on the measured noise (>1 means noisier)")
    ap.add_argument("--foot-impacts", action="store_true",
                    help="add short impulses at foot strike (experimental)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mount", default=None,
                    help="JSON from mount_calib.py; without it the bone axes "
                         "measured on video1 are used")
    ap.add_argument("--roll", choices=["measured", "random", "zero"], default="random",
                    help="random (default): draw the roll per recording. "
                         "measured: adopt the measured mounting of the real "
                         "sensors, which is not yet accurate enough, see MOUNT_K "
                         "in mounting.py. zero: no roll.")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    profile_bank = json.loads(Path(args.profile).read_text(encoding="utf-8"))
    profile_bank = sanitise_profiles(profile_bank)
    profile_index = None
    if profile_bank.get("profiles"):
        profile_index = int(rng.integers(len(profile_bank["profiles"])))
        chosen_profile = profile_bank["profiles"][profile_index]
        prof = {"sensors": chosen_profile["sensors"],
                "aggregate": profile_bank.get("aggregate", {}),
                "source_subject": chosen_profile.get("source_subject")}
    else:
        prof = profile_bank
    agg = prof.get("aggregate", {})
    fs = args.fs or agg.get("fs_hz", 200.0)

    if args.pose:
        P, t, df = load_pose(Path(args.pose))
        root_real, seg_rot = None, None
    else:
        P, t, root_real, seg_rot = load_npz(Path(args.npz))
        df = None

    if args.smooth_hz is None:
        # 14 Hz, not the earlier 25 Hz: that value was tuned against a signal
        # still carrying the resampling image. docs/RESULTS.md section 3.
        args.smooth_hz = 5.0 if args.pose else 14.0
        print(f"smoothing: {args.smooth_hz:.0f} Hz "
              f"({'MediaPipe pose' if args.pose else 'motion capture'})")
    if P.shape[1] < 29:
        raise SystemExit(f"The pose has only {P.shape[1]} joints; MediaPipe indices "
                         f"up to 28 are required.")
    print(f"pose: {P.shape[0]} frames, {P.shape[1]} joints, {t[-1]-t[0]:.1f} s")

    # Interpolation is not band limited, so the source rate images at
    # |fs - fs_src| and omega^2 lifts it into the impact band. Everything
    # entering the sensor position is cut below half the source rate.
    fs_src = float(1.0 / np.median(np.diff(t)))
    images = image_frequencies(fs, fs_src)
    cut = args.smooth_hz
    cap = 0.40 * fs_src
    if cut > cap:
        print(f"smoothing capped at {cap:.1f} Hz: the source runs at "
              f"{fs_src:.1f} Hz, above that there is no real signal left.")
        cut = cap
    print(f"source rate {fs_src:.1f} Hz -> sensor rate {fs:.1f} Hz, low pass {cut:.1f} Hz")
    print(f"source-rate images at " + ", ".join(f"{f:.0f}" for f in images) + " Hz"
          + (f"  ({sum(20 <= f <= 90 for f in images)} inside the impact band)"
             if images else ""))

    P, tn = resample_uniform(P, t, fs)
    T = len(tn)
    flat = lowpass(P.reshape(T, -1), fs, cut)
    P = flat.reshape(T, -1, 3)

    mode = args.root_motion
    if mode == "auto":
        mode = "file" if root_real is not None else "random"
    if mode == "file":
        if root_real is None:
            raise SystemExit("--root-motion file, but the file contains no 'root'.")
        root = resample_uniform(root_real[:, None, :], t, fs)[0][:, 0, :]
        root = lowpass(root, fs, cut)
        print(f"root motion: true trajectory from the sequence ({np.linalg.norm(root[-1]-root[0]):.1f} m travelled)")
    elif mode == "random":
        root = synth_root_motion(T, fs, rng, amp_horiz=args.root_amp)
        print(f"root motion: synthesised (amplitude {args.root_amp} m)")
    else:
        root = np.zeros((T, 3))
        print("root motion: none")

    P_world = P + root[:, None, :]      # only for the sensor computation

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dt = 1.0 / fs
    g_world = np.array([0.0, +G_REF, 0.0])     # MediaPipe: +y is down

    bone_axes = mounting.load_bone_axes(args.mount)
    drawn = mounting.draw_rolls(rng)
    rolls = drawn if args.roll == "random" else {s: 0.0 for s in mounting.SENSOR_ORDER}
    mis = {}
    for n in SEGMENTS:
        r = rng.normal(0, np.deg2rad(args.misalign_deg) / 2, 3) if args.misalign_deg > 0 \
            else np.zeros(3)
        mis[n] = rotvec_to_matrix(r)
    if args.roll == "measured":
        MT = {n: mis[n] @ mounting.mount_matrix_calibrated(n, bone_axes) for n in SEGMENTS}
        print("mounting: fully measured (estimate_mount.py against video1)")
    else:
        MT = {n: mis[n] @ mounting.mount_matrix(n, bone_axes, rolls[n]) for n in SEGMENTS}
        print("mounting: bone axis measured, roll "
              + ("drawn (" + ", ".join(f"{n.split('_')[0][0]}{n.split('_')[1][0]}"
                 f" {rolls[n]:.0f} deg" for n in SEGMENTS) + ")"
                 if args.roll == "random" else "set to zero"))

    seg_on_grid = None
    if seg_rot is not None:
        # Same low pass as the positions: the rotation reaches the sensor
        # position through the lever arm and the angular rate through so3_log.
        seg_on_grid = resample_rotations(seg_rot, t, tn)          # (T,4,3,3)
        seg_on_grid = lowpass_rotations(seg_on_grid, fs, cut)
        print("segment frames: from the SMPL rotations (roll about the bone axis "
              "preserved)")
    else:
        print("segment frames: reconstructed from joint positions (no roll about "
              "the bone axis)")

    signals, seg_signals = {}, {}
    image_ratio, image_at = {}, {}
    segment_error, segment_error_angles = draw_segment_errors(
        rng, scale=args.segment_error_scale)
    for si, (name, (prox, dist, ref)) in enumerate(SEGMENTS.items()):
        if seg_on_grid is not None:
            R = seg_on_grid[:, si]
        else:
            R = segment_frames(P, prox, dist, ref)               # (T,3,3)
        pos = P_world[:, dist] + np.einsum("tij,j->ti", R, SENSOR_OFFSET)

        vel = np.gradient(pos, dt, axis=0)
        acc_w = np.gradient(vel, dt, axis=0)
        f_world = acc_w - g_world
        acc_s = np.einsum("tji,tj->ti", R, f_world)              # R^T @ f

        Rrel = np.einsum("tji,tjk->tik", R[:-2], R[2:])          # R[t-1]^T R[t+1]
        w = so3_log(Rrel) / (2 * dt)
        gyr_s = np.rad2deg(np.vstack([w[:1], w, w[-1:]]))

        sp = prof["sensors"].get(name, {})
        acc_p = dict(sp.get("acc", {}))
        gyr_p = dict(sp.get("gyr", {}))
        acc_p["scale"] = sp.get("acc_scale_vs_9_81", 1.0)        # calibration error
        acc_p.setdefault("bias", [0.0, 0.0, 0.0])

        if args.foot_impacts and name.endswith("ankle"):
            vy = np.gradient(pos[:, 1], dt)
            hit = np.where((vy[:-1] > 0.4) & (vy[1:] <= 0.4))[0]  # downward braking
            for i in hit:
                w_len = int(0.03 * fs)
                env = np.exp(-np.linspace(0, 4, w_len))
                sl = slice(i, min(T, i + w_len))
                acc_s[sl] += (rng.normal(0, 12.0, (3,))[None, :] * env[: sl.stop - sl.start, None])
            print(f"  {name}: {len(hit)} foot impacts added")

        acc_s = acc_s @ MT[name].T
        gyr_s = gyr_s @ MT[name].T

        # Fixed rotation from the sensor frame into the anatomical frame.
        A, qual = anatomical_frame(P, CHAIN[name])
        w = np.clip(qual - 0.35, 0.0, None)
        if w.sum() < 1e-6:
            w = np.ones(len(A))
        Q = mean_rotation(np.einsum("tji,tjk->tik", A, R), w / w.sum())

        # A tone at the image frequency is an artefact, not motion content.
        acc_s = degrade(acc_s, "acc", acc_p, rng, fs, 0.0, args.jitter_scale)
        gyr_s = degrade(gyr_s, "gyr", gyr_p, rng, fs, 0.0, args.jitter_scale)
        K = Q @ MT[name].T
        seg_signals[name] = (acc_s @ K.T, gyr_s @ K.T)

        out = pd.DataFrame({"t": tn,
                            "acc_x": acc_s[:, 0], "acc_y": acc_s[:, 1], "acc_z": acc_s[:, 2],
                            "gyr_x": gyr_s[:, 0], "gyr_y": gyr_s[:, 1], "gyr_z": gyr_s[:, 2]})
        out.to_csv(out_dir / f"{name}_aligned.csv", index=False, float_format="%.6g")
        image_ratio[name], image_at[name] = image_tone_ratio(acc_s, fs, images)
        signals[name] = (acc_s, gyr_s)
        print(f"  {name}: |acc| median {np.median(np.linalg.norm(acc_s,axis=1)):6.2f} m/s^2 · "
              f"|gyr| median {np.median(np.linalg.norm(gyr_s,axis=1)):7.2f} deg/s · "
              f"image tone {image_ratio[name]:5.1f}x")

    finite = [(v, image_at[k]) for k, v in image_ratio.items() if np.isfinite(v)]
    if not finite:
        print(f"\nNOTE: no source-rate image falls inside 20-90 Hz, nothing to check.")
    else:
        worst, at = max(finite)
        if worst > IMAGE_TONE_LIMIT:
            print(f"\nWARNING: a tone at {at:.0f} Hz stands {worst:.0f}x above its "
                  f"surroundings.\nThat is an image of the source rate ({fs_src:.1f} Hz), "
                  f"pulled up by omega^2 in the\ndouble differentiation, not motion "
                  f"content. Lower --smooth-hz, or pick a\nsensor rate whose images "
                  f"avoid 20 to 90 Hz.")

    for name, (acc_s, gyr_s) in signals.items():
        b_dev = np.asarray(bone_axes[name], float)
        b_dev = b_dev / np.linalg.norm(b_dev)
        acc_rest = np.tile(-b_dev * G_REF, (16, 1))
        a2, g2 = mounting.to_mp_spatial(acc_s, gyr_s, acc_rest, name)
        pd.DataFrame({"t": tn, "acc_x": a2[:, 0], "acc_y": a2[:, 1], "acc_z": a2[:, 2],
                      "gyr_x": g2[:, 0], "gyr_y": g2[:, 1], "gyr_z": g2[:, 2]}
                     ).to_csv(out_dir / f"{name}_mp_spatial.csv", index=False,
                              float_format="%.6g")

    for name, (a2, g2) in seg_signals.items():
        E = segment_error[name]
        a2 = a2 @ E.T
        g2 = g2 @ E.T
        pd.DataFrame({"t": tn, "acc_x": a2[:, 0], "acc_y": a2[:, 1], "acc_z": a2[:, 2],
                      "gyr_x": g2[:, 0], "gyr_y": g2[:, 1], "gyr_z": g2[:, 2]}
                     ).to_csv(out_dir / f"{name}_segment.csv", index=False,
                              float_format="%.6g")

    gt_name = "synthetic_gt_3d.csv"
    if df is not None:
        df.to_csv(out_dir / gt_name, index=False)          # original, unchanged
    else:
        tg = np.arange(tn[0], tn[-1], 1.0 / args.gt_fps)
        cols = {"frame": np.arange(len(tg)), "time": tg}
        for j in range(P.shape[1]):
            for k, a in enumerate("xyz"):
                cols[f"j{j}_{a}"] = np.interp(tg, tn, P[:, j, k])
        pd.DataFrame(cols).to_csv(out_dir / gt_name, index=False, float_format="%.6g")
        print(f"ground truth: {len(tg)} frames @ {args.gt_fps:.0f} fps, pelvis-centred")

    (out_dir / "synthesis_info.json").write_text(json.dumps({
        "source_pose": args.pose or args.npz, "profile": args.profile,
        "profile_source": prof.get("source_subject"), "profile_index": profile_index,
        "fs_hz": fs, "gt_fps": args.gt_fps, "seed": args.seed,
        "smooth_hz": args.smooth_hz, "smooth_cut_hz": cut,
        "source_fps": fs_src, "image_frequencies_hz": images,
        "image_tone_ratio": image_ratio, "image_tone_at_hz": image_at,
        "image_tone_limit": IMAGE_TONE_LIMIT,
        "rotation_lowpass": bool(seg_rot is not None),
        "root_motion": mode, "root_amp_m": args.root_amp,
        "segment_frames": "smpl" if seg_rot is not None else "from positions",
        "rotation_resampling": "slerp" if seg_rot is not None else None,
        "segment_frame_error_deg": segment_error_angles,
        "mount_roll_deg": rolls,
        "misalign_deg": args.misalign_deg, "jitter_scale": args.jitter_scale,
        "foot_impacts": bool(args.foot_impacts),
    }, indent=2), encoding="utf-8")
    print(f"\nwritten to {out_dir}")


if __name__ == "__main__":
    main()
