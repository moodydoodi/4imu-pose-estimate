"""12 input features per sensor from raw accelerometer and gyroscope."""
import numpy as np
from scipy import signal

from config import ACC_SCALE, FEAT_PER_SENSOR, G, GYR_SCALE, HIGHBAND


def _lowpass(x, fs, fc):
    if fc >= 0.45 * fs:
        return x
    b, a = signal.butter(4, fc / (fs / 2))
    return signal.filtfilt(b, a, x, axis=0)


def _to_grid(t_src, x, t_dst, fs_src, fs_dst, antialias=True):
    x = np.asarray(x, float)
    if antialias and fs_src > 1.6 * fs_dst:
        x = _lowpass(x, fs_src, 0.45 * fs_dst)
    if x.ndim == 1:
        return np.interp(t_dst, t_src, x)
    return np.stack([np.interp(t_dst, t_src, x[:, k]) for k in range(x.shape[1])], axis=1)


def gravity_direction(acc, gyr, fs, k=0.05, sigma=2.5):
    """Unit gravity direction in the sensor frame. acc in m/s2, gyr in rad/s."""
    T = len(acc)
    dt = 1.0 / fs
    g = np.empty((T, 3))
    n0 = np.linalg.norm(acc[0])
    g[0] = acc[0] / n0 if n0 > 1e-6 else np.array([0.0, 1.0, 0.0])
    mag = np.linalg.norm(acc, axis=1)
    trust = np.exp(-((mag - G) / sigma) ** 2)      # 1 at rest, 0 during impact
    an = acc / np.maximum(mag, 1e-9)[:, None]
    for i in range(1, T):
        v = g[i - 1] - dt * np.cross(gyr[i], g[i - 1])
        v += k * trust[i] * (an[i] - v)
        n = np.linalg.norm(v)
        g[i] = v / n if n > 1e-9 else g[i - 1]
    return g


def high_band_energy(acc_mag, fs_src, t_src, t_dst):
    """Sliding RMS in the impact band, resampled onto the target grid."""
    lo, hi = HIGHBAND
    hi = min(hi, 0.45 * fs_src)
    if hi <= lo or fs_src < 2.5 * lo:
        return np.zeros(len(t_dst))
    b, a = signal.butter(4, [lo / (fs_src / 2), hi / (fs_src / 2)], btype="band")
    y = signal.filtfilt(b, a, acc_mag - acc_mag.mean())
    step = float(np.median(np.diff(t_dst)))
    w = max(3, int(round(fs_src * step)))
    rms = np.sqrt(np.convolve(y ** 2, np.ones(w) / w, mode="same"))
    return np.interp(t_dst, t_src, rms)


def sensor_features(t_src, acc, gyr_deg, t_dst, fs_src, fs_dst):
    """(T_src,3) acc and gyro -> (T_dst, FEAT_PER_SENSOR)."""
    acc = np.asarray(acc, float)
    gyr = np.deg2rad(np.asarray(gyr_deg, float))

    hb = high_band_energy(np.linalg.norm(acc, axis=1), fs_src, t_src, t_dst)

    acc_d = _to_grid(t_src, acc, t_dst, fs_src, fs_dst)
    gyr_d = _to_grid(t_src, gyr, t_dst, fs_src, fs_dst)

    grav = gravity_direction(acc_d, gyr_d, fs_dst)
    lin = acc_d - G * grav

    F = np.concatenate([
        grav,
        lin / ACC_SCALE,
        gyr_d / GYR_SCALE,
        (np.linalg.norm(acc_d, axis=1) / ACC_SCALE)[:, None],
        (np.linalg.norm(gyr_d, axis=1) / GYR_SCALE)[:, None],
        (hb / ACC_SCALE)[:, None],
    ], axis=1)
    assert F.shape[1] == FEAT_PER_SENSOR, F.shape
    return F
