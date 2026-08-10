"""Input augmentation: yaw about gravity, small rotations, jitter, dropout."""
import torch

from config import FEAT_PER_SENSOR, SENSORS


def _rodrigues(axis, ang):
    x, y, z = axis.unbind(-1)
    c, s = torch.cos(ang), torch.sin(ang)
    C = 1 - c
    return torch.stack([
        torch.stack([c + x*x*C,   x*y*C - z*s, x*z*C + y*s], -1),
        torch.stack([y*x*C + z*s, c + y*y*C,   y*z*C - x*s], -1),
        torch.stack([z*x*C - y*s, z*y*C + x*s, c + z*z*C  ], -1),
    ], dim=-2)


def yaw_about_gravity(X, max_deg=180.0):
    B, T, _ = X.shape
    out = X.clone()
    for i in range(len(SENSORS)):
        o = i * FEAT_PER_SENSOR
        axis = torch.nn.functional.normalize(X[:, :, o:o+3], dim=-1, eps=1e-8)
        ang = (torch.rand(B, 1, device=X.device) * 2 - 1) * (max_deg * torch.pi / 180.0)
        R = _rodrigues(axis, ang.expand(B, T))
        for c in (3, 6):
            out[:, :, o+c:o+c+3] = torch.matmul(R, X[:, :, o+c:o+c+3].unsqueeze(-1)).squeeze(-1)
    return out


def small_rotation(X, max_deg=5.0):
    B, T, _ = X.shape
    out = X.clone()
    for i in range(len(SENSORS)):
        o = i * FEAT_PER_SENSOR
        a = torch.nn.functional.normalize(torch.randn(B, 3, device=X.device), dim=-1)
        ang = (torch.rand(B, device=X.device) * 2 - 1) * (max_deg * torch.pi / 180.0)
        R = _rodrigues(a, ang).unsqueeze(1)
        for c in (0, 3, 6):
            out[:, :, o+c:o+c+3] = torch.matmul(R, X[:, :, o+c:o+c+3].unsqueeze(-1)).squeeze(-1)
        out[:, :, o:o+3] = torch.nn.functional.normalize(out[:, :, o:o+3], dim=-1, eps=1e-8)
    return out


def jitter(X, acc_sigma=0.01, gyr_sigma=0.01, bias_sigma=0.01):
    """White noise plus a constant gyro offset per window."""
    out = X.clone()
    for i in range(len(SENSORS)):
        o = i * FEAT_PER_SENSOR
        out[:, :, o+3:o+6] += acc_sigma * torch.randn_like(out[:, :, o+3:o+6])
        out[:, :, o+6:o+9] += gyr_sigma * torch.randn_like(out[:, :, o+6:o+9])
        out[:, :, o+6:o+9] += bias_sigma * torch.randn(X.shape[0], 1, 3, device=X.device)
    return out


def drop_sensor(X, p=0.1):
    """Drop one sensor entirely with probability p."""
    B = X.shape[0]
    out = X.clone()
    hit = torch.rand(B, device=X.device) < p
    which = torch.randint(0, len(SENSORS), (B,), device=X.device)
    for i in range(len(SENSORS)):
        m = hit & (which == i)
        if m.any():
            o = i * FEAT_PER_SENSOR
            out[m, :, o:o+FEAT_PER_SENSOR] = 0.0
    return out


def apply(X, suffix, strength=1.0):
    if strength <= 0:
        return X
    X = yaw_about_gravity(X, 180.0) if suffix != "_segment" else small_rotation(X, 5.0)
    X = jitter(X, 0.01*strength, 0.01*strength, 0.01*strength)
    return drop_sensor(X, 0.1*strength)
