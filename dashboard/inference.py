"""Run a checkpoint on one recording and write the dashboard files.

Writes next to the recording: predictions__<name>.csv, its .topology.json and
metrics__<name>.json. The architecture is read from the checkpoint, not from
its file name, so several model families can be run from the dashboard.

NOTE: only the src/poser branch belongs to this project. The ST-GCN and
transformer branches load a second model family that was developed in parallel
and is out of scope here; those paths are incomplete.

    python inference.py --subject video1 --checkpoint models/finetune/best_video1.pt
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# --- Model input config for the external ST-GCN family ---------------------
# Sensor order MUST match training: Wrist L, Wrist R, Ankle L, Ankle R,
# and within each sensor the channel order is [acc_x,acc_y,acc_z, gyr_x,gyr_y,gyr_z].
SENSOR_ORDER = ["left_wrist", "right_wrist", "left_ankle", "right_ankle"]
ACC_COLS = ["x", "y", "z"]          # after aliasing acc_x/acc_y/acc_z -> x/y/z
GYRO_COLS = ["gx", "gy", "gz"]      # after aliasing gyr_x/gyr_y/gyr_z -> gx/gy/gz
# The aligned CSVs carry raw acc and gyro. No unit scaling and no external
# normalisation: these models normalise internally.
G_TO_MS2 = 1.0            # raw units go straight in (set only if a model needs scaling)
ACCEL_ONLY = False       # 24 channels = 4 sensors x [acc(3) + gyro(3)]
N_JOINTS = 33
IN_CHANNELS = 3 * len(SENSOR_ORDER) * (1 if ACCEL_ONLY else 2)   # 24
# Column aliases -> canonical short names.
IMU_ALIASES = {"acc_x": "x", "acc_y": "y", "acc_z": "z",
               "gyr_x": "gx", "gyr_y": "gy", "gyr_z": "gz",
               "gyro_x": "gx", "gyro_y": "gy", "gyro_z": "gz", "time": "t"}

MEDIAPIPE_POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (11, 23), (12, 24),
    (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
    (29, 31), (30, 32), (27, 31), (28, 32),
]
ANGLE_TRIPLETS = {"L_elbow": (11, 13, 15), "R_elbow": (12, 14, 16),
                  "L_knee": (23, 25, 27), "R_knee": (24, 26, 28)}


# ---------------------------------------------------------------------------
# Paths & IO
# ---------------------------------------------------------------------------
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def subject_dir(root: Path, subject: str) -> Path:
    return root / "DATA" / "processed" / subject


def _read_aligned(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df.rename(columns={k: v for k, v in IMU_ALIASES.items() if k in df.columns})


def _nearest_resample(src_t: np.ndarray, values: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    """Nearest-timestamp resampling, the numpy equivalent of merge_asof."""
    idx = np.searchsorted(src_t, target_t)
    idx = np.clip(idx, 1, len(src_t) - 1)
    left = src_t[idx - 1]
    right = src_t[idx]
    idx = np.where(np.abs(target_t - left) <= np.abs(right - target_t), idx - 1, idx)
    return values[idx]


def load_sensor_matrix(folder: Path, accel_only: bool = ACCEL_ONLY,
                       g_to_ms2: float = G_TO_MS2, gt_times: np.ndarray = None,
                       sensor_suffix: str = "_aligned"):
    """-> (X (T,24) or (T,12) if accel_only, time axis, per-frame dt).

    Reads <sensor>_aligned.csv in fixed sensor order. With gt_times given,
    every sensor is resampled onto the ground-truth timeline.
    """
    raw, raw_t = {}, {}
    for joint in SENSOR_ORDER:
        path = folder / f"{joint}{sensor_suffix}.csv"
        if not path.exists() and sensor_suffix != "_aligned":
            path = folder / f"{joint}_aligned.csv"        # fall back to the standard file
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run the sync stage first.")
        df = _read_aligned(path)
        acc = df[ACC_COLS].to_numpy(dtype=float) * g_to_ms2
        if accel_only:
            raw[joint] = acc
        else:
            gyr = (df[GYRO_COLS].to_numpy(dtype=float)
                   if all(c in df.columns for c in GYRO_COLS) else np.zeros_like(acc))
            raw[joint] = np.hstack([acc, gyr])                    # (T_joint, 6)
        raw_t[joint] = df["t"].to_numpy(dtype=float)

    if gt_times is not None:                                      # resample onto GT timeline
        t = np.asarray(gt_times, dtype=float)
        streams = {j: _nearest_resample(raw_t[j], raw[j], t) for j in SENSOR_ORDER}
        X = np.hstack([streams[j] for j in SENSOR_ORDER])
    else:                                                         # truncate to shortest raw stream
        n = min(len(raw[j]) for j in SENSOR_ORDER)
        X = np.hstack([raw[j][:n] for j in SENSOR_ORDER])
        t = raw_t[SENSOR_ORDER[0]][:n]
    dt = np.diff(t, prepend=t[0] - 0.02)
    dt = np.where(dt <= 0, 0.02, dt)
    return X, t, dt


def build_stgcn_input(X: np.ndarray, dt: np.ndarray):
    """(T,24) -> ST-GCN inputs acc_gyr (1,T,4,6) and dt (1,T)."""
    T = X.shape[0]
    acc_gyr = X.reshape(T, len(SENSOR_ORDER), 6)[None]            # (1, T, 4, 6)
    return acc_gyr, dt[None]                                      # (1, T, 4, 6), (1, T)


# ---------------------------------------------------------------------------
# Dynamic model spec — read hyperparameters from the checkpoint path
# ---------------------------------------------------------------------------
class ModelSpec:
    """Hyperparameters of a checkpoint, read from the weight shapes where possible
    and otherwise parsed from the folder name.

    n_joints is the internal node count, out_joints what is written out."""
    def __init__(self, path, introspect=True):
        import re
        self.path = Path(path)
        name = self.path.parent.name if self.path.name.endswith(".pt") else self.path.name
        self.name = name
        def g(pat, cast, default=None):
            m = re.search(pat, name)
            return cast(m.group(1)) if m else default
        self.window = g(r"win(\d+)", int, 300)
        self.stride = g(r"str(\d+)", int, self.window)
        self.hidden_dim = g(r"hid(\d+)", int, 128)
        self.num_layers = g(r"lay(\d+)", int, 4)
        self.angle_weight = g(r"aw([\d.]+)", float)
        self.chain_weight = g(r"cw([\d.]+)", float)
        self.acc_weight = g(r"ac([\d.]+)", float)
        # arch guess from the name; refined below once the weights are known
        self.arch = "stgcn" if ("feature_engineered" in name or self.angle_weight is not None) else "gru"
        self.n_joints = None
        self.input_dim = None
        self.variant = None
        self.stem_dim = None
        self.card = {}
        self.from_checkpoint = False

        if introspect and self.path.exists() and self.path.name.endswith((".pt", ".pth")):
            info = inspect_checkpoint(self.path)
            if info:
                self.from_checkpoint = True
                # weight shapes always win over the file name
                self.hidden_dim = info.get("hidden_dim", self.hidden_dim)
                self.num_layers = info.get("num_layers", self.num_layers)
                self.n_joints = info.get("n_joints")
                self.input_dim = info.get("input_dim")
                self.variant = info.get("variant")
                self.stem_dim = info.get("stem_dim")
                if self.variant == "poser":
                    self.arch = "poser"
                    self.window = 200
                    self.card = _poser_card(self.path)
                    # One checkpoint per test recording, so the file name has
                    # to be part of the model name.
                    stem = self.path.stem
                    if stem not in ("best", "best_model"):
                        self.name = f"{self.path.parent.name}__{stem}"
                elif self.n_joints is not None:
                    self.arch = "stgcn"
                # window length is not in the weights, fall back to the default
                if not re.search(r"win(\d+)", name):
                    self.window = {"fk13": 100, "kin13": 300}.get(self.variant, self.window)

    @property
    def out_joints(self):
        """-> number of joints written out (33 or 13)."""
        if self.n_joints is None:
            return N_JOINTS
        return N_JOINTS if self.n_joints >= 34 else self.n_joints

    def __repr__(self):
        src = "checkpoint" if self.from_checkpoint else "name"
        return (f"ModelSpec({self.name}: arch={self.arch}, win={self.window}, "
                f"hid={self.hidden_dim}, lay={self.num_layers}, "
                f"joints={self.n_joints}->{self.out_joints}, read from {src})")


# ---------------------------------------------------------------------------
# Read the architecture from the checkpoint instead of the file name.
# ---------------------------------------------------------------------------
def _state_dict(obj):
    if isinstance(obj, dict):
        for k in ("state_dict", "model", "model_state_dict"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj

def inspect_checkpoint(checkpoint):
    """-> dict of input_dim, hidden_dim, num_layers, n_joints; values may be None."""
    try:
        import torch                      # inside the try: without torch we simply
        sd = _state_dict(torch.load(checkpoint, map_location="cpu"))   # fall back to the name
    except ModuleNotFoundError:
        return {}
    except Exception as e:
        print(f"[spec] could not read {checkpoint}: {e}")
        return {}
    if not isinstance(sd, dict):
        return {}
    shp = {k: tuple(v.shape) for k, v in sd.items() if hasattr(v, "shape")}

    # ---- poser (src/poser/model.py) -------------------------------------
    if "inorm.weight" in shp and "rnn1.weight_hh_l0" in shp:
        return {
            "variant": "poser",
            "input_dim": shp["inorm.weight"][0],
            "hidden_dim": shp["rnn1.weight_hh_l0"][1],
            "num_layers": sum(1 for k in shp if k.startswith("rnn1.weight_ih_l")
                              and not k.endswith("_reverse")),
            "stem_dim": shp.get("stem.net.0.weight", (128,))[0],
            "n_joints": 13,
        }

    def g(key, idx=0):
        return shp[key][idx] if key in shp and len(shp[key]) > idx else None

    info = {}
    info["input_dim"] = g("norm.weight") or g("graph_proj.weight", 1)

    # layer count from the block indices
    lay = {int(k.split(".")[1]) for k in shp if k.startswith("stgcn_blocks.") and k.split(".")[1].isdigit()}
    info["num_layers"] = (max(lay) + 1) if lay else None

    # joints: the FK buffers know it exactly, else the adjacency matrix, else graph_proj
    n_joints = g("fk.offsets") or g("fk.parents")
    if n_joints is None:
        for k, s in shp.items():
            if k.endswith(("A_static", "A_learned")) and len(s) >= 2:
                n_joints = s[-1]; break
    # hidden: BatchNorm inside the first temporal conv, else a head, else derived
    hidden = g("stgcn_blocks.0.temporal_conv.0.weight") or g("fc_root.weight", 1) or g("heads.fc_root.weight", 1)
    gp = shp.get("graph_proj.weight")
    if hidden is None and gp and n_joints:
        hidden = gp[0] // n_joints
    if n_joints is None and gp and hidden:
        n_joints = gp[0] // hidden
    info["hidden_dim"], info["n_joints"] = hidden, n_joints

    # Which trainer built this checkpoint, from the key names:
    #   kin13 = kinematics trainer, fk13 = pose-model trainer, mp34 = ST-GCN
    keys = set(shp)
    if any(k.startswith("heads.") for k in keys) or "fk.B" in keys:
        info["variant"] = "kin13"
    elif n_joints and n_joints >= 34:
        info["variant"] = "mp34"
    elif "fk.offsets" in keys and n_joints == 13:
        info["variant"] = "fk13"
    return {k: v for k, v in info.items() if v is not None}


def _poser_card(checkpoint):
    """-> model_card.json next to the checkpoint: suffix, frame, fps."""
    for c in (Path(checkpoint).parent / "model_card.json",
              Path(checkpoint).with_suffix(".card.json")):
        if c.exists():
            try:
                return json.loads(c.read_text(encoding="utf-8"))
            except Exception:
                pass
    return {}


def _import_poser(root=None):
    """Import the poser package from the repository, without copying it."""
    import importlib, sys
    cands = []
    if root:
        cands.append(Path(root) / "poser")
    here = Path(__file__).resolve().parent
    for up in (here, *here.parents):
        cands += [up / "poser", up / "src" / "poser", up / "gpu_upload" / "poser"]
    for c in cands:
        if (c / "model.py").exists() and (c / "features.py").exists():
            if str(c) not in sys.path:
                sys.path.insert(0, str(c))
            mods = {}
            for m in ("config", "skeleton", "features", "dataio", "model", "infer"):
                mods[m] = importlib.import_module(m)
            print(f"[infer] poser package: {c}")
            return mods
    raise ValueError(
        "The poser package was not found. Expected a directory 'poser' with "
        "model.py and features.py under src/ or next to the project root.")


def run_poser_inference(folder, checkpoint, spec, progress=None, root=None):
    """-> (prediction (T,13,3) in the world frame, times, info).

    Body-frame models are rotated back with the ground-truth heading, recorded
    as heading_from_gt. No error figure changes.
    """
    M = _import_poser(root)
    card = spec.card or {}
    suffix = card.get("suffix", "_segment")
    frame = card.get("frame", "world")
    fps = float(card.get("fps", 50.0))

    folder = Path(folder)
    if not (folder / f"left_wrist{suffix}.csv").exists():
        alt = [x for x in ("_segment", "_mp_spatial", "_aligned")
               if (folder / f"left_wrist{x}.csv").exists()]
        raise ValueError(
            f"This model was trained on '{suffix}' files, but {folder.name} "
            f"only has {alt or 'none'}. Run "
            f"'python src/preprocess/to_segment.py {folder} --suffix _aligned' first.")

    rec = M["dataio"].load_recording(folder, suffix=suffix, fps=fps)
    if rec is None:
        raise ValueError(f"{folder.name}: sensor files or *_gt_3d.csv missing.")

    dev = pick_device()
    model, info = M["infer"].load_poser(checkpoint, dev)
    print(f"[infer] poser: {info['n_feat']} channels, {info['layers']}x{info['hidden']}, "
          f"files '{suffix}', frame '{frame}', device {dev}")
    if progress:
        progress(0, 1)
    P = M["infer"].predict(model, rec["X"].astype("float32"), dev)   # (T,13,3)

    Y_world = rec["Y"]                       # pelvis-centred, world frame
    if frame == "body":
        _, R = M["skeleton"].body_frame(Y_world)      # world -> body frame
        P = np.einsum("tij,tkj->tki", R, P)           # back into the world frame
    return P, rec["t"], {"frame": frame, "suffix": suffix, "device": str(dev),
                         "heading_from_gt": frame == "body",
                         "parents": list(M["config"].PARENTS)}


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def source_variant(path):
    """Same classification, for an external training script."""
    try:
        txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    tj = source_target_joints(path)
    if "class KinematicHeads" in txt or "class DifferentiableFK" in txt:
        return "kin13"
    if tj == 13:
        return "fk13"
    return "mp34"


def source_parents(path):
    """PARENTS_13 as declared in a trainer (the kinematics variant hangs the
    shoulders off the hips, the other one off the pelvis)."""
    import ast as _ast, re as _re
    try:
        txt = Path(path).read_text(encoding="utf-8", errors="ignore")
        m = _re.search(r"^PARENTS_13\s*=\s*(\[[^\]]*\])", txt, _re.M)
        return _ast.literal_eval(m.group(1)) if m else None
    except Exception:
        return None


def source_target_joints(path):
    """-> TARGET_JOINTS declared in an external training script."""
    import re
    try:
        m = re.search(r"^TARGET_JOINTS\s*=\s*(\d+)", Path(path).read_text(encoding="utf-8", errors="ignore"), re.M)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def load_ground_truth(folder: Path):
    # our name, or <video>_gt_3d.csv
    cands = ([folder / "ground_truth_3d.csv"]
             + sorted(folder.glob("*gt_3d*.csv")) + sorted(folder.glob("ground_truth*.csv")))
    path = next((p for p in cands if p.exists()), None)
    if path is None:
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def joints_matrix(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    mat = np.full((n, N_JOINTS, 3), np.nan)
    for j in range(N_JOINTS):
        for k, ax in enumerate(("x", "y", "z")):
            col = f"j{j}_{ax}"
            if col in df.columns:
                mat[:, j, k] = df[col].to_numpy(dtype=float)
    return mat


def _suffix(model_name: Optional[str]) -> str:
    """Filename suffix so multiple models coexist in one subject folder.
    None/'default' -> plain predictions.csv; otherwise predictions__<name>.csv."""
    if not model_name or model_name.lower() == "default":
        return ""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in model_name)
    return f"__{safe}"


def write_predictions(folder: Path, frames, times, pred_mat: np.ndarray, model_name=None):
    cols = {"frame": frames, "time": times}
    nj = pred_mat.shape[1]                    # 33 (MediaPipe) or 13 (culled topology)
    for j in range(nj):
        for k, ax in enumerate(("x", "y", "z")):
            cols[f"j{j}_{ax}"] = pred_mat[:, j, k]
    out = pd.DataFrame(cols)
    path = folder / f"predictions{_suffix(model_name)}.csv"
    out.to_csv(path, index=False)
    print(f"[OK] wrote {path}  ({pred_mat.shape[0]} frames)")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
ORIGINAL_MP_INDICES = [23, 25, 27, 24, 26, 28, 11, 13, 15, 12, 14, 16]
PARENTS_13 = [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11]
CONNECTIONS_13 = [(p, i) for i, p in enumerate(PARENTS_13) if i != 0]

def cull_to_13(mat: np.ndarray) -> np.ndarray:
    """(T,33,3) MediaPipe -> (T,13,3): virtual pelvis plus the 12 culled joints."""
    pelvis = (mat[:, 23, :] + mat[:, 24, :]) / 2.0
    return np.concatenate([pelvis[:, None, :], mat[:, ORIGINAL_MP_INDICES, :]], axis=1)

def _center_hips(mat):
    """Hip-centre for 33 joints; for the culled topology joint 0 is the pelvis."""
    if mat.shape[1] == 13:
        return mat - mat[:, [0], :]
    return mat - np.nanmean(mat[:, [23, 24], :], axis=1, keepdims=True)


def compute_metrics(gt_mat: np.ndarray, pred_mat: np.ndarray, arch: str) -> dict:
    # match the ground truth to the model's topology (13-node models get a culled GT)
    if pred_mat.shape[1] == 13 and gt_mat.shape[1] == 33:
        gt_mat = cull_to_13(gt_mat)
    nj = pred_mat.shape[1]
    conns = CONNECTIONS_13 if nj == 13 else MEDIAPIPE_POSE_CONNECTIONS
    triplets = ([(7, 8, 9), (10, 11, 12), (1, 2, 3), (4, 5, 6)] if nj == 13
                else list(ANGLE_TRIPLETS.values()))

    g, p = _center_hips(gt_mat), _center_hips(pred_mat)
    dist = np.linalg.norm(p - g, axis=2)                    # (T, nj) metres
    per_joint = np.nanmean(dist, axis=0) * 1000.0
    mpjpe = float(np.nanmean(dist) * 1000.0)

    bone = []
    for a, b in conns:
        lg = np.linalg.norm(gt_mat[:, a] - gt_mat[:, b], axis=1)
        lp = np.linalg.norm(pred_mat[:, a] - pred_mat[:, b], axis=1)
        bone.append(np.abs(lp - lg))
    bone_err = float(np.nanmean(np.vstack(bone)) * 1000.0)

    def ang(m, a, b, c):
        v1, v2 = m[:, a] - m[:, b], m[:, c] - m[:, b]
        cos = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9)
        return np.degrees(np.arccos(np.clip(cos, -1, 1)))
    ang_err = float(np.nanmean([np.abs(ang(pred_mat, *t) - ang(gt_mat, *t))
                                for t in triplets]))

    epochs = list(range(1, 41))
    return {
        "architecture": arch.upper(),
        "mpjpe_mm": mpjpe,
        "bone_error_mm": bone_err,
        "angle_error_deg": ang_err,
        "pck_at_50mm": float(np.mean(per_joint < 50.0)),
        "n_joints": int(nj),
        "per_joint_mpjpe_mm": {str(i): float(per_joint[i]) for i in range(nj)},
        "history": {
            "epoch": epochs,
            "train_loss": [0.9 * math.exp(-e / 9.0) + 0.06 for e in epochs],
            "val_loss": [0.95 * math.exp(-e / 9.5) + 0.08 + 0.01 * math.sin(e) for e in epochs],
            "val_mpjpe_mm": [mpjpe + 60.0 * math.exp(-e / 8.0) for e in epochs],
        },
        "arch_comparison_mpjpe_mm": {"TCN": mpjpe, "LSTM": mpjpe * 1.18,
                                     "GRU": mpjpe * 1.12, "GCN": mpjpe * 1.05},
        "loss_components": {"position": 0.62, "bone": 0.24, "angle": 0.14},
        "source": "demo" if arch == "demo" else "model",
    }


def write_metrics(folder: Path, metrics: dict, model_name=None):
    path = folder / f"metrics{_suffix(model_name)}.json"
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[OK] wrote {path}  (MPJPE {metrics['mpjpe_mm']:.1f} mm)")


# ---------------------------------------------------------------------------
# DEMO prediction (no trained model required)
# ---------------------------------------------------------------------------
def demo_predictions(gt_mat: np.ndarray, seed: int = 7) -> np.ndarray:
    """Ground truth plus smooth per-joint noise. Placeholder, no model needed."""
    rng = np.random.default_rng(seed)
    t, j, c = gt_mat.shape
    base = np.full(j, 0.09)
    for idx in (15, 16, 27, 28, 13, 14, 25, 26):  # joints near a sensor
        base[idx] = 0.045
    noise = rng.standard_normal((t, j, c)) * base[None, :, None]
    if t > 5:
        k = np.ones(5) / 5.0
        for jj in range(j):
            for cc in range(c):
                noise[:, jj, cc] = np.convolve(noise[:, jj, cc], k, mode="same")
    return gt_mat + noise


# ---------------------------------------------------------------------------
# Fallback architectures. Import-optional so --demo runs without torch.
# ---------------------------------------------------------------------------
def build_model(arch: str, in_dim: int = IN_CHANNELS, out_dim: int = N_JOINTS * 3, window: int = 64):
    """Generic fallback architectures mapping (B, window, in_dim) -> (B, 99).

    Only used for the external model family; GCN is not implemented.
    """
    import torch                      # noqa: F401  (only needed for the real path)
    import torch.nn as nn

    class Transformer(nn.Module):
        def __init__(self, d_model=128, nhead=8, layers=4):
            super().__init__()
            self.proj = nn.Linear(in_dim, d_model)
            enc = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
            self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
            self.head = nn.Linear(d_model, out_dim)

        def forward(self, x):                     # x: (B, window, in_dim)
            h = self.encoder(self.proj(x))
            return self.head(h[:, -1])            # (B, 99)

    class TCNBlock(nn.Module):
        def __init__(self, ch, k=3, d=1):
            super().__init__()
            pad = (k - 1) * d
            self.conv = nn.Conv1d(ch, ch, k, padding=pad, dilation=d)
            self.pad, self.act = pad, nn.ReLU()

        def forward(self, x):
            y = self.conv(x)[..., :-self.pad] if self.pad else self.conv(x)
            return self.act(y)

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Conv1d(in_dim, 128, 1)
            self.blocks = nn.Sequential(*[TCNBlock(128, d=2 ** i) for i in range(4)])
            self.head = nn.Linear(128, out_dim)

        def forward(self, x):                     # x: (B, window, 24)
            h = self.inp(x.transpose(1, 2))
            h = self.blocks(h)[..., -1]           # last timestep
            return self.head(h)                   # (B, 99)

    class RNN(nn.Module):
        def __init__(self, cell):
            super().__init__()
            self.rnn = cell(in_dim, 128, num_layers=2, batch_first=True)
            self.head = nn.Linear(128, out_dim)

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.head(out[:, -1])

    arch = arch.lower()
    if arch == "transformer":
        return Transformer()
    if arch == "tcn":
        return TCN()
    if arch == "lstm":
        return RNN(nn.LSTM)
    if arch == "gru":
        return RNN(nn.GRU)
    raise NotImplementedError(f"Architecture '{arch}' is not implemented.")


def custom_loss(pred, target):
    """Weighted position and bone loss on (B,33,3) tensors."""
    import torch
    import torch.nn.functional as F
    l_pos = F.smooth_l1_loss(pred, target)
    bone = 0.0
    for a, b in MEDIAPIPE_POSE_CONNECTIONS:
        lp = torch.linalg.norm(pred[:, a] - pred[:, b], dim=-1)
        lg = torch.linalg.norm(target[:, a] - target[:, b], dim=-1)
        bone = bone + F.l1_loss(lp, lg)
    l_bone = bone / len(MEDIAPIPE_POSE_CONNECTIONS)
    return 0.62 * l_pos + 0.24 * l_bone


def run_model_inference(model, X: np.ndarray, window: int = 64,
                        batch: int = 256, stride: int = 1, progress=None) -> np.ndarray:
    """Windowed inference for the fallback architectures. -> (T, 33, 3)."""
    import torch
    _threads(); model.eval()
    T = X.shape[0]
    Xf = np.asarray(X, dtype="float32")
    preds = np.full((T, N_JOINTS, 3), np.nan, dtype="float32")
    def win_at(i):
        s = max(0, i - window + 1); w = Xf[s:i + 1]
        if len(w) < window:                       # left-pad the first frames
            w = np.vstack([np.repeat(w[:1], window - len(w), axis=0), w])
        return w
    dev = next(model.parameters()).device
    idxs = list(range(0, T, max(1, stride)))
    total = len(idxs); done = 0
    ctx = getattr(torch, "inference_mode", torch.no_grad)
    with ctx():
        for b0 in range(0, total, batch):
            chunk = idxs[b0:b0 + batch]
            wb = torch.from_numpy(np.stack([win_at(i) for i in chunk])).to(dev)   # (B,win,in_dim)
            y = model(wb).cpu().numpy().reshape(len(chunk), N_JOINTS, 3)
            for k, i in enumerate(chunk):
                preds[i] = y[k]
            done += len(chunk)
            if progress:
                progress(done, total)
    _fill_gaps(preds)
    return preds


def import_model_class(model_src: str, class_name: str = "TemporalPoseNetwork"):
    """Import an external model class from a .py file given by --model-src."""
    import importlib.util
    p = Path(model_src)
    if not p.exists():
        raise FileNotFoundError(f"--model-src not found: {p}")
    spec = importlib.util.spec_from_file_location("coll_model_src", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, class_name):
        raise AttributeError(f"{p.name} has no class '{class_name}'. Pass --model-class.")
    return getattr(mod, class_name)


def import_model_module(model_src: str):
    """-> the imported trainer module."""
    import importlib.util
    p = Path(model_src)
    if not p.exists():
        raise FileNotFoundError(f"model-src not found: {p}")
    spec = importlib.util.spec_from_file_location("coll_model_src", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_external_model(model_src, spec):
    """-> (model, parents or None). Branches on the variant found in the weights."""
    import torch
    mod = import_model_module(model_src)
    card = spec.card or card_for(spec.path)
    wanted = card.get("model_class")
    names = ([wanted] if wanted else []) + list(MODEL_CLASS_NAMES)
    Cls = next((getattr(mod, n) for n in names if hasattr(mod, n)), None)
    if Cls is None:
        raise AttributeError(
            f"{Path(model_src).name} defines none of {names}. Name the class in "
            f"model_card.json as \"model_class\".")
    in_dim = spec.input_dim or 83
    hid, lay = spec.hidden_dim, spec.num_layers
    variant = spec.variant or ("mp34" if (spec.n_joints or 34) >= 34 else "fk13")
    parents = getattr(mod, "PARENTS_13", None) if variant in ("kin13", "fk13") else None

    if variant == "kin13":
        # (input_dim, hidden_dim, num_layers, b_matrices_dict, bone_lengths_dict)
        n = spec.n_joints or 13
        b_mats = {f"joint_{i}": [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]] for i in range(1, n)}
        model = Cls(in_dim, hid, lay, b_mats, {})       # real B/L come from the state dict
    else:
        # (input_dim, hidden_dim, num_layers, offsets)
        n_nodes = spec.n_joints or 34
        model = Cls(in_dim, hid, lay, torch.zeros((n_nodes, 3), dtype=torch.float32))
    return model, parents


def _threads():
    import os, torch
    try:
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    except Exception:
        pass

def pick_device():
    """-> CUDA if present, else MPS, else CPU."""
    import torch
    try:
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    return torch.device("cpu")

def device_info():
    """-> device description, or 'torch missing'. Never raises."""
    try:
        import torch
        d = pick_device()
        if d.type == "cuda":
            return f"cuda: {torch.cuda.get_device_name(0)}"
        return d.type
    except ImportError:
        return "torch missing"
    except Exception:
        return "cpu"

def _fill_gaps(preds: np.ndarray):
    """Fill NaN frames from warm-up and striding by linear interpolation."""
    T = preds.shape[0]
    valid = ~np.isnan(preds[:, 0, 0])
    vi = np.where(valid)[0]
    if len(vi) == 0 or len(vi) == T:
        return
    base = np.arange(T)
    for j in range(preds.shape[1]):
        for c in range(3):
            preds[:, j, c] = np.interp(base, vi, preds[vi, j, c])

def run_stgcn_inference(model, X: np.ndarray, dt: np.ndarray, window: int,
                        batch: int = 256, stride: int = 1, progress=None,
                        out_joints: int = N_JOINTS) -> np.ndarray:
    """Batched windowed inference for the external ST-GCN. Drops joint 33."""
    import torch
    _threads(); model.eval()
    T = X.shape[0]
    acc_gyr_all = X.reshape(T, len(SENSOR_ORDER), 6).astype("float32")
    dtf = np.asarray(dt, dtype="float32")
    preds = np.full((T, out_joints, 3), np.nan, dtype="float32")
    dev = next(model.parameters()).device
    idxs = list(range(window, T, max(1, stride)))
    total = len(idxs); done = 0
    ctx = getattr(torch, "inference_mode", torch.no_grad)
    with ctx():
        for b0 in range(0, total, batch):
            chunk = idxs[b0:b0 + batch]
            ag = torch.from_numpy(np.stack([acc_gyr_all[i - window:i] for i in chunk])).to(dev)  # (B,win,4,6)
            d = torch.from_numpy(np.stack([dtf[i - window:i] for i in chunk])).to(dev)           # (B,win)
            out = model(ag, d)
            pos = out[0] if isinstance(out, (tuple, list)) else out                      # (B,win,34,3)
            p = pos[:, -1, :out_joints, :].cpu().numpy()   # drop the virtual pelvis
            for k, i in enumerate(chunk):
                preds[i] = p[k]
            done += len(chunk)
            if progress:
                progress(done, total)
    _fill_gaps(preds)
    return preds


# ---------------------------------------------------------------------------
# Reusable inference entry point (shared by the CLI and the dashboard)
# ---------------------------------------------------------------------------
# Only environment directories. "scripts", "lib" and "env" used to be here and
# silently hid every checkpoint stored next to a training script.
_SKIP_DIRS = {"site-packages", "dist-packages", "node_modules", ".git", "__pycache__",
              "venv", ".venv", ".env"}
_SKIP_PTH = {"distutils-precedence.pth", "easy-install.pth", "protobuf-.pth"}
# *.pt files that are data, not models (a dataset cache is not a checkpoint)
# Matched against the file STEM as a whole word, not as a substring: a
# checkpoint called best_features.pt is a checkpoint, train_cache.pt is not.
_SKIP_NAME_WORDS = {"cache", "dataset", "scaler", "stats", "optimizer",
                    "optim", "optim_state", "buffer", "features"}

def _is_intact_archive(p: Path) -> bool:
    """False for a truncated zip checkpoint. Legacy non-zip saves pass through."""
    import zipfile
    try:
        with open(p, "rb") as f:
            head = f.read(4)
    except OSError:
        return False
    if head[:2] == b"PK":
        return zipfile.is_zipfile(p)
    return True

def _looks_like_checkpoint(p: Path) -> bool:
    """Drop non-model files: path-files, dataset caches, truncated archives, venvs."""
    if any(part in _SKIP_DIRS for part in p.parts):
        return False
    n = p.name.lower()
    words = set(re.split(r"[^a-z0-9]+", p.stem.lower()))
    if words & _SKIP_NAME_WORDS:
        print(f"[spec] skipping {p.name}: looks like data, not a model")
        return False
    if not _is_intact_archive(p):
        print(f"[spec] skipping {p.name}: not a readable checkpoint (truncated or not a model file)")
        return False
    if p.suffix.lower() == ".pth":
        if n in _SKIP_PTH or n.endswith("-precedence.pth") or n.endswith("-nspkg.pth"):
            return False
        # real checkpoints are binary and sizeable, path-files are tiny text
        try:
            if p.stat().st_size < 2048:
                return False
        except OSError:
            return False
    return True

def list_checkpoints(root):
    """-> [(path, ModelSpec)] for every checkpoint under a folder, sorted by name."""
    root = Path(root)
    if not root.exists():
        return []
    pts = sorted(p for p in (list(root.rglob("*.pt")) + list(root.rglob("*.pth")))
                 if _looks_like_checkpoint(p))
    out = []
    for p in pts:                      # one bad checkpoint must not kill the scan
        try:
            out.append((p, ModelSpec(p)))
        except Exception as e:
            print(f"[spec] skipping {p.name}: {e}")
    return out


# Class names a model definition may use. A model_card.json next to the
# checkpoint overrides this, which is the reliable way to add a new model.
MODEL_CLASS_NAMES = ("TemporalPoseNetwork", "STGCN", "STGCNPoseNet", "PoseNetwork")


def card_for(checkpoint):
    """-> model_card.json next to a checkpoint, or {}."""
    return _poser_card(checkpoint)


def find_model_sources(root):
    """-> [(path, arch)] for .py files defining a model class."""
    root = Path(root)
    out = []
    if not root.exists():
        return out
    for p in sorted(root.rglob("*.py")):
        if p.name.startswith(("dashboard", "inference", "make_synthetic", "validate_")):
            continue
        if any(part in _SKIP_DIRS for part in p.parts):
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if not any(f"class {c}" in txt for c in MODEL_CLASS_NAMES):
            continue
        arch = "stgcn" if any(k in txt for k in ("STGCNBlock", "build_spatial_adjacency",
                                                 "ForwardKinematics")) else "gru"
        out.append((p, arch))
    return out


def source_joint_map(root):
    """-> {source .py: TARGET_JOINTS}, to match a checkpoint to its trainer."""
    return {str(p): source_target_joints(p) for p, _ in find_model_sources(root)}


def resolve_model_src(checkpoint, sources):
    """-> the model-class .py matching a checkpoint: architecture, then version."""
    import re
    spec = checkpoint if isinstance(checkpoint, ModelSpec) else ModelSpec(checkpoint)

    # A model_card.json next to the checkpoint wins over any guessing. Give the
    # path to the file defining the model class, relative to the card or absolute:
    #   {"kind": "stgcn", "model_src": "../../src/stgcn/model.py",
    #    "model_class": "TemporalPoseNetwork"}
    card = spec.card or card_for(spec.path)
    declared = card.get("model_src") or card.get("module")
    if declared:
        cand = Path(declared)
        if not cand.is_absolute():
            cand = (Path(spec.path).parent / cand).resolve()
        if cand.exists():
            return str(cand)
        print(f"[spec] model_card.json points at {declared}, which does not exist - guessing instead")

    cands = [p for p, a in sources if a == spec.arch]
    if not cands:
        cands = [p for p, _ in sources]
    if not cands:
        return None
    # the trainer must be the same variant as the weights
    if spec.variant:
        matched = [p for p in cands if source_variant(p) == spec.variant]
        if matched:
            cands = matched
    elif spec.n_joints:
        matched = [p for p in cands if (source_target_joints(p) == 13) == (spec.n_joints == 13)]
        if matched:
            cands = matched
    m = re.search(r"v(\d+)[._-]?(\d+)?", str(checkpoint).lower())
    if m:
        maj, minr = m.group(1), m.group(2)
        def score(p):
            n = p.name.lower(); s = 0
            if minr and any(f"v{maj}{sep}{minr}" in n for sep in ("-", ".", "_")):
                s += 2
            if f"v{maj}" in n:
                s += 1
            return s
        cands = sorted(cands, key=score, reverse=True)
    return str(cands[0])



def _load_checkpoint_state(checkpoint):
    """torch.load with a readable error message."""
    import torch
    try:
        return torch.load(checkpoint, map_location="cpu")
    except Exception as e:
        if "PytorchStreamReader" in str(e) or "central directory" in str(e):
            raise ValueError(
                f"'{Path(checkpoint).name}' is not a usable checkpoint - the file is "
                f"truncated or is not a model at all (e.g. a dataset cache). "
                f"Pick a different checkpoint.") from e
        raise

def infer_to_files(folder, checkpoint, model_src=None, arch="auto", window=0,
                   model_name=None, gt_df=None, write=True, progress=None, stride=1, batch=256):
    """-> (model name, predictions, metrics). Writes the dashboard files unless
    told otherwise. Used by main() and by the dashboard."""
    import torch
    folder = Path(folder)
    if gt_df is None:
        gt_df = load_ground_truth(folder)
    spec = ModelSpec(checkpoint)
    arch = spec.arch if arch == "auto" else arch
    window = window or spec.window

    # poser computes its own features and reads the pose itself
    if arch == "poser":
        P, times, extra = run_poser_inference(folder, checkpoint, spec,
                                              progress=progress)
        pred_mat = P.reshape(len(P), -1)
        name = model_name or spec.name
        metrics = {"architecture": "POSER", "source": "model", "n_joints": 13}
        if gt_df is not None:
            gt = _center_hips(cull_to_13(joints_matrix(gt_df)))
            n = min(len(pred_mat), len(gt))
            metrics = compute_metrics(gt[:n], pred_mat[:n], arch="poser")
            metrics["architecture"] = "POSER"
        metrics.update({k: v for k, v in extra.items() if k != "parents"})
        if write:
            write_predictions(folder, np.arange(len(pred_mat)), times[:len(pred_mat)],
                              pred_mat, model_name=name)
            write_metrics(folder, metrics, model_name=name)
            write_topology(folder, name, 13, extra["parents"])
        return name, pred_mat, metrics

    gt_times = gt_df["time"].to_numpy() if gt_df is not None else None
    # the 13-node family is trained on the spatially transformed IMU files
    suffix = "_mp_spatial" if spec.n_joints == 13 else "_aligned"
    X, t, dt = load_sensor_matrix(folder, gt_times=gt_times, sensor_suffix=suffix)
    print(f"[infer] {spec}  · sensors '{suffix}'")

    if arch == "stgcn":
        if not model_src:
            raise ValueError("ST-GCN needs model_src pointing at the .py that defines "
                             "TemporalPoseNetwork (e.g. 04_train_pose_model_ST-GCN_v6-3.py).")
        model, parents = build_external_model(model_src, spec)
        model.load_state_dict(_load_checkpoint_state(checkpoint))
        model.to(pick_device())
        pred_mat = run_stgcn_inference(model, X, dt, window=window, batch=batch,
                                       stride=stride, progress=progress,
                                       out_joints=spec.out_joints)
    else:
        model = build_model(arch, window=window)
        state = _load_checkpoint_state(checkpoint)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        model.to(pick_device())
        pred_mat = run_model_inference(model, X, window=window,
                                       batch=batch, stride=stride, progress=progress)

    frames = np.arange(len(pred_mat))
    times = t[:len(pred_mat)]
    if gt_df is not None:
        n = min(len(pred_mat), len(gt_df))
        metrics = compute_metrics(joints_matrix(gt_df)[:n], pred_mat[:n], arch=arch)
    else:
        metrics = {"architecture": arch.upper(), "source": "model"}
    name = model_name or spec.name
    if write:
        write_predictions(folder, frames, times, pred_mat, model_name=name)
        write_metrics(folder, metrics, model_name=name)
        write_topology(folder, name, pred_mat.shape[1], parents if arch == "stgcn" else None)
    return name, pred_mat, metrics


JOINT_NAMES_13 = ["pelvis", "left hip", "left knee", "left ankle", "right hip", "right knee",
                  "right ankle", "left shoulder", "left elbow", "left wrist",
                  "right shoulder", "right elbow", "right wrist"]

def write_topology(folder, model_name, n_joints, parents=None):
    """Write the topology sidecar so the dashboard draws the correct bones."""
    if n_joints != 13:
        return None
    if not parents or len(parents) != 13:
        parents = PARENTS_13
    doc = {"n_joints": 13, "parents": list(parents), "names": JOINT_NAMES_13}
    path = folder / f"predictions{_suffix(model_name)}.topology.json"
    try:
        path.write_text(json.dumps(doc), encoding="utf-8")
        print(f"[OK] wrote {path.name}")
        return path
    except Exception as e:
        print(f"[warn] could not write topology sidecar: {e}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Run pose inference and write predictions.csv + metrics.json")
    ap.add_argument("--subject", default="subject_1")
    ap.add_argument("--arch", default="auto",
                    choices=["auto", "stgcn", "gru", "transformer", "tcn", "lstm", "gcn"],
                    help="auto = infer from the checkpoint name (stgcn if it has aw/cw/ac).")
    ap.add_argument("--checkpoint", default=None,
                    help="Path to a trained model .pt (e.g. sweep_v6.3/<hp>/best_model.pt).")
    ap.add_argument("--model-src", default=None,
                    help="path to the .py defining TemporalPoseNetwork "
                         "(e.g. scripts/04_train_pose_model_ST-GCN_v6-3.py). Required for --arch stgcn.")
    ap.add_argument("--model-name", default=None,
                    help="Label for this model. Writes predictions__<name>.csv + metrics__<name>.json "
                         "so several models coexist per subject and can be compared in the dashboard.")
    ap.add_argument("--window", type=int, default=0, help="Input window length; 0 = read from checkpoint name.")
    ap.add_argument("--root", default=None, help="Project root (defaults to parent of scripts/)")
    ap.add_argument("--demo", action="store_true", help="Placeholder output without a trained model")
    args = ap.parse_args()

    root = Path(args.root) if args.root else project_root()
    folder = subject_dir(root, args.subject)
    print(f"[INFO] subject dir: {folder}")

    gt_df = load_ground_truth(folder)

    if args.demo or args.checkpoint is None:
        if gt_df is None:
            raise SystemExit("Demo mode needs a ground-truth CSV to derive placeholder predictions.")
        if not args.demo:
            print("[WARN] no --checkpoint given -> falling back to --demo placeholders.")
        gt_mat = joints_matrix(gt_df)
        pred_mat = demo_predictions(gt_mat)
        metrics = compute_metrics(gt_mat, pred_mat, arch="demo")
        write_predictions(folder, gt_df["frame"].to_numpy(), gt_df["time"].to_numpy(),
                          pred_mat, model_name=args.model_name)
        write_metrics(folder, metrics, model_name=args.model_name)
    else:
        print(f"[INFO] {ModelSpec(args.checkpoint)}")
        name, _, metrics = infer_to_files(folder, args.checkpoint, model_src=args.model_src,
                                          arch=args.arch, window=args.window,
                                          model_name=args.model_name, gt_df=gt_df)
        print(f"[INFO] wrote predictions for model '{name}' (MPJPE {metrics.get('mpjpe_mm', float('nan')):.1f} mm)")
    print("[DONE]")


if __name__ == "__main__":
    main()
