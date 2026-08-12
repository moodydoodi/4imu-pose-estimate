"""Run a trained checkpoint on one recording and write the dashboard files.

Writes next to the recording: predictions__<name>.csv, its .topology.json and
metrics__<name>.json. The architecture and the training settings are read from
the checkpoint and its model_card.json, not from the file name.

    python inference.py --subject video1 --checkpoint models/ft_s0/best_video1.pt
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# The ground-truth CSVs carry the full MediaPipe skeleton; the model works on
# the 13-joint topology and the ground truth is culled to match.
N_JOINTS = 33
ORIGINAL_MP_INDICES = [23, 25, 27, 24, 26, 28, 11, 13, 15, 12, 14, 16]
PARENTS_13 = [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11]
CONNECTIONS_13 = [(p, i) for i, p in enumerate(PARENTS_13) if i != 0]
JOINT_NAMES_13 = ["pelvis", "left hip", "left knee", "left ankle",
                  "right hip", "right knee", "right ankle",
                  "left shoulder", "left elbow", "left wrist",
                  "right shoulder", "right elbow", "right wrist"]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def subject_dir(root: Path, subject: str) -> Path:
    return root / "DATA" / "processed" / subject


# ---------------------------------------------------------------------------
# Read the architecture from the checkpoint instead of the file name
# ---------------------------------------------------------------------------
def _state_dict(obj):
    if isinstance(obj, dict):
        for k in ("state_dict", "model", "model_state_dict"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj


def inspect_checkpoint(checkpoint):
    """-> dict with input_dim, hidden_dim, num_layers, stem_dim, n_joints.

    Empty if the file is not a checkpoint of this model, or if torch is missing.
    """
    try:
        import torch
        sd = _state_dict(torch.load(checkpoint, map_location="cpu"))
    except ModuleNotFoundError:
        return {}
    except Exception as e:
        print(f"[spec] could not read {checkpoint}: {e}")
        return {}
    if not isinstance(sd, dict):
        return {}
    shp = {k: tuple(v.shape) for k, v in sd.items() if hasattr(v, "shape")}

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
    return {}


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


def card_for(checkpoint):
    """-> model_card.json next to a checkpoint, or {}."""
    return _poser_card(checkpoint)


class ModelSpec:
    """What a checkpoint contains, read from the weight shapes.

    arch is 'poser' for a checkpoint of this project and 'unknown' otherwise.
    """

    WINDOW = 200          # 4 s at 50 Hz, fixed by the training setup

    def __init__(self, path, introspect=True):
        self.path = Path(path)
        self.name = (self.path.parent.name if self.path.name.endswith(".pt")
                     else self.path.name)
        self.arch = "unknown"
        self.window = self.WINDOW
        self.hidden_dim = None
        self.num_layers = None
        self.n_joints = None
        self.input_dim = None
        self.stem_dim = None
        self.card = {}
        self.from_checkpoint = False

        if introspect and self.path.exists() and self.path.suffix in (".pt", ".pth"):
            info = inspect_checkpoint(self.path)
            if info.get("variant") == "poser":
                self.from_checkpoint = True
                self.arch = "poser"
                self.hidden_dim = info.get("hidden_dim")
                self.num_layers = info.get("num_layers")
                self.n_joints = info.get("n_joints")
                self.input_dim = info.get("input_dim")
                self.stem_dim = info.get("stem_dim")
                self.card = _poser_card(self.path)
                # One checkpoint per test recording, so the file name has to be
                # part of the model name.
                stem = self.path.stem
                if stem not in ("best", "best_model"):
                    self.name = f"{self.path.parent.name}__{stem}"

    @property
    def out_joints(self):
        return self.n_joints or 13

    def __repr__(self):
        src = "checkpoint" if self.from_checkpoint else "name"
        return (f"ModelSpec({self.name}: arch={self.arch}, win={self.window}, "
                f"hid={self.hidden_dim}, lay={self.num_layers}, "
                f"joints={self.out_joints}, read from {src})")


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Running the model
# ---------------------------------------------------------------------------
def _import_poser(root=None):
    """Import the poser package from the repository, without copying it."""
    import importlib
    import sys
    cands = []
    if root:
        cands.append(Path(root) / "poser")
    here = Path(__file__).resolve().parent
    for up in (here, *here.parents):
        cands += [up / "poser", up / "src" / "poser"]
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


# ---------------------------------------------------------------------------
# Ground truth and output files
# ---------------------------------------------------------------------------
def load_ground_truth(folder: Path):
    """-> the ground-truth dataframe of a recording, or None."""
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
    """Filename suffix so multiple models coexist in one recording folder.
    None/'default' -> plain predictions.csv; otherwise predictions__<name>.csv."""
    if not model_name or model_name.lower() == "default":
        return ""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in model_name)
    return f"__{safe}"


def write_predictions(folder: Path, frames, times, pred_mat: np.ndarray, model_name=None):
    cols = {"frame": frames, "time": times}
    for j in range(pred_mat.shape[1]):
        for k, ax in enumerate(("x", "y", "z")):
            cols[f"j{j}_{ax}"] = pred_mat[:, j, k]
    path = folder / f"predictions{_suffix(model_name)}.csv"
    pd.DataFrame(cols).to_csv(path, index=False)
    print(f"[OK] wrote {path}  ({pred_mat.shape[0]} frames)")


def write_metrics(folder: Path, metrics: dict, model_name=None):
    path = folder / f"metrics{_suffix(model_name)}.json"
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[OK] wrote {path}  (MPJPE {metrics['mpjpe_mm']:.1f} mm)")


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
# Metrics
# ---------------------------------------------------------------------------
def cull_to_13(mat: np.ndarray) -> np.ndarray:
    """(T,33,3) MediaPipe -> (T,13,3): virtual pelvis plus the 12 culled joints."""
    pelvis = (mat[:, 23, :] + mat[:, 24, :]) / 2.0
    return np.concatenate([pelvis[:, None, :], mat[:, ORIGINAL_MP_INDICES, :]], axis=1)


def _center_hips(mat):
    """Hip-centre for 33 joints; for the culled topology joint 0 is the pelvis."""
    if mat.shape[1] == 13:
        return mat - mat[:, [0], :]
    return mat - np.nanmean(mat[:, [23, 24], :], axis=1, keepdims=True)


def compute_metrics(gt_mat: np.ndarray, pred_mat: np.ndarray) -> dict:
    """-> MPJPE, per-joint error, bone-length error and joint-angle error.

    The dashboard recomputes its own figures from the prediction CSV; this is
    what the CLI reports and what ends up in metrics__<name>.json.
    """
    if pred_mat.shape[1] == 13 and gt_mat.shape[1] == 33:
        gt_mat = cull_to_13(gt_mat)
    nj = pred_mat.shape[1]
    triplets = [(7, 8, 9), (10, 11, 12), (1, 2, 3), (4, 5, 6)]     # shoulder/hip chains

    g, p = _center_hips(gt_mat), _center_hips(pred_mat)
    dist = np.linalg.norm(p - g, axis=2)                            # (T, nj) metres
    per_joint = np.nanmean(dist, axis=0) * 1000.0
    mpjpe = float(np.nanmean(dist) * 1000.0)

    bone = []
    for a, b in CONNECTIONS_13:
        lg = np.linalg.norm(gt_mat[:, a] - gt_mat[:, b], axis=1)
        lp = np.linalg.norm(pred_mat[:, a] - pred_mat[:, b], axis=1)
        bone.append(np.abs(lp - lg))
    bone_err = float(np.nanmean(np.vstack(bone)) * 1000.0)

    def ang(m, a, b, c):
        v1, v2 = m[:, a] - m[:, b], m[:, c] - m[:, b]
        cos = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1)
                                         * np.linalg.norm(v2, axis=1) + 1e-9)
        return np.degrees(np.arccos(np.clip(cos, -1, 1)))
    ang_err = float(np.nanmean([np.abs(ang(pred_mat, *t) - ang(gt_mat, *t))
                                for t in triplets]))

    return {
        "architecture": "POSER",
        "mpjpe_mm": mpjpe,
        "bone_error_mm": bone_err,
        "angle_error_deg": ang_err,
        "pck_at_50mm": float(np.mean(per_joint < 50.0)),
        "n_joints": int(nj),
        "per_joint_mpjpe_mm": {str(i): float(per_joint[i]) for i in range(nj)},
        "source": "model",
    }


# ---------------------------------------------------------------------------
# Finding checkpoints
# ---------------------------------------------------------------------------
_SKIP_DIRS = {"site-packages", "dist-packages", "node_modules", ".git", "__pycache__",
              "venv", ".venv", ".env"}
_SKIP_PTH = {"distutils-precedence.pth", "easy-install.pth", "protobuf-.pth"}
# *.pt files that are data, not models. Matched against the file stem as whole
# words: best_features.pt is a checkpoint, train_cache.pt is not.
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
        print(f"[spec] skipping {p.name}: not a readable checkpoint")
        return False
    if p.suffix.lower() == ".pth":
        if n in _SKIP_PTH or n.endswith("-precedence.pth") or n.endswith("-nspkg.pth"):
            return False
        try:                       # real checkpoints are binary and sizeable
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


# ---------------------------------------------------------------------------
# Entry point, shared by the CLI and the dashboard
# ---------------------------------------------------------------------------
def infer_to_files(folder, checkpoint, model_name=None, gt_df=None, write=True,
                   progress=None, **_ignored):
    """-> (model name, predictions, metrics). Writes the dashboard files unless
    told otherwise."""
    folder = Path(folder)
    if gt_df is None:
        gt_df = load_ground_truth(folder)
    spec = ModelSpec(checkpoint)
    if spec.arch != "poser":
        raise ValueError(
            f"'{Path(checkpoint).name}' is not a checkpoint of this model. "
            f"Expected the weights written by src/poser/train.py.")

    P, times, extra = run_poser_inference(folder, checkpoint, spec, progress=progress)
    name = model_name or spec.name

    metrics = {"architecture": "POSER", "source": "model", "n_joints": 13}
    if gt_df is not None:
        gt = _center_hips(cull_to_13(joints_matrix(gt_df)))
        n = min(len(P), len(gt))
        metrics = compute_metrics(gt[:n], P[:n])
    metrics.update({k: v for k, v in extra.items() if k != "parents"})

    if write:
        write_predictions(folder, np.arange(len(P)), times[:len(P)], P, model_name=name)
        write_metrics(folder, metrics, model_name=name)
        write_topology(folder, name, 13, extra["parents"])
    return name, P, metrics


def main():
    ap = argparse.ArgumentParser(
        description="Run pose inference and write predictions.csv + metrics.json")
    ap.add_argument("--subject", default="video1")
    ap.add_argument("--checkpoint", required=True,
                    help="path to a trained checkpoint, e.g. models/ft_s0/best_video1.pt")
    ap.add_argument("--model-name", default=None,
                    help="label for this model. Writes predictions__<name>.csv and "
                         "metrics__<name>.json so several models can be compared "
                         "in the dashboard.")
    ap.add_argument("--root", default=None, help="project root")
    args = ap.parse_args()

    root = Path(args.root) if args.root else project_root()
    folder = subject_dir(root, args.subject)
    print(f"[INFO] recording folder: {folder}")
    print(f"[INFO] {ModelSpec(args.checkpoint)}")

    name, _, metrics = infer_to_files(folder, args.checkpoint,
                                      model_name=args.model_name,
                                      gt_df=load_ground_truth(folder))
    print(f"[INFO] wrote predictions for '{name}' "
          f"(MPJPE {metrics.get('mpjpe_mm', float('nan')):.1f} mm)")
    print("[DONE]")


if __name__ == "__main__":
    main()
