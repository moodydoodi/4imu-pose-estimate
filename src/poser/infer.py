"""Apply a checkpoint to a recording and write dashboard CSV.
Architecture and bone lengths are read from the checkpoint.

    python infer.py --checkpoint models/finetune/best_video1.pt \
        --cache cache/real_body --name poser --out models/finetune
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import skeleton as SK
from config import JOINT_NAMES, N_JOINTS, PARENTS
from model import Poser


# ------------------------------------------------------------------ load
def describe(sd):
    """Read the architecture from the state dict."""
    if "inorm.weight" not in sd or "rnn1.weight_hh_l0" not in sd:
        raise ValueError(
            "This checkpoint was not produced by model.py. Keys found "
            f"(excerpt): {list(sd)[:6]}")
    n_feat = int(sd["inorm.weight"].shape[0])
    stem = int(sd["stem.net.0.weight"].shape[0])
    hidden = int(sd["rnn1.weight_hh_l0"].shape[1])
    layers = sum(1 for k in sd if k.startswith("rnn1.weight_ih_l")
                 and not k.endswith("_reverse"))
    L = sd["L"].detach().cpu().numpy() if "L" in sd else None
    return dict(n_feat=n_feat, stem=stem, hidden=hidden, layers=layers, L=L)


def load_poser(path, device="cpu"):
    """Checkpoint -> ready model. The architecture is detected automatically."""
    sd = torch.load(str(path), map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    info = describe(sd)
    if info["L"] is None:
        raise ValueError("The checkpoint has no bone lengths ('L').")
    model = Poser(info["L"], hidden=info["hidden"], layers=info["layers"],
                  stem=info["stem"], n_feat=info["n_feat"])
    model.load_state_dict(sd)
    model.to(device).eval()
    return model, info


# ----------------------------------------------------------------- apply
@torch.no_grad()
def predict(model, X, device="cpu", win=200, hop=100):
    """(T,n_feat) -> (T,13,3). Sliding windows, triangular weighting."""
    T = len(X)
    acc = np.zeros((T, N_JOINTS, 3))
    wsum = np.zeros((T, 1, 1))
    w = np.minimum(np.arange(win), np.arange(win)[::-1]).astype(float) + 1.0
    w /= w.max()
    starts = list(range(0, max(T - win + 1, 1), hop))
    if starts and starts[-1] + win < T:
        starts.append(max(T - win, 0))
    for i in range(0, len(starts), 32):
        ch = starts[i:i + 32]
        xb = torch.as_tensor(np.stack([X[s:s + win] for s in ch]),
                             dtype=torch.float32, device=device)
        pb = model(xb)["pos"].float().cpu().numpy()
        for k, s in enumerate(ch):
            n = min(win, T - s)
            acc[s:s + n] += pb[k, :n] * w[:n, None, None]
            wsum[s:s + n] += w[:n, None, None]
    return acc / np.maximum(wsum, 1e-9)


def procrustes(P, Y):
    Pc, Yc = P - P.mean(1, keepdims=True), Y - Y.mean(1, keepdims=True)
    H = np.einsum("tji,tjk->tik", Pc, Yc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.einsum("tij,tjk->tik",
                                        Vt.transpose(0, 2, 1), U.transpose(0, 2, 1))))
    E = np.zeros_like(H); E[:, 0, 0] = E[:, 1, 1] = 1.0; E[:, 2, 2] = d
    R = np.einsum("tij,tjk,tkl->til", Vt.transpose(0, 2, 1), E, U.transpose(0, 2, 1))
    s = (S[:, :2].sum(1) + d * S[:, 2]) / np.maximum((Pc ** 2).sum((1, 2)), 1e-12)
    return s[:, None, None] * np.einsum("tij,tkj->tki", R, Pc) + Y.mean(1, keepdims=True)


def write_outputs(out_dir, name, t, P, Y=None):
    """Write predictions__<name>.csv, its topology sidecar and metrics__<name>.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = ["t"] + [f"j{j}_{a}" for j in range(N_JOINTS) for a in "xyz"]
    rows = np.concatenate([np.asarray(t)[:, None], P.reshape(len(P), -1)], axis=1)
    csv = out_dir / f"predictions__{name}.csv"
    with csv.open("w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(f"{v:.6g}" for v in r) + "\n")

    (out_dir / f"predictions__{name}.topology.json").write_text(json.dumps({
        "n_joints": N_JOINTS, "parents": PARENTS, "joint_names": JOINT_NAMES,
        "root": 0, "units": "m", "frame": "pelvis-centred",
    }, indent=2))

    met = {"architecture": "poser (Conv+BiLSTM, 12 bone directions)",
           "n_joints": N_JOINTS}
    if Y is not None:
        e = np.linalg.norm(P - Y, axis=2)
        pa = np.linalg.norm(procrustes(P, Y) - Y, axis=2)
        mean_pose = np.repeat(Y.mean(0)[None], len(Y), 0)
        met.update({
            "mpjpe_mm": float(e.mean() * 1000),
            "pa_mpjpe_mm": float(pa.mean() * 1000),
            "pck_at_50mm": float((e < 0.05).mean()),
            "pck_at_100mm": float((e < 0.10).mean()),
            "baseline_mean_pose_mm": float(np.linalg.norm(mean_pose - Y, axis=2).mean() * 1000),
            "per_joint_mpjpe_mm": {str(i): float(v) for i, v in
                                   enumerate(e.mean(0) * 1000)},
        })
    (out_dir / f"metrics__{name}.json").write_text(json.dumps(met, indent=2))
    return csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache", default=None, help="directory with the .npz files from prepare.py")
    ap.add_argument("--only", nargs="*", default=[], help="only these recordings")
    ap.add_argument("--name", default="poser")
    ap.add_argument("--out", default=None,
                    help="output directory; defaults to next to each .npz")
    ap.add_argument("--per-recording-dir", action="store_true",
                    help="write into a per-recording subdirectory, as the dashboard expects")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fps", type=float, default=50.0)
    a = ap.parse_args()

    model, info = load_poser(a.checkpoint, a.device)
    print(f"loaded: {info['n_feat']} input channels, stem {info['stem']}, "
          f"{info['layers']} LSTM layers of {info['hidden']}, "
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f} M parameters")
    print(f"bone lengths (mm): {np.round(info['L']*1000).astype(int)}")

    files = sorted(Path(a.cache).glob("*.npz"))
    if a.only:
        files = [f for f in files if f.stem in a.only]
    if not files:
        raise SystemExit(f"Nothing found under {a.cache}.")

    for p in files:
        z = np.load(p, allow_pickle=True)
        X, Y = z["X"].astype(np.float32), z["Y"].astype(float)
        t = z["t"].astype(float) if "t" in z else np.arange(len(X)) / a.fps
        P = predict(model, X, a.device)
        base = Path(a.out) if a.out else p.parent
        dst = base / str(z["name"]) if a.per_recording_dir else base
        csv = write_outputs(dst, a.name, t, P, Y)
        e = np.linalg.norm(P - Y, axis=2).mean() * 1000
        print(f"  {str(z['name']):16s} {len(X):6d} frames   MPJPE {e:6.1f} mm   -> {csv}")


if __name__ == "__main__":
    main()
