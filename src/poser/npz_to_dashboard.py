"""Convert stored predictions into dashboard input. Does not require torch.

train.py writes one pred_<video>.npz per test video containing both the
predicted and the true pose frame by frame. The dashboard needs nothing beyond a
predictions__<name>.csv next to the *_gt_3d.csv, so this writes those directly,
for any number of models side by side.

Body-frame models predict the pose without heading, which four limb sensors
without a magnetometer cannot observe. For display next to the video the heading
is taken from the ground truth. That changes no error figure, since both sides
are rotated by the same matrix, but it is recorded as "heading_from_gt".

    python npz_to_dashboard.py --models models/finetune_s0 --cache cache/real_body \
        --data data/processed
"""
import argparse
import json
from pathlib import Path

import numpy as np

import skeleton as SK
from config import JOINT_NAMES, N_JOINTS, PARENTS
import dataio


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


def write(folder, name, t, P, Y, extra):
    folder = Path(folder)
    cols = ["t"] + [f"j{j}_{a}" for j in range(N_JOINTS) for a in "xyz"]
    rows = np.concatenate([np.asarray(t)[:, None], P.reshape(len(P), -1)], axis=1)
    with (folder / f"predictions__{name}.csv").open("w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(f"{v:.6g}" for v in r) + "\n")
    (folder / f"predictions__{name}.topology.json").write_text(json.dumps(
        {"n_joints": N_JOINTS, "parents": list(PARENTS), "names": JOINT_NAMES}))
    e = np.linalg.norm(P - Y, axis=2)
    pa = np.linalg.norm(procrustes(P, Y) - Y, axis=2)
    met = {"architecture": "POSER", "source": "model", "n_joints": N_JOINTS,
           "mpjpe_mm": float(e.mean() * 1000),
           "pa_mpjpe_mm": float(pa.mean() * 1000),
           "pck_at_50mm": float((e < 0.05).mean()),
           "pck_at_100mm": float((e < 0.10).mean()),
           "baseline_mean_pose_mm": float(
               np.linalg.norm(np.repeat(Y.mean(0)[None], len(Y), 0) - Y, axis=2).mean() * 1000),
           "per_joint_mpjpe_mm": {str(i): float(v) for i, v in enumerate(e.mean(0) * 1000)}}
    met.update(extra)
    (folder / f"metrics__{name}.json").write_text(json.dumps(met, indent=2))
    return met


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="directories containing pred_*.npz from train.py")
    ap.add_argument("--cache", default="cache/real_body",
                    help="the cache used for testing; provides the time axis")
    ap.add_argument("--data", default="data/processed",
                    help="directory holding the recording subfolders; output goes there")
    ap.add_argument("--fps", type=float, default=50.0)
    a = ap.parse_args()

    tcache = {}
    for p in sorted(Path(a.cache).glob("*.npz")):
        z = np.load(p, allow_pickle=True)
        tcache[str(z["name"])] = z["t"].astype(float) if "t" in z else None

    for mdir in a.models:
        mdir = Path(mdir)
        card = {}
        cp = mdir / "model_card.json"
        if cp.exists():
            card = json.loads(cp.read_text())
        frame = card.get("frame", "body")
        label = mdir.name
        preds = sorted(mdir.glob("pred_*.npz"))
        if not preds:
            print(f"[skipped] {mdir}: no pred_*.npz")
            continue
        print(f"\n{label}  (reference frame {frame})")
        for pf in preds:
            z = np.load(pf, allow_pickle=True)
            vid = str(z["name"])
            P = z["P"].astype(float)      # (T,13,3)
            Y = z["Y"].astype(float)
            dst = Path(a.data) / vid
            if not dst.is_dir():
                print(f"  {vid}: target directory missing ({dst})")
                continue
            t = tcache.get(vid)
            if t is None or len(t) < len(P):
                t = np.arange(len(P)) / a.fps
            t = t[:len(P)]

            extra = {"frame_trained": frame, "heading_from_gt": False}
            if frame == "body":
                # Take the world pose from the recording folder to get the rotation
                tg, W = dataio.read_pose(dst)
                if W is None:
                    print(f"  {vid}: no *_gt_3d.csv, cannot recover the heading")
                    continue
                Ww = np.stack([np.interp(t, tg, W[:, j, k])
                               for j in range(13) for k in range(3)], axis=1).reshape(-1, 13, 3)
                _, R = SK.body_frame(Ww - Ww[:, 0:1])
                P = np.einsum("tij,tkj->tki", R, P)
                Y = np.einsum("tij,tkj->tki", R, Y)
                extra["heading_from_gt"] = True

            met = write(dst, label, t, P, Y, extra)
            print(f"  {vid:9s} {len(P):6d} frames   MPJPE {met['mpjpe_mm']:6.1f} mm   "
                  f"PA {met['pa_mpjpe_mm']:5.1f}   mean pose {met['baseline_mean_pose_mm']:6.1f}")
    print("\nDone. Reload the recording in the dashboard; the models then appear "
          "under Evaluation.")


if __name__ == "__main__":
    main()
