"""AMASS SMPL-H sequences -> 33-joint MediaPipe pose format.
Stores joints, root translation, segment rotations and the frame rate."""
import argparse
import json
from pathlib import Path

import numpy as np

import synth_imu                      # for the shared zero-phase low pass
from smpl_joints import BodyModel, find_model

SMPL_TO_MP = {1: 23, 2: 24, 4: 25, 5: 26, 7: 27, 8: 28,
              16: 11, 17: 12, 18: 13, 19: 14, 20: 15, 21: 16}
SMPL_HEAD, SMPL_FOOT_L, SMPL_FOOT_R = 15, 10, 11

SENSOR_JOINTS = {"left_wrist": (18, 20), "right_wrist": (19, 21),
                 "left_ankle": (4, 7), "right_ankle": (5, 8)}
SENSOR_ORDER = ["left_wrist", "right_wrist", "left_ankle", "right_ankle"]


def axis_matrix(up):
    if up == 2:
        return np.array([[1., 0, 0], [0, 0, -1.], [0, 1., 0]])
    if up == 1:
        return np.diag([1., -1., -1.])
    return np.array([[0, 1., 0], [-1., 0, 0], [0, 0, 1.]])

def find_body_model(root: Path, quiet: bool = True):
    p, g = find_model(root)
    if not quiet:
        print(f"body model: {p}  ({g})")
    return p, g


_CACHE = {}


def joints_from_amass(npz_path: Path, model_root: Path, max_frames=None, step=1,
                      start_frame=0):
    d = np.load(npz_path, allow_pickle=True)
    if "poses" not in d:
        raise ValueError("contains no 'poses'")
    poses = np.asarray(d["poses"], dtype=np.float64)
    trans = np.asarray(d["trans"], dtype=np.float64) if "trans" in d.files \
        else np.zeros((len(poses), 3))
    betas = np.asarray(d["betas"], dtype=np.float64) if "betas" in d.files else None
    fps = float(d["mocap_framerate"]) if "mocap_framerate" in d.files else \
        float(d["mocap_frame_rate"]) if "mocap_frame_rate" in d.files else 120.0

    if start_frame:
        poses, trans = poses[start_frame:], trans[start_frame:]
    if step > 1:
        poses, trans, fps = poses[::step], trans[::step], fps / step
    if max_frames and len(poses) > max_frames:
        poses, trans = poses[:max_frames], trans[:max_frames]
    if len(poses) < 30:
        raise ValueError("zu kurz")

    key = str(model_root)
    if key not in _CACHE:
        path, _ = find_model(model_root)
        _CACHE[key] = BodyModel(path)
    bm = _CACHE[key]
    J, Rg = bm.joints(poses, trans, betas, return_rot=True)
    C = bm.segment_align({SENSOR_JOINTS[s][0]: SENSOR_JOINTS[s][1]
                          for s in SENSOR_ORDER}, betas)
    seg = np.stack([Rg[:, SENSOR_JOINTS[s][0]] @ C[SENSOR_JOINTS[s][0]]
                    for s in SENSOR_ORDER], axis=1)          # (T, 4, 3, 3)
    return J, seg, fps


# ---------------------------------------------------------------- conversion
def detect_up_axis(J):
    head = J[:, SMPL_HEAD].mean(axis=0)
    foot = J[:, [SMPL_FOOT_L, SMPL_FOOT_R]].mean(axis=(0, 1))
    return int(np.argmax(np.abs(head - foot)))


def to_mediapipe_frame(J, up):
    """-> MediaPipe world frame: x right, y down, z back."""
    X, Y, Z = J[..., 0], J[..., 1], J[..., 2]
    if up == 2:
        return np.stack([X, -Z, Y], axis=-1)       # AMASS default: z up
    if up == 1:
        return np.stack([X, -Y, -Z], axis=-1)
    return np.stack([Y, -X, Z], axis=-1)


def to_mp_layout(J):
    T = len(J)
    P = np.zeros((T, 33, 3))
    for s, m in SMPL_TO_MP.items():
        P[:, m] = J[:, s]
    head = J[:, SMPL_HEAD]
    for m in range(0, 11):
        P[:, m] = head
    for m in (17, 19, 21):
        P[:, m] = J[:, 20]
    for m in (18, 20, 22):
        P[:, m] = J[:, 21]
    P[:, 29], P[:, 30] = J[:, 7], J[:, 8]
    P[:, 31], P[:, 32] = J[:, SMPL_FOOT_L], J[:, SMPL_FOOT_R]
    return P


def _band_limit(x2d, fps_in, fps_out, out_grid):
    """Band-limit x2d (T,C) to the lower of the two Nyquist rates.

    Linear interpolation is not band limited. Downsampling without a filter
    aliases; upsampling leaves an image of the source rate at every multiple of
    fps_in, which the double differentiation in synth_imu lifts by omega^2 into
    the impact band. Filter on whichever grid is the finer one.
    """
    cut = 0.45 * min(fps_in, fps_out)
    fs = fps_out if out_grid else fps_in
    if cut >= 0.5 * fs:
        return x2d
    return synth_imu.lowpass(x2d, fs, cut)


def resample(P, fps_in, fps_out):
    """-> (poses on the fps_out grid, the rate actually used)."""
    if not fps_out or abs(fps_in - fps_out) < 1e-6:
        return P, fps_in
    T, J, _ = P.shape
    flat = P.reshape(T, -1)
    if fps_out < fps_in:                       # decimation: filter first
        flat = _band_limit(flat, fps_in, fps_out, out_grid=False)
    t_in = np.arange(T) / fps_in
    t_out = np.arange(0, t_in[-1], 1.0 / fps_out)
    out = np.stack([np.interp(t_out, t_in, flat[:, c])
                    for c in range(flat.shape[1])], axis=1)
    if fps_out > fps_in:                       # upsampling: remove the images
        out = _band_limit(out, fps_in, fps_out, out_grid=True)
    return out.reshape(len(t_out), J, 3), fps_out


def resample_rot(R, fps_in, fps_out):
    if not fps_out or abs(fps_in - fps_out) < 1e-6:
        return R
    flat = R.reshape(len(R), -1)
    if fps_out < fps_in:
        flat = _band_limit(flat, fps_in, fps_out, out_grid=False)
    t_in = np.arange(len(R)) / fps_in
    t_out = np.arange(0, t_in[-1], 1.0 / fps_out)
    out = np.stack([np.interp(t_out, t_in, flat[:, c])
                    for c in range(flat.shape[1])], axis=1)
    if fps_out > fps_in:
        out = _band_limit(out, fps_in, fps_out, out_grid=True)
    M = out.reshape(len(t_out), *R.shape[1:])
    U, _, Vt = np.linalg.svd(M)                # back onto SO(3)
    Q = U @ Vt
    d = np.linalg.det(Q)
    Q[d < 0] = (U[d < 0] * np.array([1.0, 1.0, -1.0])) @ Vt[d < 0]
    return Q


def convert(J_smpl, fps, fps_out, target_scale, bone_targets=None, seg=None):
    up = detect_up_axis(J_smpl)
    P = to_mp_layout(to_mediapipe_frame(J_smpl, up))
    root = (P[:, 23] + P[:, 24]) / 2.0                       # true pelvis trajectory
    P = P - root[:, None, :]                                 # pelvis-centred
    P, fps_new = resample(P, fps, fps_out)
    root, _ = resample(root[:, None, :], fps, fps_out)
    root = root[:, 0, :]
    if seg is not None:
        seg = axis_matrix(up)[None, None] @ seg
        seg = resample_rot(seg, fps, fps_out)[:len(P)]
    root = root - root[0]                                    # start at the origin
    scale = float(np.linalg.norm(P[:, 11] - P[:, 23], axis=1).mean())

    if bone_targets:
        import retarget
        before = retarget.measure(P)
        P = retarget.apply(P, bone_targets)
        f = bone_targets["pelvis_shoulder"] / max(before["pelvis_shoulder"], 1e-6)
        root = root * f
        return P, root, seg, fps_new, up, scale, before
    if target_scale:
        f = target_scale / max(scale, 1e-6)
        P *= f
        root *= f
    return P, root, seg, fps_new, up, scale, None


def main():
    ap = argparse.ArgumentParser(description="AMASS -> pose sequences")
    ap.add_argument("--amass", required=True, help="AMASS .npz file or directory")
    ap.add_argument("--body-model", required=True, help="directory containing smplh/ or smpl/")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--glob", default="*.npz")
    ap.add_argument("--fps", type=float, default=120.0,
                    help="output frame rate; AMASS is natively 120 Hz")
    ap.add_argument("--target-scale", type=float, default=0.48,
                    help="mean shoulder-to-hip distance in m, taken from the real recordings")
    ap.add_argument("--limit", type=int, default=0, help="only the first N sequences")
    ap.add_argument("--sample", type=int, default=0,
                    help="pick N sequences spread evenly over all subjects instead of "
                         "the first N, which keeps the selection diverse.")
    ap.add_argument("--selection", default=None,
                    help="manifest from build_amass_manifest.py; overrides --sample/--limit")
    ap.add_argument("--overwrite", action="store_true",
                    help="overwrite existing pose files")
    ap.add_argument("--max-seconds", type=float, default=60.0)
    ap.add_argument("--min-seconds", type=float, default=4.0)
    ap.add_argument("--bone-target", default=None,
                    help="a real recording folder; its bone lengths are adopted so the "
                         "skeleton matches the real data.")
    args = ap.parse_args()

    bone_targets = None
    if args.bone_target:
        import retarget
        bone_targets = retarget.targets_from_recording(args.bone_target)
        print(f"Zielmasse aus {bone_targets.pop('_source')} "
              f"(links/rechts gemittelt):")
        for k, v in bone_targets.items():
            print(f"    {k:18s} {v*100:5.1f} cm")
        print()

    src = Path(args.amass).resolve()
    files = sorted(p for p in (src.rglob(args.glob) if src.is_dir() else [src])
                   if p.is_file() and "shape" not in p.name.lower()
                   and p.name != "neutral_stagei.npz")
    if not files:
        raise SystemExit(f"No .npz files under {src}")
    selected = None
    if args.selection:
        selected = json.loads(Path(args.selection).read_text(encoding="utf-8")).get("selected", [])
        by_source = {str(p.resolve()): p for p in files}
        missing = [r["source"] for r in selected if str(Path(r["source"]).resolve()) not in by_source]
        if missing:
            raise SystemExit(f"{len(missing)} manifest entries are not under --amass, e.g. {missing[0]}")
        files = [by_source[str(Path(r["source"]).resolve())] for r in selected]
    if not selected and args.sample and args.sample < len(files):
        idx = np.unique(np.linspace(0, len(files) - 1, args.sample).astype(int))
        files = [files[i] for i in idx]
    if args.limit:
        files = files[:args.limit]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    find_body_model(Path(args.body_model), quiet=False)
    print(f"{len(files)} sequence(s) found\n")

    ok, skipped = 0, []
    for i, f in enumerate(files):
        try:
            entry = selected[i] if selected else {}
            name = entry.get("output_stem") or f"{f.parent.name}_{f.stem}".replace(" ", "_")
            dest = out_dir / f"{name}.npz"
            if dest.exists() and not args.overwrite:
                print(f"  {name[:40]:42s} already present, skipped")
                continue
            crop_start = int(round(float(entry.get("suggested_crop_start_s", 0.0)) *
                                   float(entry.get("fps", 120.0))))
            J, seg, fps = joints_from_amass(f, Path(args.body_model),
                                            max_frames=int(args.max_seconds * 120),
                                            start_frame=crop_start)
            if len(J) / fps < args.min_seconds:
                skipped.append((f.name, "zu kurz"))
                continue
            # fps_actual is what convert() really used: args.fps, or the source
            # rate when args.fps is 0 or already equal. It is what gets stored.
            P, root, seg, fps_actual, up, scale, before = convert(
                J, fps, args.fps, args.target_scale, bone_targets, seg)
            np.savez_compressed(dest, joints=P, root=root,
                                seg_rot=seg.astype(np.float32), fps=fps_actual,
                                source=str(f), dataset=entry.get("dataset", ""),
                                subject=entry.get("subject", ""),
                                motion_profile=entry.get("motion_profile", ""),
                                crop_start_s=float(entry.get("suggested_crop_start_s", 0.0)))
            travel = float(np.linalg.norm(root[-1] - root[0]))
            if ok == 0 and before is not None:
                import retarget
                print("  Skelett angepasst:")
                print(retarget.report(before, retarget.measure(P), bone_targets))
                print()
            print(f"  {name[:40]:42s} {P.shape[0]:5d} Frames @ {fps_out:.0f} Hz "
                  f"| Achse {'XYZ'[up]} | Weg {travel:5.1f} m")
            ok += 1
        except Exception as e:
            skipped.append((f.name, str(e)[:70]))

    print(f"\n{ok} converted, {len(skipped)} skipped")
    for n, why in skipped[:8]:
        print(f"  - {n}: {why}")
    (out_dir / "conversion_info.json").write_text(json.dumps(
        {"converted": ok, "skipped": len(skipped), "fps": args.fps,
         "target_scale": args.target_scale, "selection": args.selection,
         "overwrite": bool(args.overwrite)}, indent=2), encoding="utf-8")
    if ok == 0:
        raise SystemExit("Not a single sequence could be converted.")


if __name__ == "__main__":
    main()
