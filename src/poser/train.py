"""Training and leave-one-recording-out evaluation.
One recording tests, one validates, the rest train. --loro repeats this with
every recording as the test case. Writes weights, model_card.json, predictions
and metrics.json to --out.

    python train.py --dry-run
    python train.py --cache cache/real_body --loro
    python train.py --cache cache/synth_body --epochs 1 --out models/pretrain
    python train.py --cache cache/real_body --init models/pretrain/best.pt --lr 2e-4 --loro
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import augment
import skeleton as SK
from config import FPS, HOP, N_FEAT, N_JOINTS, WIN
from model import PoseLoss, Poser, fk


# ------------------------------------------------------------------- data
class Cache:
    def __init__(self, folder=None, names=None, recs=None):
        self.meta = {}
        if recs is not None:
            self.recs = recs
        else:
            self.recs = []
            self.meta = {}
            for p in sorted(Path(folder).glob("*.npz")):
                z = np.load(p, allow_pickle=True)
                self.recs.append({"name": str(z["name"]), "X": z["X"],
                                  "Y": z["Y"], "D": z["D"]})
                for k in ("frame", "suffix", "fps"):
                    if k in z:
                        self.meta[k] = z[k].item() if z[k].shape == () else z[k]
        if names is not None:
            self.recs = [r for r in self.recs if r["name"] in names]
        if not self.recs:
            raise SystemExit(f"Nothing usable in cache {folder} (looking for {names}).")

    def limit(self, n, rng):
        """Keep N randomly drawn recordings."""
        if not n or n >= len(self.recs):
            return self
        idx = rng.choice(len(self.recs), n, replace=False)
        c = Cache(recs=[self.recs[i] for i in sorted(idx)])
        c.meta = dict(self.meta)          # else model_card.json reports the wrong frame
        return c

    def scramble(self):
        """Pair each recording's features with another one's poses (control)."""
        n = len(self.recs)
        if n < 2:
            raise SystemExit("--scramble-targets needs at least two recordings.")
        out = []
        for i, r in enumerate(self.recs):
            src = self.recs[(i + 1) % n]
            T = len(r["X"])
            reps = int(np.ceil(T / len(src["Y"])))
            Y = np.tile(src["Y"], (reps, 1, 1))[:T]
            D = np.tile(src["D"], (reps, 1, 1))[:T]
            out.append({"name": r["name"], "X": r["X"], "Y": Y, "D": D})
        c = Cache(recs=out)
        c.meta = dict(self.meta)
        return c

    def subset(self, names):
        """Subset without re-reading the files."""
        c = Cache(recs=[r for r in self.recs if r["name"] in names], names=None)
        c.meta = dict(self.meta)
        return c

    @property
    def names(self):
        return [r["name"] for r in self.recs]

    def index(self, win=WIN, hop=HOP):
        return [(i, s) for i, r in enumerate(self.recs)
                for s in range(0, max(len(r["X"]) - win + 1, 0), hop)]

    def batch(self, idx, win=WIN, device="cpu"):
        X = np.stack([self.recs[i]["X"][s:s+win] for i, s in idx])
        Y = np.stack([self.recs[i]["Y"][s:s+win] for i, s in idx])
        D = np.stack([self.recs[i]["D"][s:s+win] for i, s in idx])
        f = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
        return f(X), f(Y), f(D)


# ---------------------------------------------------------------- metrics
def procrustes_align(P, Y):
    """Per-frame Procrustes alignment of P onto Y (rotation and scale)."""
    Pc, Yc = P - P.mean(1, keepdims=True), Y - Y.mean(1, keepdims=True)
    H = np.einsum("tji,tjk->tik", Pc, Yc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.einsum("tij,tjk->tik", Vt.transpose(0, 2, 1), U.transpose(0, 2, 1))))
    E = np.zeros_like(H); E[:, 0, 0] = E[:, 1, 1] = 1.0; E[:, 2, 2] = d
    R = np.einsum("tij,tjk,tkl->til", Vt.transpose(0, 2, 1), E, U.transpose(0, 2, 1))
    s = (S[:, :2].sum(1) + d * S[:, 2]) / np.maximum((Pc ** 2).sum((1, 2)), 1e-12)
    return s[:, None, None] * np.einsum("tij,tkj->tki", R, Pc) + Y.mean(1, keepdims=True)


def metrics(P, Y):
    e = np.linalg.norm(P - Y, axis=2)
    pa = np.linalg.norm(procrustes_align(P, Y) - Y, axis=2)
    return {"mpjpe": float(e.mean() * 1000), "pa_mpjpe": float(pa.mean() * 1000),
            "pck50": float((e < 0.05).mean() * 100),
            "pck100": float((e < 0.10).mean() * 100),
            "per_joint": (e.mean(0) * 1000).tolist()}


@torch.no_grad()
def predict_sequence(model, X, device, win=WIN, hop=None):
    """(T,N_FEAT) -> (T,13,3), overlapping windows with triangular blending."""
    hop = hop or win // 2
    T = len(X)
    acc = np.zeros((T, N_JOINTS, 3)); wsum = np.zeros((T, 1, 1))
    w = np.minimum(np.arange(win), np.arange(win)[::-1]).astype(float) + 1.0
    w /= w.max()
    starts = list(range(0, max(T - win + 1, 1), hop))
    if starts[-1] + win < T:
        starts.append(max(T - win, 0))
    for i in range(0, len(starts), 32):
        chunk = starts[i:i+32]
        xb = torch.as_tensor(np.stack([X[s:s+win] for s in chunk]),
                             dtype=torch.float32, device=device)
        pb = model(xb)["pos"].float().cpu().numpy()
        for k, s in enumerate(chunk):
            n = min(win, T - s)
            acc[s:s+n] += pb[k, :n] * w[:n, None, None]
            wsum[s:s+n] += w[:n, None, None]
    return acc / np.maximum(wsum, 1e-9)


# --------------------------------------------------------------- training
def run_one(cache, test, val, args, canon, device, evalcache=None):
    if evalcache is None:
        train_names = [n for n in cache.names if n not in (test, val)]
        tr = cache.subset(set(train_names))
        va = cache.subset({val}) if val else None
        te = cache.subset({test})
    else:
        # Test and validation from a separate pool: train on synthetic, test on real.
        train_names = list(cache.names)
        tr = cache
        va = evalcache.subset({val}) if val else None
        te = evalcache.subset({test})

    torch.manual_seed(args.seed)          # every fold starts from the same state
    torch.cuda.manual_seed_all(args.seed)
    model = Poser(canon, hidden=args.hidden, layers=args.layers,
                  dropout=args.dropout).to(device)
    if args.init:
        sd = torch.load(args.init, map_location=device)
        model.load_state_dict(sd)
        print(f"  initialised from {args.init}")
    crit = PoseLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    idx = tr.index(hop=args.hop)
    steps = max(1, len(idx) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * steps, pct_start=0.25)
    rng = np.random.default_rng(args.seed)

    best, best_state, bad = 1e9, None, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(idx)
        tot, t0 = 0.0, time.time()
        for b in range(steps):
            xb, yb, db = tr.batch(idx[b*args.batch:(b+1)*args.batch], device=device)
            xb = augment.apply(xb, args.suffix, args.aug)
            out = model(xb)
            loss, parts = crit(out, yb, db)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item()

        model.eval()
        if va is not None:
            r = va.recs[0]
            vm = metrics(predict_sequence(model, r["X"], device), r["Y"])["mpjpe"]
        else:
            vm = tot / steps
        flag = ""
        if vm < best - 0.1:
            best, best_state, bad = vm, {k: v.detach().cpu().clone()
                                         for k, v in model.state_dict().items()}, 0
            flag = "  *"
        else:
            bad += 1
        print(f"  epoch {ep:3d}  loss {tot/steps:.4f}  "
              f"val {vm:6.1f} mm  {time.time()-t0:5.1f}s{flag}", flush=True)
        if bad >= args.patience:
            print(f"  early stop after {ep} epochs")
            break

    if best_state:
        model.load_state_dict(best_state)
    r = te.recs[0]
    P = predict_sequence(model, r["X"], device)
    m = metrics(P, r["Y"])
    m["baseline_mean_pose"] = SK.mpjpe(
        np.repeat(np.concatenate([q["Y"] for q in tr.recs]).mean(0)[None], len(r["Y"]), 0),
        r["Y"])
    m["val_best"] = best
    m["n_train_windows"] = len(idx)
    m["train_recordings"] = train_names
    return model, m, P, r["Y"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/real")
    ap.add_argument("--skeleton", default="config/skeleton.json")
    ap.add_argument("--suffix", default=None,
                    help="which augmentation to use; taken from the cache when omitted")
    ap.add_argument("--test", default=None)
    ap.add_argument("--val", default=None)
    ap.add_argument("--loro", action="store_true", help="use every recording once as the test case")
    ap.add_argument("--eval-cache", default=None,
                    help="take test and validation recordings from a different cache, "
                         "which measures pure transfer (trained on synthetic, "
                         "tested on real)")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="drop recordings entirely, e.g. the second video of the same "
                         "subject so the test really is an unseen person")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hop", type=int, default=HOP)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--aug", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--max-recordings", type=int, default=0,
                    help="use only N randomly drawn recordings, which separates the "
                         "amount of pre-training from its content")
    ap.add_argument("--scramble-targets", action="store_true",
                    help="control condition: every recording gets the target poses of a "
                         "DIFFERENT one. If that yields the same gain, the effect "
                         "was not transfer but a better weight initialisation")
    ap.add_argument("--seed", type=int, default=0,
                    help="controls initialisation, shuffling and augmentation. Two runs "
                         "with the same seed are comparable, two with different "
                         "seeds show the spread of the method")
    ap.add_argument("--deterministic", action="store_true",
                    help="make cuDNN reproducible, at some cost in speed")
    ap.add_argument("--init", default=None)
    ap.add_argument("--out", default="models/run")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        return dry_run(args)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    canon, _ = SK.load_skeleton(args.skeleton)
    cache = Cache(args.cache)
    evalcache = Cache(args.eval_cache) if args.eval_cache else None
    if args.exclude:
        cache = cache.subset({n for n in cache.names if n not in args.exclude})
        print(f"excluded: {', '.join(args.exclude)}")
    if args.max_recordings:
        cache = cache.limit(args.max_recordings, np.random.default_rng(args.seed))
        print(f"limited to {len(cache.recs)} randomly drawn recordings (seed {args.seed})")
    if args.scramble_targets:
        cache = cache.scramble()
        print("NOTE: target poses scrambled - control condition, not real training")
    # The augmentation depends on the reference frame of the features, which the
    # cache knows. A forgotten --suffix used to silently train with the wrong one.
    cached_suffix = cache.meta.get("suffix")
    if args.suffix is None:
        args.suffix = cached_suffix or "_segment"
        if cached_suffix:
            print(f"suffix {args.suffix} (from the cache)")
    elif cached_suffix and args.suffix != cached_suffix:
        raise SystemExit(
            f"--suffix {args.suffix} but the cache was built from {cached_suffix} "
            f"files.\nThe augmentation would not match the data. Drop --suffix to "
            f"use the cached value.")

    names = cache.names
    n_par = sum(p.numel() for p in
                Poser(canon, hidden=args.hidden, layers=args.layers).parameters())
    print(f"{len(names)} recordings: {', '.join(names)}")
    print(f"seed {args.seed}"
          + ("  (cuDNN deterministic)" if args.deterministic else ""))
    print(f"device {device}, {len(cache.index(hop=args.hop))} windows, "
          f"{n_par/1e6:.2f} M parameters")
    print(f"parameters per window {n_par/max(len(cache.index(hop=args.hop)),1):.0f}:1"
          f"   (far above 100:1 means the model can memorise)\n")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    pool = evalcache.names if evalcache is not None else names
    if evalcache is not None:
        print(f"evaluating on {args.eval_cache}: {', '.join(pool)}")
        print(f"training on all {len(names)} recordings from {args.cache}\n")
    combos = ([(n, pool[(pool.index(n)+1) % len(pool)]) for n in pool] if args.loro
              else [(args.test or pool[-1], args.val or pool[0])])
    if not args.loro and combos[0][0] == combos[0][1]:
        combos = [(combos[0][0], None)]

    allm = []
    for test, val in combos:
        print(f"--- test {test}, validation {val}")
        model, m, P, Ytrue = run_one(cache, test, val, args, canon, device, evalcache)
        torch.save(model.state_dict(), Path(args.out) / f"best_{test}.pt")
        # Under --loro every fold has its own weights; a single "best.pt" would
        # just be whichever fold ran last. Only write it for a single run.
        if not args.loro:
            torch.save(model.state_dict(), Path(args.out) / "best.pt")
        # Model card: suffix, reference frame and rate the weights belong to.
        (Path(args.out) / "model_card.json").write_text(json.dumps({
            "kind": "poser",
            "suffix": cache.meta.get("suffix", args.suffix),
            "frame": cache.meta.get("frame", "world"),
            "fps": float(cache.meta.get("fps", FPS)),
            "window": WIN, "hidden": args.hidden, "layers": args.layers,
            "n_feat": N_FEAT, "seed": args.seed, "trained_on": args.cache,
            "init_from": args.init,
        }, indent=2))
        # Predictions, enough to redo the evaluation without retraining.
        np.savez_compressed(Path(args.out) / f"pred_{test}.npz",
                            P=P.astype(np.float16), Y=Ytrue.astype(np.float16),
                            name=test)
        print(f"  RESULT {test}: MPJPE {m['mpjpe']:.1f} mm, "
              f"PA-MPJPE {m['pa_mpjpe']:.1f}, PCK@100 {m['pck100']:.0f} %, "
              f"mean-pose baseline {m['baseline_mean_pose']:.1f}\n", flush=True)
        m["test"] = test
        allm.append(m)

    print("=" * 66)
    print(f"{'Test':12s}{'MPJPE':>9s}{'PA':>8s}{'PCK@50':>9s}{'PCK@100':>9s}{'mean pose':>12s}")
    for m in allm:
        print(f"{m['test']:12s}{m['mpjpe']:9.1f}{m['pa_mpjpe']:8.1f}"
              f"{m['pck50']:8.0f}%{m['pck100']:8.0f}%{m['baseline_mean_pose']:12.1f}")
    if len(allm) > 1:
        k = lambda f: float(np.mean([m[f] for m in allm]))
        print(f"{'mean':12s}{k('mpjpe'):9.1f}{k('pa_mpjpe'):8.1f}"
              f"{k('pck50'):8.0f}%{k('pck100'):8.0f}%{k('baseline_mean_pose'):12.1f}")
    pj = np.mean([m["per_joint"] for m in allm], axis=0)
    print("\nper-joint error (mm):")
    for i, n in enumerate(SK.JOINT_NAMES):
        print(f"  {n:14s}{pj[i]:7.1f}")
    (Path(args.out) / "metrics.json").write_text(json.dumps(
        {"args": vars(args), "folds": allm,
         "mean": {k: float(np.mean([m[k] for m in allm]))
                  for k in ["mpjpe", "pa_mpjpe", "pck50", "pck100",
                            "baseline_mean_pose"]}}, indent=2, default=str))
    print(f"\nwritten to {args.out}")
    print("metrics.json and pred_*.npz are enough to reproduce the evaluation.")


# ---------------------------------------------------------------- dry run
def dry_run(args):
    """Check torch FK against the numpy reference and bone-length exactness."""
    print("Dry run, no training.\n")
    torch.manual_seed(0)
    L = np.array([0.10, 0.40, 0.39, 0.10, 0.33, 0.44,
                  0.48, 0.22, 0.23, 0.51, 0.30, 0.23])
    B, T = 4, 64

    d = torch.randn(B, T, 12, 3)
    P_t = fk(d, torch.as_tensor(L, dtype=torch.float32)).numpy()
    P_n = np.stack([SK.forward(d[b].numpy(), L) for b in range(B)])
    print(f"1. forward kinematics torch vs numpy: "
          f"max deviation {np.abs(P_t-P_n).max():.2e} m")

    Lp = np.stack([np.linalg.norm(P_t[:, :, i] - P_t[:, :, SK.PARENTS[i]], axis=-1)
                   for i in range(1, 13)], axis=-1)
    print(f"2. bone lengths in the output: max error "
          f"{np.abs(Lp - L).max()*1000:.4f} mm  (must be 0)")

    model = Poser(L, hidden=64, layers=2)
    x = torch.randn(B, T, N_FEAT)
    out = model(x)
    print(f"3. shapes: input {tuple(x.shape)} -> pose {tuple(out['pos'].shape)}, "
          f"leaf pose {tuple(out['pos_leaf'].shape)}, directions {tuple(out['dirs'].shape)}")
    print(f"   parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    xa = augment.apply(x.clone(), "_mp_spatial", 1.0)
    g0 = x[:, :, 0:3]; g1 = xa[:, :, 0:3]
    print(f"4. augmentation: gravity axis unchanged? "
          f"max deviation {float((g0-g1).abs().max()):.2e}   "
          f"angular rate changed? {float((x[:,:,6:9]-xa[:,:,6:9]).abs().max()):.3f}")

    y = torch.randn(B, T, 13, 3) * 0.3
    yd = torch.nn.functional.normalize(torch.randn(B, T, 12, 3), dim=-1)
    loss, parts = PoseLoss()(out, y, yd)
    loss.backward()
    gn = sum(float(p.grad.norm()) for p in model.parameters() if p.grad is not None)
    print(f"5. loss {loss.item():.4f} {parts}, gradient norm sum {gn:.3f}")
    print("\nIf 1 and 2 are zero and 5 shows a non-zero gradient, the mechanics "
          "are sound.")


if __name__ == "__main__":
    main()
