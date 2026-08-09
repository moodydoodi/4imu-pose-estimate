"""
serve.py — local launcher for the Movement Analysis Suite web dashboard.

Run ONCE:  python serve.py
It opens the dashboard in your browser and adds a small local API so you can
**run model inference from inside the dashboard** (no CLI switching):
  • scan a project root for checkpoints + subjects,
  • load a subject's data straight from disk,
  • run a checkpoint -> writes predictions__<name>.csv AND shows it live.

Only your own machine talks to it (localhost). Requires: torch (only for the
actual inference step) + the project's inference.py.
"""
import http.server, socketserver, json, os, sys, urllib.parse, importlib.util, webbrowser, threading, mimetypes, argparse, shutil, subprocess, uuid, time, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
JOBS = {}          # job_id -> {status, done, total, stage, name, url, mpjpe, error, t0}

def find_inference(explicit=None):
    cands = [explicit] if explicit else []
    cands += [HERE / "inference.py",
              HERE.parent / "src" / "poser" / "inference.py"]
    for c in cands:
        if c and Path(c).exists():
            spec = importlib.util.spec_from_file_location("inference", str(c))
            m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
            print(f"[serve] using inference: {c}")
            return m
    print("[serve] WARNING: inference.py not found — inference disabled. Pass --inference PATH.")
    return None

INF = None  # set in main

def scan_subjects(root):
    root = Path(root); out = []
    if not root.exists():
        return out
    for d in [root] + [p for p in root.rglob("*") if p.is_dir()]:
        try:
            names = [f.name.lower() for f in d.iterdir() if f.is_file()]
        except Exception:
            continue
        has_gt = any(("gt_3d" in n or n.startswith("ground_truth")) and n.endswith(".csv") for n in names)
        has_imu = any("aligned" in n and n.endswith(".csv") for n in names)
        if has_gt or has_imu:
            out.append({"dir": str(d), "name": d.name})
    return out

def subject_files(d):
    d = Path(d); res = {"gt": None, "sensors": {}, "models": {}, "topologies": {}, "video": None}
    TOK = {"Left Ankle": ["left_ankle"], "Right Ankle": ["right_ankle"],
           "Left Wrist": ["left_wrist"], "Right Wrist": ["right_wrist"]}
    for f in sorted(d.iterdir()):
        if not f.is_file():
            continue
        n = f.name.lower()
        if n.endswith((".mp4", ".mov")):
            if n.endswith("__h264.mp4"):        # derived copy — only use if nothing else
                if not res["video"]:
                    res["video"] = str(f)
            else:
                res["video"] = str(f)
            continue
        if not n.endswith(".csv"):
            continue
        if n.endswith(".topology.json"):
            continue                                   # handled together with its CSV
        if n.startswith(("prediction", "pred_")) or "predictions" in n:
            import re
            mn = re.sub(r"^(predictions|prediction|pred)[_]*", "", f.stem, flags=re.I) or "default"
            res["models"][mn] = str(f)
            topo = f.with_name(f.stem + ".topology.json")
            if topo.exists():
                res.setdefault("topologies", {})[mn] = str(topo)
            continue
        if ("gt_3d" in n or n.startswith("ground_truth")) and "head" not in n and "test" not in n:
            res["gt"] = str(f); continue
        if "mp_spatial" in n:
            continue
        for s, toks in TOK.items():
            if any(t in n for t in toks) and ("aligned" in n or s not in res["sensors"]):
                res["sensors"][s] = str(f)
    return res

def file_url(p):
    return "/api/file?path=" + urllib.parse.quote(str(p))


WEB_CODECS = {"h264", "vp8", "vp9", "av1"}

def ffprobe_codec(path):
    """Video codec of the first video stream (lowercase), or None if ffprobe is missing/fails."""
    if not shutil.which("ffprobe"):
        return None
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=codec_name", "-of", "default=nw=1:nk=1", str(path)],
                           capture_output=True, timeout=30)
        return (r.stdout.decode(errors="ignore").strip().lower() or None)
    except Exception:
        return None

def usable_h264(p):
    """True if a previously converted file exists AND is actually readable.
    Without ffprobe we can only fall back to a size check."""
    try:
        if not p.exists() or p.stat().st_size < 1024:
            return False
    except OSError:
        return False
    if not shutil.which("ffprobe"):
        return True                      # cannot verify — assume it is fine
    return ffprobe_codec(p) in WEB_CODECS # empty/None => unreadable => redo it

def ffprobe_duration(path):
    if not shutil.which("ffprobe"):
        return 0.0
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "default=nw=1:nk=1", str(path)], capture_output=True, timeout=30)
        return float(r.stdout.decode(errors="ignore").strip() or 0)
    except Exception:
        return 0.0

def run_convert_job(jid, src):
    """Background H.264 conversion with live progress (parsed from ffmpeg -progress)."""
    j = JOBS[jid]
    try:
        src = Path(src)
        if not src.exists():
            j.update(status="error", error="video not found"); return
        out = src.with_name(src.stem + "__h264.mp4")
        if usable_h264(out):                       # a good conversion already exists
            j.update(status="done", url=file_url(out)); return
        if not shutil.which("ffmpeg"):
            j.update(status="error", error="ffmpeg not installed / not on PATH"); return
        dur = ffprobe_duration(src)
        j.update(status="running", total=round(dur, 2), done=0, stage="converting to H.264")
        print(f"[convert] start {src.name} ({dur:.0f}s) -> {out.name}")
        # write to a temp file and rename at the end, so an interrupted run never
        # leaves a half-written file that looks like a finished conversion
        tmp = out.with_name(out.stem + ".part.mp4")
        proc = subprocess.Popen(["ffmpeg", "-y", "-i", str(src), "-c:v", "libx264", "-preset", "veryfast",
                                 "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart",
                                 "-progress", "pipe:1", "-nostats", str(tmp)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("out_time_ms="):
                try: j["done"] = round(int(line.split("=")[1]) / 1e6, 2)
                except Exception: pass
            elif line == "progress=end":
                j["done"] = j.get("total", 0)
        ret = proc.wait()
        if ret != 0:
            err = (proc.stderr.read() or "")[-300:]
            print("[convert] ffmpeg error:", err)
            try: tmp.unlink()
            except OSError: pass
            j.update(status="error", error="ffmpeg failed: " + err); return
        os.replace(str(tmp), str(out))             # atomic: only now does out exist
        print(f"[convert] done -> {out}")
        j.update(status="done", url=file_url(out))
    except Exception as e:
        print("[convert] ERROR:\n" + traceback.format_exc())
        j.update(status="error", error=str(e))


def run_infer_job(jid, body):
    """Background worker: runs one checkpoint and streams progress into JOBS[jid]."""
    j = JOBS[jid]
    try:
        ck = body["checkpoint"]; spec = INF.ModelSpec(ck)
        stride = int(body.get("stride", 1) or 1)
        sources = INF.find_model_sources(body["root"]) if body.get("root") else []
        msrc = INF.resolve_model_src(ck, sources) if spec.arch == "stgcn" else None
        try:
            dev = INF.device_info() if hasattr(INF, "device_info") else "cpu"
        except Exception:
            dev = "cpu"
        print(f"[infer] start '{spec.name}' arch={spec.arch} win={spec.window} stride={stride} device={dev}")
        j.update(status="running", stage=f"loading model on {dev}", device=dev)

        def prog(done, total):
            j["done"] = done; j["total"] = total; j["stage"] = f"running on {dev}"

        name, _, mets = INF.infer_to_files(body["dir"], ck, model_src=msrc,
                                            model_name=spec.name, progress=prog, stride=stride)
        pred = str(Path(body["dir"]) / f"predictions__{name}.csv")
        dur = time.time() - j["t0"]
        print(f"[infer] done '{name}' in {dur:.1f}s -> {pred}")
        j.update(status="done", name=name, url=file_url(pred),
                 mpjpe=mets.get("mpjpe_mm"), secs=round(dur, 1))
    except ModuleNotFoundError as e:
        mod = getattr(e, "name", None) or str(e)
        msg = (f"module '{mod}' is missing in the Python running serve.py ({sys.executable}). "
               f"Start serve.py from your venv, then: pip install {mod}")
        print("[infer] ERROR:", msg)
        j.update(status="error", error=msg)
    except Exception as e:
        print("[infer] ERROR:\n" + traceback.format_exc())
        j.update(status="error", error=str(e))


class H(http.server.BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urllib.parse.urlparse(self.path); q = urllib.parse.parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            return self.serve_file(HERE / "index.html")
        if u.path == "/api/scan":
            root = q.get("root", [""])[0]
            cks, srcs = [], []
            if INF:
                for p, s in INF.list_checkpoints(root):
                    cks.append({"path": str(p), "name": s.name, "arch": s.arch,
                                "window": s.window, "hidden": s.hidden_dim, "layers": s.num_layers})
                srcs = [{"path": str(p), "arch": a} for p, a in INF.find_model_sources(root)]
            return self._json({"subjects": scan_subjects(root), "checkpoints": cks, "sources": srcs,
                               "torch": _has_torch(), "ffmpeg": shutil.which("ffmpeg") is not None})
        if u.path == "/api/data":
            d = q.get("dir", [""])[0]; sf = subject_files(d)
            v_url = v_src = None; v_play = None
            if sf["video"]:
                orig = Path(sf["video"])
                h264 = orig.with_name(orig.stem + "__h264.mp4")
                # only trust a previous conversion if it is actually readable (an
                # interrupted run can leave a truncated file behind)
                if usable_h264(h264):
                    v_url = file_url(h264); v_play = True; v_src = str(orig)
                else:
                    if h264.exists():
                        print(f"[video] ignoring unreadable {h264.name} — using the original")
                    codec = ffprobe_codec(orig)         # None if ffprobe missing
                    v_play = (codec in WEB_CODECS) if codec else None
                    v_url = file_url(orig); v_src = str(orig)
            return self._json({
                "gt": file_url(sf["gt"]) if sf["gt"] else None,
                "sensors": {k: file_url(v) for k, v in sf["sensors"].items()},
                "models": {k: file_url(v) for k, v in sf["models"].items()},
                "topologies": {k: file_url(v) for k, v in sf.get("topologies", {}).items()},
                "video": v_url, "video_src": v_src, "video_playable": v_play,
                "ffmpeg": shutil.which("ffmpeg") is not None})
        if u.path == "/api/progress":
            j = JOBS.get(q.get("job", [""])[0])
            return self._json(j or {"status": "unknown"})
        if u.path == "/api/file":
            return self.serve_file(Path(q.get("path", [""])[0]))
        if u.path == "/api/pickfolder":
            try:
                import tkinter, tkinter.filedialog as fd
                r = tkinter.Tk(); r.withdraw(); r.attributes("-topmost", True)
                p = fd.askdirectory(title="Select your project root (e.g. TechnicalColl)")
                r.destroy()
                return self._json({"path": p or None})
            except Exception as e:
                return self._json({"path": None, "error": str(e)})
        self.send_error(404)

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/api/transcode":
            ln = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(ln) or b"{}")
            src = Path(body.get("path", ""))
            if not src.exists():
                return self._json({"error": "video not found"}, 404)
            if not shutil.which("ffmpeg"):
                return self._json({"error": "ffmpeg not installed — install it or transcode manually"}, 500)
            out = src.with_name(src.stem + "__h264.mp4")
            if not out.exists():
                try:
                    subprocess.run(["ffmpeg", "-y", "-i", str(src), "-c:v", "libx264",
                                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart",
                                    str(out)], check=True, capture_output=True, timeout=1800)
                except subprocess.CalledProcessError as e:
                    return self._json({"error": "ffmpeg failed: " + e.stderr.decode(errors="ignore")[-300:]}, 500)
                except Exception as e:
                    return self._json({"error": str(e)}, 500)
            return self._json({"url": file_url(out)})
        if u.path == "/api/video_convert":
            ln = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(ln) or b"{}")
            src = body.get("path", "")
            if not src or not Path(src).exists():
                return self._json({"error": "video not found"}, 404)
            jid = uuid.uuid4().hex[:12]
            JOBS[jid] = {"status": "starting", "done": 0, "total": 0,
                         "stage": "starting ffmpeg", "t0": time.time()}
            threading.Thread(target=run_convert_job, args=(jid, src), daemon=True).start()
            return self._json({"job": jid})
        if u.path == "/api/infer":
            ln = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(ln) or b"{}")
            if not INF:
                return self._json({"error": "inference.py not loaded"}, 500)
            jid = uuid.uuid4().hex[:12]
            JOBS[jid] = {"status": "starting", "done": 0, "total": 0,
                         "stage": "loading model + preparing IMU data", "t0": time.time()}
            threading.Thread(target=run_infer_job, args=(jid, body), daemon=True).start()
            return self._json({"job": jid})
        self.send_error(404)

    def serve_file(self, p):
        p = Path(p)
        if not p.exists() or not p.is_file():
            return self.send_error(404)
        ctype = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
        if p.suffix.lower() == ".csv":
            ctype = "text/plain"
        size = p.stat().st_size
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            a, b = rng[6:].split("-"); start = int(a or 0); end = int(b) if b else size - 1
            end = min(end, size - 1); length = end - start + 1
            self.send_response(206); self.send_header("Content-Type", ctype)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Accept-Ranges", "bytes"); self.send_header("Content-Length", str(length))
            self.end_headers()
            with open(p, "rb") as f:
                f.seek(start); self.wfile.write(f.read(length))
            return
        self.send_response(200); self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Content-Length", str(size)); self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        with open(p, "rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                self.wfile.write(chunk)


def _has_torch():
    return importlib.util.find_spec("torch") is not None


def main():
    global INF
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--inference", default=None, help="path to inference.py")
    args = ap.parse_args()
    INF = find_inference(args.inference)
    port = args.port
    class TS(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
    while True:
        try:
            httpd = TS(("127.0.0.1", port), H); break
        except OSError:
            port += 1
    url = f"http://127.0.0.1:{port}/"
    print(f"[serve] Movement Analysis Suite → {url}")
    print(f"[serve] python: {sys.executable}")
    tqdm_ok = importlib.util.find_spec("tqdm") is not None
    try:
        dev = INF.device_info() if (INF and hasattr(INF, "device_info")) else ("torch missing" if not _has_torch() else "cpu")
    except Exception:
        dev = "unknown"
    print(f"[serve] torch: {_has_torch()} · device: {dev} · tqdm: {tqdm_ok} · ffmpeg: {shutil.which('ffmpeg') is not None} · Ctrl+C to stop")
    if not _has_torch():
        print("[serve] ⚠ torch NOT found in THIS python — to run models, start serve.py from your venv:")
        print("[serve]     (activate venv)  then  python serve.py    — the venv's torch/tqdm must be visible here")
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] bye")


if __name__ == "__main__":
    main()
