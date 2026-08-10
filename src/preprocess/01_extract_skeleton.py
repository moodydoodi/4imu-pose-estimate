import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt
from tqdm import tqdm
import argparse
import concurrent.futures
from queue import Queue
from threading import Thread
import multiprocessing
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')
cv2.setNumThreads(0)

mp_pose = mp.solutions.pose
LPF_CUTOFF_HZ = 6.0

def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    n = len(data)
    if n < 15: 
        return data
    padlen = min(n // 2, 10)
    
    try:
        return filtfilt(b, a, data, padlen=padlen)
    except:
        return data

class VideoStream:
    def __init__(self, path, queue_size=256):
        self.stream = cv2.VideoCapture(str(path))
        self.q = Queue(maxsize=queue_size)
        self.stopped = False
        self.total_frames = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)

    def start(self):
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            if not self.q.full():
                grabbed, frame = self.stream.read()
                if not grabbed:
                    self.stop()
                    return
                frame = cv2.resize(frame, (1280, 720))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.q.put(frame)

    def read(self):
        return self.q.get()

    def more(self):
        return self.q.qsize() > 0 or not self.stopped

    def stop(self):
        self.stopped = True
        self.stream.release()

def process_subject_video(video_path, output_dir):
    vs = VideoStream(video_path).start()
    
    if vs.total_frames <= 0 or vs.fps <= 0:
        return

    data = []
    current_process = multiprocessing.current_process()
    worker_id = current_process._identity[0] if current_process._identity else 0
    
    with mp_pose.Pose(min_detection_confidence=0.5, 
                      min_tracking_confidence=0.5, 
                      model_complexity=2) as pose:
        
        frame_idx = 0
        with tqdm(total=vs.total_frames, desc=video_path.name[:15], position=worker_id, leave=True) as pbar:
            while vs.more():
                try:
                    img = vs.read()
                except:
                    break
                
                results = pose.process(img)
                row = [frame_idx, frame_idx / vs.fps]
                
                if results.pose_world_landmarks:
                    for lm in results.pose_world_landmarks.landmark:
                        row.extend([lm.x, lm.y, lm.z, lm.visibility])
                else:
                    row.extend([np.nan] * 132)
                
                data.append(row)
                frame_idx += 1
                pbar.update(1)
            
    vs.stop()
    
    if not data:
        return

    cols = ['frame', 'time']
    for i in range(33):
        cols.extend([f'j{i}_x', f'j{i}_y', f'j{i}_z', f'j{i}_vis'])
        
    df = pd.DataFrame(data, columns=cols)
    df = df.interpolate(method='linear', limit=30, limit_direction='both')
    
    cols_to_filter = [c for c in df.columns if c.startswith('j') and not c.endswith('vis')]
    
    for col in cols_to_filter:
        if df[col].isnull().any():
            df[col] = df[col].bfill().ffill().fillna(0)
        df[col] = butter_lowpass_filter(df[col].values, cutoff=LPF_CUTOFF_HZ, fs=vs.fps)
    
    out_path = output_dir / f"{video_path.stem}_gt_3d.csv"
    df.to_csv(out_path, index=False)

def process_wrapper(args):
    video_path, proc_dir = args
    process_subject_video(video_path, proc_dir)

PROJECT = Path(__file__).resolve().parents[2]     # repository root


def _dir(p, default):
    """Resolve a path against the repository root, not the working directory."""
    q = Path(p) if p else Path(default)
    return q if q.is_absolute() else (PROJECT / q)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/raw")
    parser.add_argument("--out", default="data/processed")
    args = parser.parse_args()

    raw_root = _dir(args.root, 'data/raw')
    out_root = _dir(args.out, 'data/processed')
    print(f"raw: {raw_root}\nout: {out_root}")
    if not raw_root.exists():
        return

    tasks = []
    for subj in raw_root.iterdir():
        if subj.is_dir():
            vids = list(subj.glob("*.mp4")) + list(subj.glob("*.mov")) + list(subj.glob("*.MOV"))
            if not vids:
                continue
            
            proc_dir = out_root / subj.name
            proc_dir.mkdir(parents=True, exist_ok=True)
            
            # Predict the output filename based on the logic in process_subject_video
            out_path = proc_dir / f"{vids[0].stem}_gt_3d.csv"
            
            # Check if file exists before queueing task
            if not out_path.exists():
                tasks.append((vids[0], proc_dir))

    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        executor.map(process_wrapper, tasks)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()