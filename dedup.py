# dedup.py
import argparse
import glob
import os
import shutil
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="in_dir", required=True)
ap.add_argument("--out", dest="out_dir", default="frames_dedup")
ap.add_argument("--sim-thresh", type=float, default=0.97,
                help="порог схожести (1.0 = идентичны)")
args = ap.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
files = sorted(glob.glob(os.path.join(args.in_dir, "*.jpg")))

prev_hist = None
kept = 0
for f in files:
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    hist = cv2.calcHist([img], [0], None, [64], [0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    if prev_hist is not None:
        sim = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
        if sim > args.sim_thresh:
            continue  
    shutil.copy(f, args.out_dir)
    prev_hist = hist
    kept += 1

print(f"[ok] {kept} из {len(files)} оставлено в {args.out_dir}")