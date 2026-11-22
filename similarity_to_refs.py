# similarity_to_refs.py
import argparse
import glob
import os
import shutil
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="in_dir", required=True)
ap.add_argument("--refs", required=True,
                help="папка с 3-5 эталонными кадрами шва")
ap.add_argument("--out", dest="out_dir", default="frames_ranked")
ap.add_argument("--top-k", type=int, default=500)
args = ap.parse_args()

# загружаем эталоны
refs = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) 
        for f in glob.glob(os.path.join(args.refs, "*.jpg"))]
refs = [cv2.resize(r, (320, 240)) for r in refs if r is not None]

files = sorted(glob.glob(os.path.join(args.in_dir, "*.jpg")))
scored = []
for f in files:
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    if img is None: continue
    img = cv2.resize(img, (320, 240))
    h1 = cv2.calcHist([img], [0], None, [64], [0, 256])
    cv2.normalize(h1, h1, 0, 1, cv2.NORM_MINMAX)
    best = 0
    for r in refs:
        h2 = cv2.calcHist([r], [0], None, [64], [0, 256])
        cv2.normalize(h2, h2, 0, 1, cv2.NORM_MINMAX)
        sim = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
        best = max(best, sim)
    scored.append((best, f))

scored.sort(reverse=True)
os.makedirs(args.out_dir, exist_ok=True)
for i, (s, f) in enumerate(scored[:args.top_k]):
    new_name = f"{i:04d}_sim{s:.3f}_{os.path.basename(f)}"
    shutil.copy(f, os.path.join(args.out_dir, new_name))

print(f"[ok] {args.top_k} лучших кадров в {args.out_dir}")