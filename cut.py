# cut.py
import argparse
import csv
import os
import time

import cv2
import numpy as np


def is_seam_candidate(img_gray, block_size=32, texture_thresh=1.2,
                      gradient_thresh=2.0, blur_size=5):
    """
    Собирает кадры с аномалиями. Пороги намеренно занижены,
    чтобы в выборку попали и швы, и провода, и грязь.
    """
    h, w = img_gray.shape
    blurred = cv2.GaussianBlur(img_gray, (blur_size, blur_size), 0)
    
    gh, gw = h // block_size, w // block_size
    stds = []
    for y in range(gh):
        for x in range(gw):
            block = blurred[y*block_size:(y+1)*block_size, 
                            x*block_size:(x+1)*block_size]
            stds.append(float(block.std()))
    if len(stds) < 4:
        return False, 0.0
    stds = np.array(stds)
    texture_score = float(stds.max() / (np.median(stds) + 1e-6))
    has_texture = texture_score > texture_thresh
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    
    grad_peak = np.percentile(grad_mag, 99)
    grad_mean = grad_mag.mean() + 1e-6
    gradient_score = float(grad_peak / grad_mean)
    has_gradient = gradient_score > gradient_thresh

    score = max(texture_score / texture_thresh, gradient_score / gradient_thresh)
    return (has_texture or has_gradient), score


def process_video(video_path, args, out_cand, csv_path):
    """Обрабатывает видео потоково, пишет кандидатов и CSV."""
    os.makedirs(out_cand, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[skip] не открывается: {video_path}")
        return 0

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(1, int(round(src_fps / args.fps)))
    print(f"[start] {video_path}")
    print(f"        src_fps={src_fps:.2f} total={total} step={step}")

    saved = 0
    idx = 0
    t0 = time.time()

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "frame_idx", "timestamp_sec", "score",
                         "saved_path"])

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if idx % step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                is_cand, score = is_seam_candidate(
                    gray,
                    block_size=args.block_size,
                    texture_thresh=args.texture_thresh,
                    gradient_thresh=args.gradient_thresh,
                    blur_size=args.blur_size
                )

                if is_cand:
                    ts = idx / src_fps
                    # Имя файла: score_таймкод.jpg (чтобы сортировать по уверенности)
                    name = f"score_{score:.2f}_t{ts:.1f}s.jpg"
                    path = os.path.join(out_cand, name)
                    cv2.imwrite(path, frame)
                    writer.writerow([os.path.basename(video_path),
                                     idx, f"{ts:.2f}", f"{score:.2f}", path])
                    saved += 1

            if idx and idx % 1000 == 0:
                el = time.time() - t0
                pct = (idx / total * 100) if total else 0
                print(f"        {idx}/{total} ({pct:.1f}%) "
                      f"кандидатов={saved} {el:.0f}s")

            idx += 1

    cap.release()
    print(f"[done] {video_path}: прочитано={idx}, кандидатов={saved}")
    return saved


def collect_videos(arg):
    exts = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm")
    if os.path.isdir(arg):
        result = []
        for root, _, files in os.walk(arg):
            for f in sorted(files):
                if f.lower().endswith(exts):
                    result.append(os.path.join(root, f))
        return result
    return [arg]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="видеофайл или папка с видео")
    ap.add_argument("--fps", type=float, default=0.25,
                    help="кадров в секунду для анализа")
    ap.add_argument("--out-cand", default="frames_candidates")
    ap.add_argument("--block-size", type=int, default=32)
    # СНИЖЕНЫ ПОРОГИ: собираем все подозрительное
    ap.add_argument("--texture-thresh", type=float, default=1.2)
    ap.add_argument("--gradient-thresh", type=float, default=2.0)
    ap.add_argument("--blur-size", type=int, default=5)
    
    args = ap.parse_args()

    videos = collect_videos(args.input)
    if not videos:
        raise SystemExit(f"Не найдено видео в: {args.input}")
    print(f"[info] видео для обработки: {len(videos)}")

    total_saved = 0
    for v in videos:
        base = os.path.splitext(os.path.basename(v))[0]
        out_cand = os.path.join(args.out_cand, base)
        csv_path = os.path.join(args.out_cand, f"{base}.csv")
        total_saved += process_video(v, args, out_cand, csv_path)

    print(f"\n[all done] всего кандидатов: {total_saved}")
    print(f"[hint] проверь {args.out_cand}/ — там по папке на каждое видео")


if __name__ == "__main__":
    main()