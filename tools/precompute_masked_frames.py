"""One-time preprocessing: bake YOLOP's own road/lane/vehicle segmentation into a hard
pixel-space mask for every frame in the comma2k19 manifest, writing the masked frames to a
parallel directory that mirrors `frames_dir`'s own per-segment subfolder structure.

Why this is a separate script and not done in `Comma2k19ControlDataset.__getitem__`: masking
needs `lane_mask.segment_full_res`, which runs a full frozen-backbone forward pass per image.
Doing that on the fly inside `__getitem__` would mean every epoch, every sample, pays for a
second backbone forward (the training loop already runs the backbone forward once per batch
inside `YOLOPControlNet.forward`) -- for a 10-epoch run over 122K+ train frames that's a huge,
repeated, and entirely avoidable cost. Running it once here and pointing
`Comma2k19ControlDataset.frames_root` at the output directory instead means the per-sample
training path stays exactly what it already was (`cv2.imread` + `preprocess_image_bgr`), just
reading already-masked pixels.

Uses `full_road_mask` only (drivable area OR lane line OR vehicle, thresholded) -- not
`fit_lane_curves`/`road_mask_with_lane_attention`'s curve-fitting, which is for the closed
loop's optional visual attention overlay, not for training: training should see the same
"normal YOLOP road+lane+vehicle" signal the live "road" mask mode already uses, not an
extra parametric line model on top.

Idempotent/resumable: frames whose output file already exists are skipped, so an interrupted
run (or a manifest that grows later) can just be re-run.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from control_finetune import YOLOPControlNet  # noqa: E402
from lane_mask import apply_mask, full_road_mask, segment_full_res  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/comma2k19/manifest.csv")
    parser.add_argument("--frames-dir", default="data/comma2k19/frames")
    parser.add_argument("--out-dir", default="data/comma2k19/frames_masked")
    parser.add_argument("--threshold", type=float, default=0.7,
                         help="da/ll sigmoid threshold, passed through to full_road_mask.")
    parser.add_argument("--log-every", type=int, default=500)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[precompute_masked_frames] device={device}")

    frames_root = Path(args.frames_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest)
    image_paths = df["image_path"].drop_duplicates().tolist()
    print(f"[precompute_masked_frames] {len(image_paths)} unique frames in {args.manifest}")

    # Only need the frozen backbone + its own NMS thresholds (DET_CONF_THRES/DET_IOU_THRES),
    # which segment_full_res reads off the model object -- the trainable heads it also builds
    # are unused here but cheap (94K params) and this is the same loading path already used
    # everywhere else in the project, so there's no separate "backbone-only" constructor to
    # keep in sync with it.
    model = YOLOPControlNet(device=device).to(device)
    model.eval()

    n_done, n_skipped, n_failed = 0, 0, 0
    t_start = time.time()
    for i, rel_path in enumerate(image_paths, start=1):
        src = frames_root / rel_path
        dst = out_root / rel_path
        if dst.exists():
            n_skipped += 1
            continue

        img_bgr = cv2.imread(str(src))
        if img_bgr is None:
            print(f"[precompute_masked_frames] WARNING: could not read {src}, skipping.")
            n_failed += 1
            continue

        da_conf, ll_conf, vehicle_boxes = segment_full_res(model, img_bgr, device)
        mask = full_road_mask(da_conf, ll_conf, vehicle_boxes, threshold=args.threshold)
        masked = apply_mask(img_bgr, mask)

        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), masked)
        n_done += 1

        if i % args.log_every == 0 or i == len(image_paths):
            elapsed = time.time() - t_start
            rate = n_done / elapsed if elapsed > 0 else 0.0
            remaining = len(image_paths) - i
            eta_min = (remaining / rate / 60.0) if rate > 0 else float("nan")
            print(f"[precompute_masked_frames] {i}/{len(image_paths)} "
                  f"(done={n_done} skipped={n_skipped} failed={n_failed}) "
                  f"{rate:.1f} frames/s, ETA {eta_min:.1f} min")

    print(f"[precompute_masked_frames] finished: done={n_done} skipped={n_skipped} "
          f"failed={n_failed} total={len(image_paths)} -> {out_root}")


if __name__ == "__main__":
    main()
