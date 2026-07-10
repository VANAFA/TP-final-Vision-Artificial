"""
Generate two paper figures illustrating the classical lane detection pipeline.
Uses detect_lane_lines() output directly so the ROI / colours exactly match
what the running system produces.

  fig4_lane_detection.png  — original frame vs detection overlay (green=left, red=right)
  fig5_lane_mask.png       — colour mask pipeline: raw mask → ROI-clipped mask
"""

from __future__ import annotations
import os, sys
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from lane_pid import detect_lane_lines, _white_yellow_mask

matplotlib.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":        9,
    "axes.titlesize":   9,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.03,
})

OUT   = os.path.join(ROOT, "paper_figures")
FRAME = os.path.join(ROOT, "debug_logs", "20260708_142507", "event_frame_00072.png")

img = cv2.imread(FRAME)
assert img is not None

# ── Run detection — use the overlay EXACTLY as the live system draws it ───────
left_pts, right_pts, overlay = detect_lane_lines(img)
print(f"Frame {img.shape}  left={left_pts}  right={right_pts}")

# ── Mask pipeline for fig5 ────────────────────────────────────────────────────
mask   = _white_yellow_mask(img)          # binary 0/255

# ROI binary (same trapezoid as _roi_mask with defaults row_top=0.50, row_bot=0.75)
h, w = img.shape[:2]
roi_bin = np.zeros((h, w), dtype=np.uint8)
roi_pts_arr = np.array([[
    [int(w * 0.0),  int(h * 0.75)],
    [int(w * 0.40), int(h * 0.50)],
    [int(w * 0.60), int(h * 0.50)],
    [int(w * 1.0),  int(h * 0.75)],
]], dtype=np.int32)
cv2.fillPoly(roi_bin, roi_pts_arr, 255)
mask_roi = cv2.bitwise_and(mask, roi_bin)

# Visualisation for panel c: dim outside ROI, highlight surviving mask pixels
vis_c = (img.astype(np.float32) * 0.28).astype(np.uint8)
in_roi = (roi_bin == 255)
vis_c[in_roi] = img[in_roi]                                # restore ROI at full brightness
lane_hit = (mask_roi == 255)
vis_c[lane_hit] = np.clip(
    img[lane_hit].astype(np.float32) * 0.2
    + np.array([0, 210, 60], np.float32) * 0.8,
    0, 255,
).astype(np.uint8)                                         # green tint on lane pixels


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Lane detection (use detect_lane_lines overlay directly)
# ════════════════════════════════════════════════════════════════════════════
fig4, axes4 = plt.subplots(1, 2, figsize=(7.16, 3.5))

axes4[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes4[0].set_title("(a) Input frame", fontsize=9)
axes4[0].axis("off")

axes4[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
axes4[1].set_title("(b) Lane detection output", fontsize=9)
axes4[1].axis("off")

patches = [
    mpatches.Patch(color="#00FF00", label="Left lane"),
    mpatches.Patch(color="#FF0000", label="Right lane"),
    mpatches.Patch(color="#00C8FF", label="ROI boundary"),
]
axes4[1].legend(handles=patches, loc="lower right",
                fontsize=7, framealpha=0.88, edgecolor="#999999")

plt.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.02, wspace=0.04)

for ext in ("pdf", "png"):
    p = os.path.join(OUT, f"fig4_lane_detection.{ext}")
    fig4.savefig(p, format=ext)
    print(f"  [fig4] {p}")
plt.close(fig4)


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — White/yellow colour-filter mask pipeline
# ════════════════════════════════════════════════════════════════════════════
fig5, axes5 = plt.subplots(1, 3, figsize=(7.16, 2.8))

axes5[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes5[0].set_title("(a) Input frame", fontsize=9)
axes5[0].axis("off")

axes5[1].imshow(mask, cmap="gray", vmin=0, vmax=255)
axes5[1].set_title("(b) HLS colour mask\n(full frame — sky fires too)", fontsize=8)
axes5[1].axis("off")

axes5[2].imshow(cv2.cvtColor(vis_c, cv2.COLOR_BGR2RGB))
axes5[2].set_title("(c) After ROI clipping\n(green = pixels fed to Hough)", fontsize=8)
axes5[2].axis("off")

plt.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.04)

for ext in ("pdf", "png"):
    p = os.path.join(OUT, f"fig5_lane_mask.{ext}")
    fig5.savefig(p, format=ext)
    print(f"  [fig5] {p}")
plt.close(fig5)

# tidy check images
for f in ("_check_overlay.png", "_check_mask.png"):
    fp = os.path.join(OUT, f)
    if os.path.exists(fp):
        os.remove(fp)

print("Done.")
