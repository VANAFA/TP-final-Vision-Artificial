"""
Generate two paper figures illustrating the classical lane detection pipeline:

  fig4_lane_detection.png  — original frame with detected left/right lane lines
  fig5_lane_mask.png       — white/yellow colour-filter mask overlaid on the frame
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

from lane_pid import detect_lane_lines, _white_yellow_mask, _roi_mask

matplotlib.rcParams.update({
    "font.family":    "serif",
    "font.serif":     ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":      9,
    "axes.titlesize": 9,
    "figure.dpi":     150,
    "savefig.dpi":    300,
    "savefig.bbox":   "tight",
    "savefig.pad_inches": 0.03,
})

OUT   = os.path.join(ROOT, "paper_figures")
FRAME = os.path.join(ROOT, "debug_logs", "20260708_142507", "event_frame_00072.png")

img_raw = cv2.imread(FRAME)
assert img_raw is not None, f"Could not read {FRAME}"

# ── Crop away the bottom HUD strip (steering wheel / status panel) ────────────
# The HUD occupies roughly the bottom 30 % of the 640×640 frame.
# Keeping 0..460 px keeps the full road view and removes the HUD.
H_CROP = 460
img = img_raw[:H_CROP].copy()

# ── Run lane detection ────────────────────────────────────────────────────────
left_pts, right_pts, overlay_full = detect_lane_lines(img)

# Rebuild a clean overlay (no small white-mask inset; just lines + ROI)
h, w = img.shape[:2]
overlay_clean = img.copy()

# NOTE: cv2 uses BGR; matplotlib imshow receives the result of BGR→RGB
# conversion, so every colour below is specified in BGR so that the
# displayed (RGB) colour is the intended one.
#   BGR(0,   230, 0)   → RGB(0,   230, 0)   = green   (left lane)
#   BGR(230, 0,   0)   → RGB(0,   0,   230) = blue    (right lane)
#   BGR(255, 200, 0)   → RGB(0,   200, 255) = cyan    (ROI)
#   BGR(0,   230, 230) → RGB(230, 230, 0)   = yellow  (centre)

# ROI trapezoid (→ cyan in display)
roi_pts = np.array([[
    [int(w * 0.0),  int(h * 0.75)],
    [int(w * 0.40), int(h * 0.50)],
    [int(w * 0.60), int(h * 0.50)],
    [int(w * 1.0),  int(h * 0.75)],
]], dtype=np.int32)
cv2.polylines(overlay_clean, roi_pts, isClosed=True, color=(255, 200, 0), thickness=2)

# Lane lines
if left_pts is not None:
    cv2.line(overlay_clean, left_pts[0], left_pts[1], (0, 230, 0), 5)   # → green
if right_pts is not None:
    cv2.line(overlay_clean, right_pts[0], right_pts[1], (230, 0, 0), 5) # → blue

# Mid-lane marker (→ yellow in display)
if left_pts is not None and right_pts is not None:
    cx = (left_pts[0][0] + right_pts[0][0]) // 2
    cy = left_pts[0][1]
    cv2.circle(overlay_clean, (cx, cy), 10, (0, 230, 230), -1)
    cv2.line(overlay_clean, (w//2, cy), (cx, cy), (0, 230, 230), 2)

# ── Colour mask ───────────────────────────────────────────────────────────────
mask = _white_yellow_mask(img)          # binary 0/255

# ROI mask: same trapezoid used by _roi_mask(), but applied to the colour mask
import cv2 as _cv2
roi_mask_bin = np.zeros(mask.shape, dtype=np.uint8)
roi_trap = np.array([[
    [int(w * 0.0),  int(h * 0.75)],
    [int(w * 0.40), int(h * 0.50)],
    [int(w * 0.60), int(h * 0.50)],
    [int(w * 1.0),  int(h * 0.75)],
]], dtype=np.int32)
_cv2.fillPoly(roi_mask_bin, roi_trap, 255)
mask_in_roi = cv2.bitwise_and(mask, roi_mask_bin)  # only lane pixels inside ROI

# Darkened overlay: non-ROI area dimmed; surviving mask pixels → bright green
dark = (img.astype(np.float32) * 0.30).astype(np.uint8)
roi_overlay = dark.copy()
# Restore ROI area at full brightness
roi_region = (roi_mask_bin == 255)
roi_overlay[roi_region] = img[roi_region]
# Highlight surviving lane pixels inside ROI as green
lane_in_roi = (mask_in_roi == 255)
roi_overlay[lane_in_roi] = np.clip(
    img[lane_in_roi].astype(np.float32) * 0.25
    + np.array([0, 210, 70], dtype=np.float32) * 0.75,
    0, 255,
).astype(np.uint8)


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Lane detection overlay
# ════════════════════════════════════════════════════════════════════════════
fig4, axes4 = plt.subplots(1, 2, figsize=(7.16, 3.2))

axes4[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes4[0].set_title("(a) Input frame", fontsize=9)
axes4[0].axis("off")

axes4[1].imshow(cv2.cvtColor(overlay_clean, cv2.COLOR_BGR2RGB))
axes4[1].set_title("(b) Detected lane lines", fontsize=9)
axes4[1].axis("off")

# Legend
patches = [
    mpatches.Patch(color="#00E600", label="Left lane"),
    mpatches.Patch(color="#0000E6", label="Right lane"),
    mpatches.Patch(color="#00C8FF", label="ROI boundary"),
    mpatches.Patch(color="#E6E600", label="Lane centre"),
]
axes4[1].legend(handles=patches, loc="lower right",
                fontsize=6.5, framealpha=0.88, edgecolor="#999999")

plt.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.02, wspace=0.04)

for ext in ("pdf", "png"):
    p = os.path.join(OUT, f"fig4_lane_detection.{ext}")
    fig4.savefig(p, format=ext)
    print(f"  [fig4] {p}")
plt.close(fig4)


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — White/yellow colour-filter mask
# ════════════════════════════════════════════════════════════════════════════
fig5, axes5 = plt.subplots(1, 3, figsize=(7.16, 2.6))

# Panel a — original
axes5[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes5[0].set_title("(a) Input frame", fontsize=9)
axes5[0].axis("off")

# Panel b — binary mask (full frame; sky/road both pass the colour filter)
axes5[1].imshow(mask, cmap="gray")
axes5[1].set_title("(b) HLS colour mask\n(L > 140, S < 100) — full frame", fontsize=8)
axes5[1].axis("off")

# Panel c — ROI-masked overlay (dim outside ROI, green inside where mask fires)
axes5[2].imshow(cv2.cvtColor(roi_overlay, cv2.COLOR_BGR2RGB))
axes5[2].set_title("(c) After ROI clipping\n(green = pixels fed to Hough)", fontsize=8)
axes5[2].axis("off")

plt.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.04)

for ext in ("pdf", "png"):
    p = os.path.join(OUT, f"fig5_lane_mask.{ext}")
    fig5.savefig(p, format=ext)
    print(f"  [fig5] {p}")
plt.close(fig5)

print("Done.")
