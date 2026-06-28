"""Hard input-pixel masking for YOLOPControlNet: turns the frozen YOLOP backbone's own
segmentation/detection into a *pre*-processing step (mask the image, then feed the masked image
to the model) rather than only a post-hoc feature-pooling weight
(`YOLOPControlNet._road_mask`/`_masked_pool`, applied *after* the encoder has already looked at
the full frame).

Convolutional receptive fields mean masking only at the pooling step doesn't fully stop
background content from influencing the encoder's features at road-adjacent locations -- a
tree right next to the road still bleeds into nearby feature-map cells through the backbone's
own convolutions, before any mask is ever applied. Blacking out non-road pixels in the *image*
itself, before the backbone sees it at all, is a strictly stronger cut. This module only wires
that into the closed loop for now (`2 Closed Loop BeamNG Control.ipynb`) -- doing the same
during training (`1 Fine Tuning.ipynb`/`control_finetune.py`) is a later step, once this is
validated live.

The base mask keeps detected vehicles visible (see `vehicle_mask` below) -- cars sit *on* the
road but aren't "drivable surface", so da_seg_out alone excludes them; without this, brake's
masked image feature would carry zero visual signal about the thing it's braking for, even
though it's right there in the frame.

- `full_road_mask`: da_seg_out OR ll_seg_out OR a detected vehicle, thresholded -- the whole
  drivable area + visible lane lines + cars, the same signal `_road_mask` already uses, just
  applied as a hard image mask instead of a soft pooling weight. This is the base everything
  else builds on.
- `fit_lane_curves` / `draw_lane_curves`: two earlier attempts at isolating *just this lane*
  (filling the area between its two lines; later, keeping only the two line shapes) both turned
  out fragile in practice -- masking based on raw, noisy per-pixel/per-component segmentation is
  sensitive to single bad detections and gaps. This replaces raw-pixel masking with an explicit
  *parametric model* of where the two lines are: fit a quadratic `x = f(y)` to each side's
  thresholded pixels (a curve from near the car up toward the middle of the frame, never lower
  -- lines reliably curve, not zig-zag, across that range), restricted to a horizontal band
  around frame-center so far-lane lines on a multi-lane road get excluded before they can ever
  be selected ("our lines are almost always in the same part of the screen"). If one side has
  too few confident points, it's reconstructed by assuming the two lines are parallel: same
  curve shape as the side that *was* found, shifted by an empirically-typical lane width. The
  result is a smooth, consistent curve instead of whatever gaps/noise the raw segmentation had
  -- drawn as a highlighted overlay on top of `full_road_mask`'s output (not used to cut
  anything out), so the model sees the whole road+cars as before, with attention drawn to
  exactly where the lane lines mathematically are.

All of these run the frozen backbone once (`segment_full_res`) on the raw (un-letterboxed)
frame to get full-resolution da_seg_out/ll_seg_out/det_out, since masking needs to happen
before `preprocess_image_bgr`'s letterbox + normalize step, not after.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from control_finetune import preprocess_image_bgr


def segment_full_res(model, img_bgr: np.ndarray, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs the frozen backbone once on `img_bgr` and returns everything masking needs, all at
    `img_bgr`'s own native resolution (not 640x640):

    - `da_conf`, `ll_conf`: `(H, W)` float arrays in [0, 1] -- the letterbox pad bars cropped
      back off and the rest resized to the native frame size, mirroring `resize_unscale`'s own
      geometry exactly so they line up pixel-for-pixel with the unmasked photo.
    - `vehicle_boxes`: `(N, 6)` array of `(x1, y1, x2, y2, conf, cls)` after NMS, rescaled from
      640x640 letterboxed space back to the same native coordinates (mirrors
      `yolop_beamng.py`'s own box rescaling)."""
    from lib.core.general import non_max_suppression  # YOLOP_REPO already on sys.path

    h, w = img_bgr.shape[:2]
    image_tensor = preprocess_image_bgr(img_bgr).unsqueeze(0).to(device)
    with torch.no_grad():
        det_out, da_seg_out, ll_seg_out = model.backbone(image_tensor)
    da_640 = da_seg_out[0, 1].float().cpu().numpy()
    ll_640 = ll_seg_out[0, 1].float().cpu().numpy()

    r = min(640 / h, 640 / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    dw, dh = (640 - new_w) // 2, (640 - new_h) // 2
    da_cropped = da_640[dh:dh + new_h, dw:dw + new_w]
    ll_cropped = ll_640[dh:dh + new_h, dw:dw + new_w]
    da_conf = cv2.resize(da_cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    ll_conf = cv2.resize(ll_cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    z_cat = det_out[0].float()  # (1, N, 6): xywh (640-space) + obj_conf + cls_conf
    with torch.no_grad():
        preds = non_max_suppression(z_cat, conf_thres=model.DET_CONF_THRES, iou_thres=model.DET_IOU_THRES)
    vehicle_boxes = preds[0].cpu().numpy()
    if vehicle_boxes.shape[0] > 0:
        vehicle_boxes = vehicle_boxes.copy()
        vehicle_boxes[:, [0, 2]] = (vehicle_boxes[:, [0, 2]] - dw) / r
        vehicle_boxes[:, [1, 3]] = (vehicle_boxes[:, [1, 3]] - dh) / r

    return da_conf, ll_conf, vehicle_boxes


def vehicle_mask(shape_hw: tuple[int, int], vehicle_boxes: np.ndarray) -> np.ndarray:
    """Fills detected vehicle boxes (native pixel coords, from `segment_full_res`) into a
    binary `(H, W)` mask."""
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    for box in vehicle_boxes:
        x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
        x2, y2 = min(w, int(box[2])), min(h, int(box[3]))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
    return mask


def full_road_mask(
    da_conf: np.ndarray,
    ll_conf: np.ndarray,
    vehicle_boxes: np.ndarray,
    threshold: float = 0.7,
) -> np.ndarray:
    """Binary `(H, W)` mask: drivable area OR lane line OR a detected vehicle, thresholded.

    `threshold=0.7` (not the more obvious 0.5): empirically (checked across several comma2k19
    frames) this checkpoint's da/ll sigmoid outputs are sharply bimodal -- background sits at a
    near-constant ~0.41, road/lane at ~1.0 -- so anything comfortably between those two clusters
    works; 0.7 leaves equal margin on both sides rather than hugging the background cluster."""
    road = (da_conf > threshold) | (ll_conf > threshold)
    veh = vehicle_mask(da_conf.shape, vehicle_boxes).astype(bool)
    return (road | veh).astype(np.uint8)


# Fallback horizontal offset between this lane's two lines, as a *fraction of frame width*
# (not a fixed pixel count -- BeamNG's 640-wide capture and comma2k19's 1164-wide frames need
# very different pixel values for "about half the frame"). Only used to reconstruct a side that
# has too few confident points to fit on its own. Measured empirically: independently fitting
# both sides on several comma2k19 frames where both were clearly visible and comparing them at
# their deepest common row gave ~0.49-0.59x frame width -- 0.5 sits in the middle of that.
DEFAULT_LANE_OFFSET_FRAC = 0.5


def fit_lane_curves(
    ll_conf: np.ndarray,
    threshold: float = 0.7,
    y_top_frac: float = 0.45,
    y_bottom_frac: float = 0.98,
    max_offset_frac: float = 0.35,
    degree: int = 2,
    min_points: int = 30,
    bin_height: int = 8,
    min_bins: int = 5,
    min_span_for_quadratic: int = 150,
):
    """Fits a smooth `x = f(y)` to each of this lane's two bounding lines and returns
    `(left_fit, right_fit, reconstructed)`. `left_fit`/`right_fit` are each either `None` or
    `(coeffs, (y_min, y_max))` -- `coeffs` for `np.polyval`, `(y_min, y_max)` the row range
    actually backed by data (never extrapolated beyond it, since a quadratic's curvature term
    diverges fast outside the rows it was fit on). `reconstructed` is `"left"`, `"right"`, or
    `None`, naming whichever side (if any) came from the parallel-lines fallback rather than
    being fit directly from segmentation pixels.

    1. Threshold `ll_conf`, restrict to rows in `[y_top_frac, y_bottom_frac]` of the frame --
       lines reliably curve from near the car up to about the middle of the frame; above that
       segmentation confidence/perspective makes individual line pixels unreliable to fit
       against.
    2. Also restrict to columns within `max_offset_frac` of frame-center. This lane's lines are
       almost always in roughly the same part of the screen (the camera is mounted on the
       vehicle's own centerline); a line from an adjacent lane on a wide multi-lane road is
       reliably farther out than that, so excluding it here means it's never even a candidate,
       rather than relying on a later "pick the nearest one" step to reject it.
    3. Split the remaining points at frame-center. Fitting a polynomial directly to every
       thresholded pixel weights it by how *thick* the line's segmentation happens to be at
       each row, not by its actual centerline -- a line a few rows tall but a hundred pixels
       wide (common in the distance, where a single dash is viewed nearly edge-on) can pull a
       quadratic into a non-monotonic hook that has nothing to do with the real line shape.
       Bin the points into `bin_height`-row strips and take each strip's *median* x first, so
       the fit sees one representative point per row-strip instead of every noisy pixel.
    4. Fit `np.polyfit(bin_ys, bin_xs, fit_degree)` to those binned points. `fit_degree` drops
       to 1 (a straight line) whenever the data's vertical span is under
       `min_span_for_quadratic` -- a quadratic's curvature term isn't reliably constrained by a
       short baseline, and a short visible segment of a real lane line is close to straight
       anyway.
    5. If one side doesn't have enough points, reconstruct it by assuming the two lines are
       parallel: reuse the found side's exact curve shape (same quadratic/linear coefficients),
       shifted sideways by `DEFAULT_LANE_OFFSET_FRAC * frame_width` -- shifting only the
       constant term of the polynomial is exactly "the same curve, offset sideways by a fixed
       amount at every row", which is what "parallel" means here."""
    h, w = ll_conf.shape
    frame_center_x = w / 2.0
    y_top, y_bottom = int(h * y_top_frac), int(h * y_bottom_frac)
    max_offset = w * max_offset_frac

    ys, xs = np.where(ll_conf > threshold)
    in_band = (ys >= y_top) & (ys <= y_bottom) & (np.abs(xs - frame_center_x) <= max_offset)
    ys, xs = ys[in_band], xs[in_band]

    left_sel = xs < frame_center_x
    right_sel = ~left_sel

    def _fit(sel):
        if sel.sum() < min_points:
            return None
        sel_ys, sel_xs = ys[sel], xs[sel]
        y_min, y_max = int(sel_ys.min()), int(sel_ys.max())

        bin_ys, bin_xs = [], []
        for y0 in range(y_min, y_max + 1, bin_height):
            in_bin = (sel_ys >= y0) & (sel_ys < y0 + bin_height)
            if in_bin.any():
                bin_ys.append(y0 + bin_height / 2.0)
                bin_xs.append(float(np.median(sel_xs[in_bin])))
        if len(bin_ys) < min_bins:
            return None

        fit_degree = degree if (y_max - y_min) >= min_span_for_quadratic else 1
        fit_degree = min(fit_degree, len(bin_ys) - 1)
        coeffs = np.polyfit(bin_ys, bin_xs, fit_degree)
        return coeffs, (y_min, y_max)

    left_fit = _fit(left_sel)
    right_fit = _fit(right_sel)

    offset_px = DEFAULT_LANE_OFFSET_FRAC * w
    reconstructed = None
    if left_fit is None and right_fit is not None:
        coeffs, y_range = right_fit
        mirrored = coeffs.copy()
        mirrored[-1] -= offset_px
        left_fit = (mirrored, y_range)
        reconstructed = "left"
    elif right_fit is None and left_fit is not None:
        coeffs, y_range = left_fit
        mirrored = coeffs.copy()
        mirrored[-1] += offset_px
        right_fit = (mirrored, y_range)
        reconstructed = "right"

    return left_fit, right_fit, reconstructed


def draw_lane_curves(
    img_bgr: np.ndarray,
    left_fit,
    right_fit,
    color: tuple[int, int, int] = (0, 255, 255),
    thickness: int = 6,
) -> np.ndarray:
    """Draws the curves `fit_lane_curves` found (or reconstructed) onto `img_bgr` as a
    highlighted overlay -- bright/saturated by default so it stands out from the photographic
    road/car content underneath rather than blending into it. Each curve is only drawn across
    its own backing row range, never extrapolated past it."""
    out = img_bgr.copy()
    h, w = img_bgr.shape[:2]
    for fit in (left_fit, right_fit):
        if fit is None:
            continue
        coeffs, (y0, y1) = fit
        rows = np.arange(y0, y1 + 1)
        cols = np.polyval(coeffs, rows)
        valid = (cols >= 0) & (cols < w)
        pts = np.stack([cols[valid], rows[valid]], axis=1).astype(np.int32)
        if len(pts) >= 2:
            cv2.polylines(out, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    return out


def road_mask_with_lane_attention(
    img_bgr: np.ndarray,
    da_conf: np.ndarray,
    ll_conf: np.ndarray,
    vehicle_boxes: np.ndarray,
    road_threshold: float = 0.7,
    line_threshold: float = 0.7,
    line_color: tuple[int, int, int] = (0, 255, 255),
    line_thickness: int = 6,
):
    """Convenience wrapper: `full_road_mask` (road + lane lines + vehicles, blacking out
    everything else) with `fit_lane_curves`'s curves drawn on top as a highlighted overlay.
    Returns `(image, left_fit, right_fit, reconstructed)` -- the fits (and which side, if any,
    was parallel-reconstructed) are returned too since callers (e.g. the closed loop's debug
    overlay) may want to report them, not just draw them."""
    base_mask = full_road_mask(da_conf, ll_conf, vehicle_boxes, threshold=road_threshold)
    masked = apply_mask(img_bgr, base_mask)
    left_fit, right_fit, reconstructed = fit_lane_curves(ll_conf, threshold=line_threshold)
    highlighted = draw_lane_curves(masked, left_fit, right_fit, color=line_color, thickness=line_thickness)
    return highlighted, left_fit, right_fit, reconstructed


def apply_mask(img_bgr: np.ndarray, mask: np.ndarray, fill: int = 0) -> np.ndarray:
    """Blacks out (or fills with `fill`) every pixel where `mask` is 0."""
    out = img_bgr.copy()
    out[mask == 0] = fill
    return out
