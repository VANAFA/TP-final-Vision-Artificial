"""
Classical Hough-based lane detection + PID lane-keeping controller.

Pipeline per frame (raw BGR from BeamNG camera):
  White/yellow color mask → Canny edges → trapezoidal ROI →
  Hough lines → weighted-average left/right lane → lane-center error →
  PID → BeamNG steer command

The color mask step is the critical difference from naive grayscale Canny:
  - Lane markings are bright WHITE (HLS lightness > 170, saturation < 80)
  - Concrete median walls are GRAY   (HLS lightness 130-160, saturation < 20)
  - Guardrails are GRAY-SILVER        (HLS lightness 120-150, saturation < 30)
Filtering for high-lightness pixels before Canny means the walls and
guardrails produce no edges at all, so Hough only ever sees lane markings.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Color-based preprocessing (the primary fix for wall/guardrail false detections)
# ---------------------------------------------------------------------------

def _white_yellow_mask(img_bgr):
    """
    Binary mask keeping only white and yellow lane-marking pixels.

    Works in HLS space (hue, lightness, saturation):
      White  markings: any hue, L > 170, S < 80   (bright, unsaturated)
      Yellow markings: H 10-35,  L > 100, S > 100  (for yellow centre lines)

    Concrete walls (L ≈ 130-160, S < 20) and metal guardrails (L ≈ 120-150,
    S < 30) both fall below the L > 170 threshold, so they produce zero mask
    pixels and therefore zero Canny edges -- the critical distinction.

    A small dilation connects the gaps between dashed line segments so Hough
    can fit a single long line through them.
    """
    hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
    white  = cv2.inRange(hls, (0,   140,  0), (180, 255, 100))
    yellow = cv2.inRange(hls, (10,  100, 100), (35,  255, 255))
    mask = cv2.bitwise_or(white, yellow)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.dilate(mask, kernel, iterations=2)


# ---------------------------------------------------------------------------
# ROI
# ---------------------------------------------------------------------------

def _roi_mask(edges, row_top=0.60, row_bot=0.90):
    """
    Trapezoidal mask focused on the road surface where lane markings appear.

    Bottom corners are kept well away from the image edges (0.18 / 0.82)
    to avoid the very base of the concrete wall on the left and the guardrail
    on the right, which are taller than the asphalt and still present even
    after color filtering.
    Top corners (0.44 / 0.56) are tight around the vanishing point.
    """
    h, w = edges.shape[:2]
    mask = np.zeros_like(edges)
    vertices = np.array([[
        [int(w * 0.0),  int(h * row_bot)],
        [int(w * 0.40), int(h * row_top)],
        [int(w * 0.60), int(h * row_top)],
        [int(w * 1.0),  int(h * row_bot)],
    ]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(edges, mask)


def _hough(roi):
    return cv2.HoughLinesP(
        roi, rho=1, theta=np.pi / 180,
        threshold=15, minLineLength=15, maxLineGap=200,
    )


# ---------------------------------------------------------------------------
# Lane averaging
# ---------------------------------------------------------------------------

def _avg_lane(lines, img_width):
    """
    Classify Hough segments as left/right lane by slope AND x-position.

    After color filtering, nearly all surviving edges ARE lane markings --
    but we still gate on slope and x-side as a secondary defence:
      Left  lane: slope < -0.3, segment mid-x in left  65% of frame
      Right lane: slope >  0.3, segment mid-x in right 65% of frame
    (Overlap in the 35-65% band lets both lines converge to the vanishing
    point without being rejected.)
    """
    left_si, left_w = [], []
    right_si, right_w = [], []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.3:
                continue
            x_mid = (x1 + x2) / 2.0
            intercept = y1 - slope * x1
            length = np.hypot(x2 - x1, y2 - y1)
            if slope < 0 and x_mid < img_width * 0.55:
                left_si.append((slope, intercept))
                left_w.append(length)
            elif slope > 0 and x_mid > img_width * 0.45:
                right_si.append((slope, intercept))
                right_w.append(length)
    left  = np.dot(left_w,  left_si)  / np.sum(left_w)  if left_w  else None
    right = np.dot(right_w, right_si) / np.sum(right_w) if right_w else None
    return left, right


def _line_pts(y_bot, y_top, lane):
    if lane is None:
        return None
    slope, intercept = lane
    if abs(slope) < 1e-6:
        return None
    x_bot = int((y_bot - intercept) / slope)
    x_top = int((y_top - intercept) / slope)
    return (x_bot, int(y_bot)), (x_top, int(y_top))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_lane_lines(img_bgr, row_top=0.50, row_bot=0.75):
    """
    Detect left and right lane lines from a raw BeamNG BGR camera frame.

    Returns
    -------
    left_pts, right_pts : ((x_bot, y_bot), (x_top, y_top)) or None
    overlay : np.ndarray
        img_bgr with detected lines, ROI boundary, and color-mask preview
        (small inset top-right showing which pixels passed the white filter).
    """
    h, w = img_bgr.shape[:2]

    # 1. Color filter → only bright-white / yellow pixels survive
    color_mask = _white_yellow_mask(img_bgr)

    # 2. Apply mask to grayscale, then Canny
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bitwise_and(gray, gray, mask=color_mask)
    blur = cv2.GaussianBlur(filtered, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)

    # 3. Trapezoidal ROI
    roi = _roi_mask(edges, row_top=row_top, row_bot=row_bot)

    # 4. Hough
    lines = _hough(roi)

    # Build overlay
    overlay = img_bgr.copy()

    # Small color-mask preview inset (top-right corner) so you can see
    # whether the white filter is catching the lane markings.
    preview_h, preview_w = h // 4, w // 4
    mask_preview = cv2.resize(color_mask, (preview_w, preview_h))
    mask_preview_bgr = cv2.cvtColor(mask_preview, cv2.COLOR_GRAY2BGR)
    overlay[4:4 + preview_h, w - preview_w - 4:w - 4] = mask_preview_bgr
    cv2.rectangle(overlay, (w - preview_w - 4, 4), (w - 4, 4 + preview_h), (200, 200, 0), 1)

    # ROI boundary
    roi_pts = np.array([[
        [int(w * 0.0), int(h * row_bot)],  # bottom-left  → lower 0.18 to go more left
        [int(w * 0.40), int(h * row_top)],  # top-left     → lower 0.44 to go more left
        [int(w * 0.60), int(h * row_top)],  # top-right    → raise 0.56 to go more right
        [int(w * 1.0), int(h * row_bot)],  # bottom-right → raise 0.82 to go more right
    ]], dtype=np.int32)
    cv2.polylines(overlay, roi_pts, isClosed=True, color=(255, 200, 0), thickness=1)

    if lines is None:
        return None, None, overlay

    left_lane, right_lane = _avg_lane(lines, img_width=w)
    y_bot = h * row_bot
    y_top = h * row_top
    left_pts  = _line_pts(y_bot, y_top, left_lane)
    right_pts = _line_pts(y_bot, y_top, right_lane)

    # Reject if lines cross at bottom OR top (inverted detection)
    if left_pts is not None and right_pts is not None:
        if left_pts[0][0] > right_pts[0][0] or left_pts[1][0] > right_pts[1][0]:
            left_pts = right_pts = None

    for pts, color in [(left_pts, (0, 255, 0)), (right_pts, (0, 0, 255))]:
        if pts is not None:
            cv2.line(overlay, pts[0], pts[1], color, 4)

    return left_pts, right_pts, overlay


def lane_center_error(left_pts, right_pts, img_width, last_error=0.0):
    """
    Lateral offset of the car from the lane center, normalised to [-1, +1].
    Positive = car is right of centre → PID steers left (negative command).
    Returns last_error when no lines are detected (holds last known state).
    """
    cx = img_width / 2.0
    half_lane_px = img_width * 0.25

    if left_pts is not None and right_pts is not None:
        lane_center = (left_pts[0][0] + right_pts[0][0]) / 2.0
    elif left_pts is not None:
        lane_center = left_pts[0][0] + half_lane_px
    elif right_pts is not None:
        lane_center = right_pts[0][0] - half_lane_px
    else:
        return last_error

    return float(np.clip((cx - lane_center) / (img_width / 2.0), -1.0, 1.0))


class PIDLaneKeeper:
    """
    PID controller: lane centre error → BeamNG steering command.
      error > 0  (car right of centre) → negative steer (turn left)
      error < 0  (car left  of centre) → positive steer (turn right)
    """

    def __init__(self, kp=0.4, ki=0.002, kd=0.08, max_steer=0.5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_steer = max_steer
        self._integral = 0.0
        self._prev_error = None

    def update(self, error):
        self._integral = float(np.clip(self._integral + error, -1.0, 1.0))
        d = (error - self._prev_error) if self._prev_error is not None else 0.0
        self._prev_error = error
        steer = -(self.kp * error + self.ki * self._integral + self.kd * d)
        return float(np.clip(steer, -self.max_steer, self.max_steer))

    def reset(self):
        self._integral = 0.0
        self._prev_error = None
