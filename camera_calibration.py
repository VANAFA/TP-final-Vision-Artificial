"""Camera-geometry alignment between comma2k19 (training domain) and the BeamNG.drive
camera sensor (inference domain), plus a steering-bias compensator for the road-camber
offset baked into comma2k19's human-driving steering targets.

Why this exists
----------------
`tools/prepare_comma2k19_control_dataset.py` extracts raw frames straight out of
`video.hevc` (no openpilot "road frame" dewarp / pitch-yaw zeroing is applied), and
`control_finetune.py:preprocess_image_bgr` just letterbox-resizes those raw frames into
the 640x640 network input -- there is no ROI crop, no IPM, no calibration matrix anywhere
in this repo's training pipeline. So `YOLOPControlNet` was trained on the comma EON's raw
pinhole geometry. For BeamNG frames to land in the same input distribution, the BeamNG
camera sensor has to reproduce that pinhole geometry (FOV, height, pitch) itself -- nothing
downstream will correct a mismatch.

One thing that *is* already correct: BeamNG's 640x480 capture (aspect 1.333) and the EON's
native 1164x874 (aspect 1.332) are close enough that `resize_unscale`'s letterbox padding
works out almost identically in both domains (~80px gray bars top/bottom either way). The
gap is the FOV: this repo's camera rigs use 70 deg vertical FOV; the EON's real vertical
FOV is ~51 deg (see COMMA_VFOV_DEG below). A wider FOV at the same mount height squeezes
more world into the same pixels -- it inflates the hood, and changes how fast lane lines
converge toward the vanishing point. That's the "perspective warp" mismatch.
"""

from __future__ import annotations

import numpy as np
import cv2

# ---------------------------------------------------------------------------
# 1. Comma EON / OnePlus 3T camera intrinsics (openpilot's published calibration:
#    common/transformations/camera.py -- fx == fy == 910.0 at 1164x874)
# ---------------------------------------------------------------------------
COMMA_WIDTH = 1164
COMMA_HEIGHT = 874
COMMA_FOCAL_PX = 910.0
COMMA_CX = COMMA_WIDTH / 2.0    # 582.0
COMMA_CY = COMMA_HEIGHT / 2.0   # 437.0
COMMA_ASPECT = COMMA_WIDTH / COMMA_HEIGHT  # 1.332

# Typical EON mount height in the comma2k19 fleet vehicles (mirror/windshield mount).
COMMA_MOUNT_HEIGHT_M = 1.22


def fov_deg(focal_px: float, extent_px: float) -> float:
    """Full field of view (degrees) subtended by `extent_px` pixels at focal length
    `focal_px` for a pinhole camera with the principal point centered on that extent."""
    return float(np.degrees(2.0 * np.arctan((extent_px / 2.0) / focal_px)))


COMMA_HFOV_DEG = fov_deg(COMMA_FOCAL_PX, COMMA_WIDTH)    # ~65.2 deg
COMMA_VFOV_DEG = fov_deg(COMMA_FOCAL_PX, COMMA_HEIGHT)   # ~51.2 deg

# ---------------------------------------------------------------------------
# 2. Recommended BeamNG Camera sensor config
# ---------------------------------------------------------------------------
BEAMNG_RESOLUTION = (640, 480)  # aspect 1.333, already ~matches COMMA_ASPECT (1.332)

# beamngpy's Camera sensor defaults near_far_planes to (0.05, 100.0) -- a 100m far plane
# clips lane lines/vehicles well before a highway sightline would, even before BeamNG's own
# terrain/object LOD streaming has a chance to cull anything. comma2k19 is highway driving
# with long unobstructed sightlines, so 100m is itself a domain-gap source independent of
# FOV. 1000m comfortably covers any straightaway without materially hurting depth precision
# (the depth buffer only needs ~3-4 useful bits more than 100m would).
BEAMNG_FAR_PLANE_M = 1000.0

RECOMMENDED_BEAMNG_CAMERA = dict(
    resolution=BEAMNG_RESOLUTION,
    field_of_view_y=round(COMMA_VFOV_DEG, 1),  # ~51.2 deg -- was 70 in both rigs in this repo
    near_far_planes=(0.05, BEAMNG_FAR_PLANE_M),  # was (0.05, 100.0) -- see note above
    pos=(0.0, -0.5, COMMA_MOUNT_HEIGHT_M),     # windshield/mirror mount, not hood-mounted
    dir=(0, 1, 0),                              # level (0 pitch) -- verify sign in-sim, see note
    up=(0, 0, 1),
)
# NOTE on `dir`/`pos` sign: yolop_beamng.py uses pos=(0, 1.7, 1.2), dir=(0, 1, 0); the
# closed-loop notebook uses pos=(0.1, -0.6, 1.2), dir=(0, -1, 0) -- opposite signs for what's
# meant to be the same "forward, level" mount. Only one of those is actually pointed down
# the road for etk800's local axes. This module can't verify that without BeamNG running --
# confirm by polling one frame and checking the hood/road appear the right way up before
# trusting `RECOMMENDED_BEAMNG_CAMERA["dir"]` as-is.


# ---------------------------------------------------------------------------
# 3. Pitch / horizon-row calibration
# ---------------------------------------------------------------------------
def horizon_row_fraction(pitch_deg: float, vfov_deg: float) -> float:
    """Fractional row (0=top, 1=bottom) where the true horizon lands for a camera with a
    centered principal point, given downward pitch `pitch_deg` (positive = tilted down)
    and vertical FOV `vfov_deg`. Resolution-independent."""
    p = np.radians(pitch_deg)
    half_v = np.radians(vfov_deg / 2.0)
    return float(0.5 + 0.5 * np.tan(p) / np.tan(half_v))


def required_pitch_deg(target_row_fraction: float, vfov_deg: float) -> float:
    """Inverse of horizon_row_fraction: pitch needed to put the horizon at a given row
    fraction for a chosen vertical FOV."""
    half_v = np.radians(vfov_deg / 2.0)
    return float(np.degrees(np.arctan((2.0 * target_row_fraction - 1.0) * np.tan(half_v))))


def estimate_horizon_row_fraction(img_bgr: np.ndarray, search_band=(0.25, 0.65)) -> float:
    """Best-effort sky/road boundary estimate: the row within `search_band` (fraction of
    image height) with the largest mean horizontal-edge response, after blurring vertically
    to suppress lane-marking edges (which are mostly vertical, not horizontal).

    This is a heuristic for comparing the *average* horizon row across several straight-road
    frames from each domain -- don't trust it on a single frame, and don't run it on a frame
    with a hill, overpass, or curve in view.
    """
    h = img_bgr.shape[0]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (1, 9), 0)
    grad = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5))
    row_energy = grad.mean(axis=1)
    lo, hi = int(search_band[0] * h), int(search_band[1] * h)
    row = lo + int(np.argmax(row_energy[lo:hi]))
    return row / h


# ---------------------------------------------------------------------------
# 4. Post-capture re-pitch (homography safety net / dynamic correction)
# ---------------------------------------------------------------------------
def camera_intrinsics(width: int, height: int, vfov_deg: float) -> np.ndarray:
    f = (height / 2.0) / np.tan(np.radians(vfov_deg / 2.0))
    return np.array(
        [[f, 0.0, width / 2.0],
         [0.0, f, height / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def repitch_image(img_bgr: np.ndarray, vfov_deg: float, delta_pitch_deg: float) -> np.ndarray:
    """Re-render `img_bgr` as if the camera had an extra `delta_pitch_deg` of downward tilt,
    via the homography of a pure-rotation pinhole camera: H = K @ R_x(theta) @ inv(K).

    Use this as:
      - a calibration safety net, if BeamNG's editor-level pitch control isn't fine enough
        to hit `required_pitch_deg(...)` exactly;
      - a per-frame dynamic correction, if you feed in the vehicle's *current* pitch
        deviation from its resting pose (e.g. from an IMU/orientation sensor) to counter
        chassis squat/dive under throttle and braking -- the BeamNG analogue of openpilot's
        dynamic road-frame pitch correction.

    Rotation about the camera's local X axis only changes vertical perspective (horizon
    row, vertical line convergence). It does not touch left/right symmetry, so it cannot
    fix or cause a left/right steering bias on its own -- pair it with
    SteeringBiasCompensator for that.
    """
    h, w = img_bgr.shape[:2]
    K = camera_intrinsics(w, h, vfov_deg)
    theta = np.radians(delta_pitch_deg)
    R = np.array(
        [[1.0, 0.0, 0.0],
         [0.0, np.cos(theta), -np.sin(theta)],
         [0.0, np.sin(theta), np.cos(theta)]],
    )
    H = K @ R @ np.linalg.inv(K)
    H /= H[2, 2]
    return cv2.warpPerspective(img_bgr, H, (w, h), borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------------------
# 5. Steering bias compensator
# ---------------------------------------------------------------------------
class SteeringBiasCompensator:
    """Removes a slowly-drifting constant offset from the model's raw steer output.

    Why this is needed even after camera/FOV alignment: comma2k19 is almost entirely
    US right-hand-traffic highway driving, where road camber/crown imposes a small,
    *constant* counter-steer that has nothing to do with lane position -- human drivers
    apply it unconsciously, and the network can learn it as a fixed additive term on
    `steer` that's independent of the visual input. That shows up downstream as a
    persistent left (or right) pull even on a dead-straight, well-aligned road, and no
    amount of camera-geometry fixing removes it -- it has to be subtracted explicitly.

    The offset is estimated as an EMA of the raw steer output, but only updated while the
    model's own output is already close to the current bias estimate (`gate_threshold`) --
    i.e. while it looks like it's holding a near-straight line, not mid-correction. That
    keeps genuine lane-keeping turns from contaminating the bias estimate.
    """

    def __init__(self, ema_alpha: float = 0.01, gate_threshold: float = 0.05):
        self.ema_alpha = ema_alpha
        self.gate_threshold = gate_threshold
        self.bias = 0.0

    def update(self, raw_steer: float) -> float:
        if abs(raw_steer - self.bias) < self.gate_threshold:
            self.bias += self.ema_alpha * (raw_steer - self.bias)
        return raw_steer - self.bias

    def reset(self) -> None:
        self.bias = 0.0
