"""Unit transforms between comma2k19 telemetry and BeamNG's vehicle.control() units.

comma2k19's processed_log exposes:
  - car_speed: m/s
  - steering_angle: degrees, steering *wheel* angle (not road-wheel angle)
  - IMU acceleration: m/s^2, vehicle frame, forward axis is the longitudinal component
    (no separate gas/brake pedal channel exists in the dataset)

BeamNG's vehicle.control() / electrics sensor expose:
  - wheelSpeed: m/s
  - steering: -1.0 (full left) .. 1.0 (full right)
  - throttle: 0.0 .. 1.0
  - brake: 0.0 .. 1.0

ETK800_STEERING_RATIO, MAX_ETK800_LOCK_DEG, ACCEL_SCALE and BRAKE_SCALE are approximations and
should be recalibrated against the actual BeamNG vehicle (e.g. etk800) if used for closed-loop
control.
"""

from __future__ import annotations

import numpy as np

COMMA_STEERING_RATIO = 10.9    # 2016 Honda Civic base steering ratio
ETK800_STEERING_RATIO = 14.5   # Approximate ETK 800 steering ratio
MAX_ETK800_LOCK_DEG = 450.0    # ETK 800 lock-to-lock (900 deg) / 2
ACCEL_SCALE = 3.0
BRAKE_SCALE = 8.0


def comma_speed_to_beamng(car_speed_mps):
    """car_speed (m/s) -> wheelSpeed (m/s). Units already match."""
    return car_speed_mps


def comma_steering_to_beamng(steering_angle_deg, match_tire_path: bool = True):
    """comma2k19 steering-wheel angle (deg, positive = left) -> BeamNG steering (-1..1).

    Inverts sign (comma left=+ -> BeamNG left=-) and, when match_tire_path, rescales by the
    ETK800/comma steering-ratio difference so the ETK800's road wheels follow the same
    trajectory as the dataset's source vehicle, then clamps to the ETK800's lock-to-lock range
    and normalizes to BeamNG's [-1, 1] steering scale.
    """
    beamng_target_deg = -np.asarray(steering_angle_deg, dtype=np.float64)

    if match_tire_path:
        beamng_target_deg = (beamng_target_deg / COMMA_STEERING_RATIO) * ETK800_STEERING_RATIO

    clamped_deg = np.clip(beamng_target_deg, -MAX_ETK800_LOCK_DEG, MAX_ETK800_LOCK_DEG)
    return clamped_deg / MAX_ETK800_LOCK_DEG


def comma_accel_to_beamng_controls(accel_long_mps2, accel_scale: float = ACCEL_SCALE, brake_scale: float = BRAKE_SCALE):
    """Signed longitudinal acceleration (m/s^2) -> (throttle, brake), each 0..1."""
    accel = np.asarray(accel_long_mps2, dtype=np.float32)
    throttle = np.clip(np.where(accel >= 0, accel / accel_scale, 0.0), 0.0, 1.0)
    brake = np.clip(np.where(accel < 0, -accel / brake_scale, 0.0), 0.0, 1.0)
    return throttle, brake


def beamng_controls_to_accel(throttle, brake, accel_scale: float = ACCEL_SCALE, brake_scale: float = BRAKE_SCALE):
    """Inverse of comma_accel_to_beamng_controls: (throttle, brake) -> signed accel (m/s^2)."""
    throttle = np.asarray(throttle, dtype=np.float32)
    brake = np.asarray(brake, dtype=np.float32)
    return throttle * accel_scale - brake * brake_scale


def beamng_steering_to_deg(steering_norm, match_tire_path: bool = True):
    """Inverse of comma_steering_to_beamng: BeamNG steering (-1..1) -> comma2k19-equivalent
    steering-wheel degrees (positive = left)."""
    beamng_target_deg = np.asarray(steering_norm, dtype=np.float64) * MAX_ETK800_LOCK_DEG

    if match_tire_path:
        beamng_target_deg = (beamng_target_deg / ETK800_STEERING_RATIO) * COMMA_STEERING_RATIO

    return -beamng_target_deg
