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

MAX_STEERING_DEG, ACCEL_SCALE and BRAKE_SCALE are approximations and should be
recalibrated against the actual BeamNG vehicle (e.g. etk800) if used for closed-loop control.
"""

from __future__ import annotations

import numpy as np

MAX_STEERING_DEG = 500.0
ACCEL_SCALE = 3.0
BRAKE_SCALE = 8.0


def comma_speed_to_beamng(car_speed_mps):
    """car_speed (m/s) -> wheelSpeed (m/s). Units already match."""
    return car_speed_mps


def comma_steering_to_beamng(steering_angle_deg, max_steering_deg: float = MAX_STEERING_DEG):
    """steering_angle (deg, steering wheel) -> steering (-1..1)."""
    return np.clip(np.asarray(steering_angle_deg) / max_steering_deg, -1.0, 1.0)


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


def beamng_steering_to_deg(steering_norm, max_steering_deg: float = MAX_STEERING_DEG):
    """Inverse of comma_steering_to_beamng: steering (-1..1) -> equivalent steering-wheel degrees."""
    return np.asarray(steering_norm, dtype=np.float32) * max_steering_deg
