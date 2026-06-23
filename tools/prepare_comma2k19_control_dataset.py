"""Build a (frame, current telemetry) -> (future telemetry) manifest from raw comma2k19 segments.

Expects segments downloaded by tools/download_comma2k19.py under a raw dir with the
standard comma2k19 layout:

    raw_dir/Chunk_n/route_id/segment_number/
        preview.png
        raw_log.bz2
        video.hevc
        processed_log/...   (numpy arrays for CAN / IMU signals)

The exact processed_log sub-layout has varied across comma2k19 mirrors (e.g.
`CAN/car_speed` vs `CAN/speed`, `IMU/acceleration` vs `IMU/accelerometer`), so
signal lookup below is fuzzy: it searches recursively under processed_log for any
.npy file whose path contains all of the given keyword tokens. Run this script
with --inspect <segment_dir> first to see what's actually on disk for your
download and adjust SIGNAL_TOKENS below if the fuzzy match doesn't find anything.

Output: one row per extracted frame, written to a CSV manifest with columns:
    segment_id, image_path,
    throttle_t, brake_t, steer_t, speed_t,
    throttle_target, brake_target, steer_target,
    split
"""

from __future__ import annotations

import argparse
import csv
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from comma_beamng_transforms import comma_accel_to_beamng_controls, comma_steering_to_beamng, comma_speed_to_beamng

SIGNAL_TOKENS = {
    "speed": ["can", "speed"],
    "steering_angle": ["can", "steer"],
    "accel": ["imu", "accel"],
}


def find_segments(raw_dir: Path):
    return sorted(p.parent for p in raw_dir.rglob("video.hevc"))


NON_NUMPY_SUFFIXES = {".png", ".bz2", ".hevc"}


def _iter_log_files(processed_log_dir: Path):
    """comma2k19 stores each signal as a folder with extensionless `t` and `value`
    files -- still real .npy binaries (np.load sniffs the header, extension or not),
    just not named *.npy. Glob everything, callers filter by filename/path tokens."""
    yield from (p for p in processed_log_dir.rglob("*") if p.is_file())


def inspect_segment(segment_dir: Path):
    print(f"Segment: {segment_dir}")
    for p in sorted(segment_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(segment_dir)
            if p.suffix.lower() in NON_NUMPY_SUFFIXES:
                print(f"  {rel}")
            else:
                try:
                    arr = np.load(p, allow_pickle=True)
                    print(f"  {rel}  shape={arr.shape} dtype={arr.dtype}")
                except Exception as exc:  # noqa: BLE001
                    print(f"  {rel}  <failed to load: {exc}>")


def find_signal(processed_log_dir: Path, tokens: list[str]) -> np.ndarray | None:
    best = None
    for p in _iter_log_files(processed_log_dir):
        if p.name != "value":
            continue
        joined = "/".join(s.lower() for s in p.parts)
        if all(t in joined for t in tokens):
            best = p
            break
    if best is None:
        return None
    arr = np.load(best, allow_pickle=True)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]  # e.g. CAN/speed/value is stored as a (N, 1) column vector
    return arr


def find_signal_timestamps(processed_log_dir: Path, tokens: list[str]) -> np.ndarray | None:
    """comma2k19 signals are stored as a folder with a `t` file and a `value` file."""
    for p in _iter_log_files(processed_log_dir):
        if p.name != "t":
            continue
        joined = "/".join(s.lower() for s in p.parts)
        if all(t in joined for t in tokens):
            return np.load(p, allow_pickle=True)
    return None


def extract_frames(segment_dir: Path, frames_out_dir: Path, ffmpeg_exe: str) -> list[Path]:
    frames_out_dir.mkdir(parents=True, exist_ok=True)
    video_path = segment_dir / "video.hevc"
    pattern = frames_out_dir / "frame_%06d.jpg"
    cmd = [ffmpeg_exe, "-y", "-i", str(video_path), "-vsync", "0", "-qmin", "1", "-qmax", "5", str(pattern)]
    subprocess.run(cmd, check=True, capture_output=True)
    return sorted(frames_out_dir.glob("frame_*.jpg"))


def nearest_value(t_query: np.ndarray, t_signal: np.ndarray, v_signal: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(t_signal, t_query)
    idx = np.clip(idx, 1, len(t_signal) - 1)
    left = t_signal[idx - 1]
    right = t_signal[idx]
    idx = np.where(np.abs(t_query - left) <= np.abs(t_query - right), idx - 1, idx)
    return v_signal[idx]


def build_segment_rows(segment_dir: Path, frames: list[Path], frames_rel_to: Path, horizon_s: float):
    processed_log_dir = segment_dir / "processed_log"

    speed_t = find_signal_timestamps(processed_log_dir, SIGNAL_TOKENS["speed"])
    speed_v = find_signal(processed_log_dir, SIGNAL_TOKENS["speed"])
    steer_t = find_signal_timestamps(processed_log_dir, SIGNAL_TOKENS["steering_angle"])
    steer_v = find_signal(processed_log_dir, SIGNAL_TOKENS["steering_angle"])
    accel_t = find_signal_timestamps(processed_log_dir, SIGNAL_TOKENS["accel"])
    accel_v = find_signal(processed_log_dir, SIGNAL_TOKENS["accel"])

    if speed_v is None or steer_v is None or accel_v is None:
        print(f"  skipping {segment_dir}: could not locate one or more signals "
              f"(speed={speed_v is not None}, steer={steer_v is not None}, accel={accel_v is not None}); "
              f"run with --inspect to see actual processed_log layout", file=sys.stderr)
        return []

    if accel_v.ndim > 1:
        accel_v = accel_v[:, 0]  # forward/longitudinal axis

    frame_times_path = segment_dir / "global_pose" / "frame_times"
    frame_times = np.load(frame_times_path, allow_pickle=True) if frame_times_path.exists() else None
    if frame_times is None or len(frame_times) != len(frames):
        t0, t1 = speed_t[0], speed_t[-1]
        frame_times = np.linspace(t0, t1, num=len(frames))

    rows = []
    segment_id = "_".join(segment_dir.parts[-3:])
    for frame_path, t in zip(frames, frame_times):
        t_future = t + horizon_s
        if t_future > speed_t[-1]:
            continue

        speed_now = nearest_value(np.array([t]), speed_t, speed_v)[0]
        steer_now = nearest_value(np.array([t]), steer_t, steer_v)[0]
        accel_now = nearest_value(np.array([t]), accel_t, accel_v)[0]
        accel_future = nearest_value(np.array([t_future]), accel_t, accel_v)[0]
        steer_future = nearest_value(np.array([t_future]), steer_t, steer_v)[0]

        throttle_now, brake_now = comma_accel_to_beamng_controls(accel_now)
        throttle_future, brake_future = comma_accel_to_beamng_controls(accel_future)

        rows.append({
            "segment_id": segment_id,
            "image_path": str(frame_path.relative_to(frames_rel_to)),
            "throttle_t": float(throttle_now),
            "brake_t": float(brake_now),
            "steer_t": float(comma_steering_to_beamng(steer_now)),
            "speed_t": float(comma_speed_to_beamng(speed_now)),
            "throttle_target": float(throttle_future),
            "brake_target": float(brake_future),
            "steer_target": float(comma_steering_to_beamng(steer_future)),
        })
    return rows


def assign_splits(segment_ids: list[str], train_frac: float, dev_frac: float, seed: int = 0) -> dict[str, str]:
    unique = sorted(set(segment_ids))
    rng = random.Random(seed)
    rng.shuffle(unique)
    n_train = round(len(unique) * train_frac)
    n_dev = round(len(unique) * dev_frac)
    split_of = {}
    for i, sid in enumerate(unique):
        if i < n_train:
            split_of[sid] = "train"
        elif i < n_train + n_dev:
            split_of[sid] = "dev"
        else:
            split_of[sid] = "test"
    return split_of


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/comma2k19/raw"))
    parser.add_argument("--frames-dir", type=Path, default=Path("data/comma2k19/frames"))
    parser.add_argument("--manifest", type=Path, default=Path("data/comma2k19/manifest.csv"))
    parser.add_argument("--horizon-seconds", type=float, default=1.0)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--dev-frac", type=float, default=0.15)
    parser.add_argument("--inspect", type=Path, default=None, help="Print processed_log layout for one segment dir and exit")
    args = parser.parse_args()

    if args.inspect:
        inspect_segment(args.inspect)
        return

    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    segments = find_segments(args.raw_dir)
    if not segments:
        print(f"No segments (video.hevc) found under {args.raw_dir}", file=sys.stderr)
        sys.exit(1)

    all_rows = []
    for segment_dir in segments:
        segment_id = "_".join(segment_dir.parts[-3:])
        print(f"Processing {segment_id}...")
        frames_out_dir = args.frames_dir / segment_id
        frames = extract_frames(segment_dir, frames_out_dir, ffmpeg_exe)
        rows = build_segment_rows(segment_dir, frames, args.frames_dir, args.horizon_seconds)
        all_rows.extend(rows)
        print(f"  {len(rows)} usable frames")

    if not all_rows:
        print("No rows produced across any segment.", file=sys.stderr)
        sys.exit(1)

    split_of = assign_splits([r["segment_id"] for r in all_rows], args.train_frac, args.dev_frac)
    for r in all_rows:
        r["split"] = split_of[r["segment_id"]]

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(all_rows[0].keys())
    with open(args.manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows from {len(segments)} segments to {args.manifest}")
    for split in ("train", "dev", "test"):
        n = sum(1 for r in all_rows if r["split"] == split)
        print(f"  {split}: {n} rows")


if __name__ == "__main__":
    main()
