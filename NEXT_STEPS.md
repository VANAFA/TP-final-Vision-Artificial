# Next Step: Closed-Loop LKA + ACC on `yolop_beamng.py`

Context: BeamNG.tech (needed for the in-sim Camera sensor and true semantic
ground truth) is not available on this BeamNG.drive install — confirmed
directly via the simulator (`"This feature requires a BeamNG.tech
license."`). `yolop_beamng.py` already gets real frames (screen capture of
the driver-view window), a real lane mask, and real detected vehicle boxes
via the YOLOP ONNX model. What's missing is turning that perception into
steering/throttle/brake commands.

## LKA (lateral)

Skip the full bird's-eye IPM homography from `pipeline.md` for v1 — it's the
most calibration-heavy, error-prone part of the plan and not needed to get a
first working loop:

1. Take a horizontal strip of `ll_seg_mask` near the bottom of the frame
   (closest to the car).
2. Compute the pixel centroid of the lane-mask pixels in that strip.
3. `e_y = centroid_x - frame_width / 2` (normalize by frame width).
4. PID: `steering = Kp*e_y + Ki*integral(e_y) + Kd*d(e_y)/dt`, clipped to
   `[-1, 1]`.

Add the proper IPM/homography later only if this proxy isn't accurate
enough in tighter curves.

## ACC (longitudinal)

1. From `boxes` (already returned by `YOLOPDetector.process_frame`), filter
   to the box most centered in front of the car (x near frame center) and
   closest (largest bbox height/width).
2. Estimate distance with the pinhole formula already in `pipeline.md`:
   `d = f * W_real / w_pixel` (f from camera FOV + resolution, W_real ≈ 1.8m
   average vehicle width).
3. Target distance from a time-gap: `d_target = v_ego * T_gap` (T_gap ≈ 2s).
4. PID on `e_d = d - d_target` → throttle/brake.

## Known constraint: loop rate

Frame source is screen capture (mss), not a direct sensor — slower and
noisier than the original plan assumed. CPU YOLOP inference alone measured
~76ms/frame in the BDD100K reproduction run; screen-grab adds more on top.
Expect ~8-12 FPS for the full loop. Keep AI/manual speed conservative until
the loop is confirmed stable — a stale frame at high speed is what causes
oscillation or off-road excursions, not a bug in the control math itself.

## Order of operations

1. Implement LKA alone, tune PID gains with the car held to a constant low
   speed, confirm no zigzag.
2. Implement ACC alone on a straight road behind another car, confirm
   smooth following without harsh braking.
3. Combine both, re-tune if needed.
4. Only then move to the plan's later step (removing telemetry inputs to
   force vision-only estimation) for the paper's sensitivity analysis.
