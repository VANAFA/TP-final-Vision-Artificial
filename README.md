# Vision-Based Lane-Keeping in BeamNG — Final Project

End-to-end pipeline for autonomous lane-keeping in BeamNG.tech using a fine-tuned YOLOP backbone and a classical PID fallback. The work covers dataset download, model training, live closed-loop control, and a full set of paper figures.

---

## Overview

| Component | What it does |
|-----------|-------------|
| `1 Fine Tuning.ipynb` | Fine-tunes `YOLOPControlNet` on Comma2k19 — steer + brake heads on top of a frozen YOLOP backbone |
| `2 Closed Loop BeamNG Control.ipynb` | Live control loop inside BeamNG.tech: NN steering, PID lane-keeping assist, cruise control, traffic |
| `3 YOLOP Segmentation Sanity Check.ipynb` | Validates that the YOLOP backbone's drivable-area and lane-line outputs look correct on BeamNG frames |
| `4 Lane Mask Preview.ipynb` | Interactive preview of the HLS colour mask and trapezoidal ROI used by `lane_pid.py` |
| `5 Lane Detection Tuning.ipynb` | Parameter sweep for Hough-line detection thresholds and ROI geometry |
| `experiments/generate_figures.py` | Generates paper figures 1–3: steering comparison, Grad-CAM grid, variance bar chart |
| `experiments/generate_lane_figures.py` | Generates paper figures 4–5: lane detection overlay and colour-mask pipeline |

---

## Requirements

### System

- **OS**: Windows 10/11 (BeamNG.tech is Windows-only)
- **GPU**: NVIDIA GPU with CUDA 12.6+ recommended (trained on RTX 4060 Ti)
- **Python**: 3.10–3.12
- **BeamNG.tech**: v0.32 or later — **not** BeamNG.drive; the `.tech` branch exposes the Python API

### Python environment

```bash
python -m venv .venv
.venv\Scripts\activate

# Install PyTorch with CUDA first (adjust cu126 to match your driver):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Then the rest:
pip install -r requirements.txt
```

> **opencv-python vs opencv-python-headless**: `albumentations` pulls in `opencv-python-headless` as a dependency, which silently breaks `cv2.imshow` / `cv2.waitKey` needed by the live control loop. If you see a "function not implemented" error from cv2 GUI calls, run:
> ```bash
> pip uninstall -y opencv-python-headless
> pip install --force-reinstall --no-deps opencv-python
> ```

### YOLOP backbone weights

The pretrained YOLOP weights are not committed to this repo (too large). Download `End-to-end.pth` from the [official YOLOP release](https://github.com/hustvl/YOLOP) and place it at:

```
YOLOP/weights/End-to-end.pth
```

---

## BeamNG.tech setup

BeamNG.tech requires a research/academic license to run headlessly with the Python API.

### 1. License key

Place your `tech.key` file in the repository root:

```
TP-final-Vision-Artificial/
└── tech.key        ← goes here (gitignored)
```

Notebook 2 (cell 1) copies it to `%LOCALAPPDATA%/BeamNG.tech` automatically on first run.

### 2. Install and launch BeamNG.tech

Install BeamNG.tech to its default location (`C:\Program Files (x86)\Steam\steamapps\common\BeamNG.tech` or wherever the installer puts it). The notebook connects to it via `beamngpy` and launches it automatically — you do **not** need to open it manually before running the notebook.

### 3. Verify the connection

Run `tools/beamngpy_test.py` to confirm `beamngpy` can reach BeamNG before attempting the full control loop:

```bash
.venv\Scripts\python tools/beamngpy_test.py
```

---

## Training pipeline (Notebook 1)

### 1. Download Comma2k19

The full dataset is ~100 GB over BitTorrent (10 chunk zips, ~9–10 GB each). The automated download uses qBittorrent's Web UI:

1. Install [qBittorrent](https://www.qbittorrent.org/) and make sure it is **running**.
2. Enable its Web UI: **Tools → Options → Web UI → Enable Web User Interface** (default port 8080).
3. Set a username and password there.
4. In notebook 1, set `QBT_USERNAME` / `QBT_PASSWORD` to match before running the download cell.

The download script (`tools/download_comma2k19.py`) fetches one chunk at a time, extracts only the requested number of segments, and deletes the zip to reclaim disk space. Adjust `--max-segments` to control how much data you use.

### 2. Build the manifest

After downloading, run the manifest builder (handled automatically inside the notebook):

```bash
.venv\Scripts\python tools/prepare_comma2k19_control_dataset.py \
    --raw-dir data/comma2k19/raw \
    --frames-dir data/comma2k19/frames \
    --manifest data/comma2k19/manifest.csv
```

This extracts frames from each segment's `video.hevc`, aligns telemetry signals, applies the Comma2k19→BeamNG unit transforms from `comma_beamng_transforms.py`, and writes `manifest.csv` with a 70/15/15 train/dev/test split by segment (not by frame, to avoid leakage).

### 3. Train

Run **Notebook 1** top to bottom. Training fine-tunes only the small heads on top of the frozen backbone (~94k trainable vs ~7.9M frozen parameters). The best checkpoint is saved to `artifacts/yolop_control_temporal_best.pt`.

For unattended overnight runs, use papermill:

```bash
.venv\Scripts\papermill "1 Fine Tuning.ipynb" artifacts/training_run.ipynb --log-output
```

---

## Live control loop (Notebook 2)

Runs the trained model inside BeamNG in real time.

### What the loop does

- Grabs the BeamNG camera frame at ~10 Hz
- Runs `YOLOPControlNet` for a steer prediction and a brake prediction
- Runs `lane_pid.py`'s Hough-line PID for a classical steer signal
- Combines them (NN steer is used when the steer head is reliable; PID is the fallback)
- Sends steer + brake + cruise throttle back to the vehicle

### Keyboard controls (while the loop is running)

| Key | Action |
|-----|--------|
| `↑` / `↓` | Increase / decrease cruise speed |
| `p` / `n` / `m` | Park / Neutral / Manual gear |
| `t` | Respawn the AI traffic car behind ego |
| `c` | Save a screenshot |
| `q` | Stop the loop cleanly |

### Steps

1. Open **Notebook 2** in Jupyter or VS Code.
2. Run **cell 1** (license key copy — only needed once).
3. Run **cell 2** (load the trained checkpoint).
4. Run **cell 3** (cell 8 in the notebook — connects to BeamNG, spawns ego + traffic).
5. Run **cell 13** (the main loop). Press `q` to stop.
6. Run **cell 14** (cleanup — closes BeamNG).

> If `cv2.imshow` fails with "not implemented", reinstall `opencv-python` as described above, then restart the kernel.

---

## Paper figures

All figures are pre-generated in `paper_figures/` (PDF + PNG at 300 dpi). To regenerate:

```bash
# Figures 1–3: steering comparison, Grad-CAM, variance bar chart
.venv\Scripts\python experiments/generate_figures.py

# Figures 4–5: lane detection overlay, colour-mask pipeline
.venv\Scripts\python experiments/generate_lane_figures.py
```

Figure generation requires at least one debug log session recorded by notebook 2 (`debug_logs/` — gitignored). The scripts automatically find the best frames with detected lanes.

---

## Project structure

```
.
├── 1 Fine Tuning.ipynb               # Training pipeline
├── 2 Closed Loop BeamNG Control.ipynb # Live control loop
├── 3 YOLOP Segmentation Sanity Check.ipynb
├── 4 Lane Mask Preview.ipynb
├── 5 Lane Detection Tuning.ipynb
├── control_finetune.py               # YOLOPControlNet model + dataset
├── lane_pid.py                       # Classical Hough/PID lane keeping
├── lane_mask.py                      # YOLOP-based road/lane mask utilities
├── gradcam.py                        # Grad-CAM for steer/brake heads
├── yolop_beamng.py                   # BeamNG camera + preprocessing helpers
├── comma_beamng_transforms.py        # Comma2k19 ↔ BeamNG unit conversions
├── camera_calibration.py             # Intrinsic calibration utilities
├── YOLOP/                            # Official YOLOP repo (submodule/copy)
│   └── weights/End-to-end.pth        # ← download separately
├── artifacts/                        # Trained checkpoints (gitignored)
│   └── yolop_control_temporal_best.pt
├── data/comma2k19/                   # Dataset (gitignored — ~tens of GB)
│   ├── raw/                          # Downloaded chunk segments
│   ├── frames/                       # Extracted frames
│   └── manifest.csv                  # Training manifest
├── debug_logs/                       # Control loop event frames + CSV logs (gitignored)
├── paper_figures/                    # Generated figures for the paper (committed)
├── experiments/                      # Figure generation scripts
├── tools/                            # Data download and preprocessing scripts
├── requirements.txt
└── tech.key                          # BeamNG.tech license (gitignored — never commit)
```

---

## Key design decisions

**Why freeze the YOLOP backbone?** The backbone has 7.9M parameters trained on BDD100k. Fine-tuning it end-to-end on ~175k Comma2k19 frames would overfit to dashcam lighting and destroy its generalization to BeamNG's synthetic frames. Freezing it and training only the ~94k-parameter heads keeps the visual features general while adapting control.

**Why separate steer and brake heads?** An earlier single-head design (steer + brake + throttle sharing one MLP) let Comma2k19's "hard brake → wheels centered" emergency-stop frames corrupt the steer signal. Separate heads with a gradient stop on the brake path (`detach_brake_grad=True`) removes this leakage structurally.

**Why a PID fallback?** The steer head trained on ~17k comma2k19 frames showed near-constant output in BeamNG (Var(δ) ≈ 1.3×10⁻⁴, r(δ, eᵧ) ≈ −0.34). The Hough/PID approach achieves Var(δ) ≈ 5.4×10⁻⁴ and r(δ, eᵧ) ≈ −0.78. Training on the full dataset (all 10 chunks, ~1.75M frames) is the intended fix for the NN head.
