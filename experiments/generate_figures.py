"""
Paper Figure Generation  —  YOLOPControlNet closed-loop driving study.

Experiment I  : Steering-command analysis (NN head collapse vs PID lane keeper)
Experiment II : Grad-CAM spatial attention of the steer / brake heads
"""

from __future__ import annotations

import glob
import os
import sys

import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import torch

# ── Make sure repo root is on the path ──────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from control_finetune import YOLOPControlNet, preprocess_image_bgr
from gradcam import YOLOPControlGradCAM, overlay_heatmap

# ── Output directory ─────────────────────────────────────────────────────────
OUT = os.path.join(ROOT, "paper_figures")
os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib / IEEE paper style
# ─────────────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update(
    {
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size":         9,
        "axes.titlesize":    9,
        "axes.labelsize":    9,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.fontsize":   8,
        "legend.framealpha": 0.92,
        "legend.edgecolor":  "#CCCCCC",
        "lines.linewidth":   1.0,
        "axes.linewidth":    0.6,
        "grid.linewidth":    0.4,
        "grid.alpha":        0.6,
        "figure.dpi":        150,     # screen preview
        "savefig.dpi":       300,     # print quality
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.03,
    }
)

# ── Palette (colorblind-safe, prints OK in greyscale with linestyle backup) ──
C_PID    = "#1B6CA8"   # blue    – hybrid PID controller
C_NN     = "#C1392B"   # red     – NN steer head (collapsed)
C_ERR_P  = "#2980B9"   # light blue fill – error right of centre
C_ERR_N  = "#E74C3C"   # light red fill  – error left of centre
C_ZERO   = "#AAAAAA"   # muted grey      – zero / reference line

IEEE_W   = 7.16        # inches – full double-column width
IEEE_H1  = 4.2         # fig 1 height
IEEE_H2  = 6.8         # fig 2 height


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def _load_window(n: int = 500) -> pd.DataFrame:
    """Collect n frames from clean PID-only runs at steady cruise speed."""
    csvs = sorted(glob.glob(os.path.join(ROOT, "debug_logs", "*", "loop_log.csv")))
    chunks: list[pd.DataFrame] = []
    for p in csvs:
        try:
            df = pd.read_csv(p)
            if (
                "steer_raw_nn" not in df.columns
                or not (df["steer_source"] == "pid").all()
            ):
                continue
            df = df[df["speed_kmh"].between(40, 80)].reset_index(drop=True)
            if len(df) >= 20:
                chunks.append(df)
        except Exception:
            pass
    combined = pd.concat(chunks, ignore_index=True)
    return combined.iloc[:n].copy()


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1  —  Steering-command comparison
# ═════════════════════════════════════════════════════════════════════════════

def figure1():
    df = _load_window(500)
    n  = len(df)
    t  = np.arange(n) / 10.0          # 10 Hz loop → seconds

    var_nn  = float(df["steer_raw_nn"].var())
    var_pid = float(df["pid_steer"].var())
    r_nn    = float(df["steer_raw_nn"].corr(df["lane_error"]))
    r_pid   = float(df["pid_steer"].corr(df["lane_error"]))
    flat_pct = (df["steer_raw_nn"].abs() < 0.005).mean() * 100

    # 2-row layout: top row = time series + 2 scatter insets; bottom = lane error
    fig = plt.figure(figsize=(IEEE_W, 5.4))
    gs  = fig.add_gridspec(
        2, 3,
        height_ratios=[1.6, 1.0],
        width_ratios=[3.2, 1.0, 1.0],
        hspace=0.12, wspace=0.08,
        left=0.09, right=0.99, top=0.97, bottom=0.10,
    )
    ax1  = fig.add_subplot(gs[0, 0])     # time series
    ax_n = fig.add_subplot(gs[0, 1])     # scatter: NN  vs eᵧ
    ax_p = fig.add_subplot(gs[0, 2])     # scatter: PID vs eᵧ
    ax2  = fig.add_subplot(gs[1, :])     # lane error (spans all cols)

    # ── Panel 1 : steering command time series ──────────────────────────────
    ax1.plot(t, df["pid_steer"],    color=C_PID, lw=1.1, ls="-",
             label="Hybrid PID",  zorder=3)
    ax1.plot(t, df["steer_raw_nn"], color=C_NN,  lw=1.0, ls="--",
             label="NN steer head", zorder=2, alpha=0.80)
    ax1.axhline(0, color=C_ZERO, lw=0.6, zorder=1)

    ax1.set_ylabel(r"Steering $\delta$  (normalised)")
    ax1.set_ylim(-0.095, 0.095)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax1.grid(True, axis="y")
    ax1.grid(True, axis="x", alpha=0.35)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_title("(a) Steering command over 500 frames", fontsize=8, loc="left", pad=3)
    leg = ax1.legend(loc="upper right", fontsize=7)
    for h in leg.legend_handles:
        h.set_linewidth(2.0)

    # ── Scatter panels : δ vs eᵧ (exposes correlation) ─────────────────────
    mk = dict(s=3, alpha=0.25, linewidths=0)
    ey_all = df["lane_error"].values

    ax_n.scatter(ey_all, df["steer_raw_nn"].values, color=C_NN,  **mk)
    ax_n.axhline(0, color=C_ZERO, lw=0.5)
    ax_n.axvline(0, color=C_ZERO, lw=0.5)
    ax_n.set_xlabel(r"$e_y$", fontsize=7)
    ax_n.set_ylabel(r"$\delta$", fontsize=7)
    ax_n.set_title(f"NN\n$r={r_nn:+.2f}$", fontsize=7.5, color=C_NN)
    ax_n.tick_params(labelsize=6)
    ax_n.spines[["top", "right"]].set_visible(False)
    ax_n.set_xlim(-0.8, 0.8); ax_n.set_ylim(-0.09, 0.09)

    ax_p.scatter(ey_all, df["pid_steer"].values,    color=C_PID, **mk)
    ax_p.axhline(0, color=C_ZERO, lw=0.5)
    ax_p.axvline(0, color=C_ZERO, lw=0.5)
    ax_p.set_xlabel(r"$e_y$", fontsize=7)
    ax_p.set_title(f"PID\n$r={r_pid:+.2f}$", fontsize=7.5, color=C_PID)
    ax_p.tick_params(labelsize=6)
    ax_p.set_yticklabels([])
    ax_p.spines[["top", "right", "left"]].set_visible(False)
    ax_p.set_xlim(-0.8, 0.8); ax_p.set_ylim(-0.09, 0.09)

    # ── Panel 3 : lateral error  ────────────────────────────────────────────
    ey = df["lane_error"].values
    ax2.fill_between(t, ey, 0, where=ey >= 0, color=C_ERR_P, alpha=0.30, lw=0,
                     label="Car right of centre")
    ax2.fill_between(t, ey, 0, where=ey <  0, color=C_ERR_N, alpha=0.30, lw=0,
                     label="Car left of centre")
    ax2.plot(t, ey, color="#333333", lw=0.75, zorder=3)
    ax2.axhline(0, color=C_ZERO, lw=0.6, zorder=1)

    ax2.set_xlabel("Time  (s)")
    ax2.set_ylabel(r"Lateral error $e_y$")
    ax2.set_ylim(-0.85, 0.85)
    ax2.grid(True, axis="y")
    ax2.grid(True, axis="x", alpha=0.35)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_title("(b) Measured lateral error (PID run)", fontsize=8, loc="left", pad=3)
    ax2.legend(loc="upper right", ncol=2, fontsize=7)

    # Shared x-axis ticks every 5 s
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(1))

    for ext in ("pdf", "png"):
        path = os.path.join(OUT, f"fig1_steering_comparison.{ext}")
        fig.savefig(path, format=ext)
        print(f"  [fig1] saved {path}")

    plt.close(fig)

    print(
        f"\n  === Experiment I stats (n={n} frames, cruise ~49 km/h) ===\n"
        f"  Var(delta_PID) = {var_pid:.2e}   r(PID, ey) = {r_pid:+.3f}\n"
        f"  Var(delta_NN)  = {var_nn:.2e}   r(NN,  ey) = {r_nn:+.3f}\n"
        f"  NN output within +/-0.005 (flat): {flat_pct:.0f}% of frames\n"
    )


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2  —  Grad-CAM spatial attention
# ═════════════════════════════════════════════════════════════════════════════

def figure2():
    device = "cpu"
    ckpt_path = os.path.join(ROOT, "artifacts", "yolop_control_head_masked_best.pt")

    model = YOLOPControlNet(device=device).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.numeric_branch.load_state_dict(ckpt["numeric_branch"])
    model.shared_trunk.load_state_dict(ckpt["shared_trunk"])
    model.steer_head.load_state_dict(ckpt["steer_head"])
    model.brake_trunk.load_state_dict(ckpt["brake_trunk"])
    model.brake_head.load_state_dict(ckpt["brake_head"])
    model.eval()

    cam = YOLOPControlGradCAM(model)

    # Typical cruise numeric: [throttle, brake, steer, speed/130]
    numeric = torch.tensor(
        [[0.60, 0.00, 0.00, 50.0 / 130.0]], dtype=torch.float32, device=device
    )

    # Two carefully selected frames:
    # Frame A: prominent tree in center-right → exposes steer head attending to vegetation
    # Frame B: clear car ahead + trees → exposes brake head correctly attending to vehicle
    frame_files = [
        os.path.join(ROOT, "debug_logs", "20260708_141626", "event_frame_00022.png"),
        os.path.join(ROOT, "debug_logs", "20260708_142507", "event_frame_00063.png"),
    ]
    row_labels = [
        "Scene A\n(prominent tree)",
        "Scene B\n(vehicle ahead)",
    ]

    rows, cols = len(frame_files), 3
    fig, axes = plt.subplots(rows, cols, figsize=(IEEE_W, 4.8))
    col_labels = [
        "Original frame",
        r"Grad-CAM — steer head $\hat{\delta}$",
        r"Grad-CAM — brake head $\hat{b}$",
    ]

    for r, fpath in enumerate(frame_files):
        img = cv2.imread(fpath)
        if img is None:
            print(f"  [fig2] WARNING: could not read {fpath}")
            continue

        tensor = preprocess_image_bgr(img).unsqueeze(0).to(device)

        steer_map = cam(tensor, numeric, target="steer")
        brake_map = cam(tensor, numeric, target="brake")

        visuals = [
            img,
            overlay_heatmap(img, steer_map, alpha=0.52),
            overlay_heatmap(img, brake_map, alpha=0.52),
        ]

        for c, vis in enumerate(visuals):
            ax = axes[r, c]
            ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
            if r == 0:
                ax.set_title(col_labels[c], fontsize=8, pad=4)

        axes[r, 0].set_ylabel(row_labels[r], fontsize=7.5, labelpad=5)

    plt.subplots_adjust(left=0.10, right=0.99, top=0.95, bottom=0.01,
                        wspace=0.03, hspace=0.05)

    for ext in ("pdf", "png"):
        path = os.path.join(OUT, f"fig2_gradcam.{ext}")
        fig.savefig(path, format=ext)
        print(f"  [fig2] saved {path}")

    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3  —  Per-method variance bar chart (summary)
# ═════════════════════════════════════════════════════════════════════════════

def figure3():
    """
    Horizontal bar chart comparing Var(δ) and |r(δ, eᵧ)| for both methods.
    Compact single-column figure (3.54" wide) suitable for side-by-side placement.
    """
    df = _load_window(500)

    methods = ["NN steer head\n(direct, collapsed)", "Hybrid PID\ncontroller"]
    var_vals = [df["steer_raw_nn"].var(), df["pid_steer"].var()]
    corr_vals = [abs(df["steer_raw_nn"].corr(df["lane_error"])),
                 abs(df["pid_steer"].corr(df["lane_error"]))]

    fig, (ax_v, ax_r) = plt.subplots(1, 2, figsize=(7.16, 2.0))
    colors = [C_NN, C_PID]
    bar_kw = dict(height=0.45, align="center")

    # Var(δ) — log scale
    bars_v = ax_v.barh(methods, var_vals, color=colors, **bar_kw)
    ax_v.set_xlabel(r"$\mathrm{Var}(\delta)$")
    ax_v.set_xscale("log")
    ax_v.set_xlim(1e-6, 5e-3)
    ax_v.spines[["top", "right"]].set_visible(False)
    ax_v.set_title("Steering command variance", fontsize=8)
    for bar, val in zip(bars_v, var_vals):
        ax_v.text(val * 1.2, bar.get_y() + bar.get_height() / 2,
                  f"{val:.2e}", va="center", ha="left", fontsize=7)

    # |r(δ, eᵧ)| — linear
    bars_r = ax_r.barh(methods, corr_vals, color=colors, **bar_kw)
    ax_r.set_xlabel(r"$|r(\delta,\,e_y)|$")
    ax_r.set_xlim(0, 1.05)
    ax_r.axvline(1.0, color=C_ZERO, lw=0.6, ls="--")
    ax_r.spines[["top", "right"]].set_visible(False)
    ax_r.set_title("Correlation with lateral error", fontsize=8)
    ax_r.set_yticklabels([])
    for bar, val in zip(bars_r, corr_vals):
        ax_r.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                  f"{val:.3f}", va="center", ha="left", fontsize=7)

    plt.subplots_adjust(wspace=0.12, left=0.24, right=0.96,
                        top=0.88, bottom=0.22)

    for ext in ("pdf", "png"):
        path = os.path.join(OUT, f"fig3_variance_bar.{ext}")
        fig.savefig(path, format=ext)
        print(f"  [fig3] saved {path}")

    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n=== Generating paper figures ===\n")

    print("--- Figure 1: Steering comparison ---")
    figure1()

    print("--- Figure 2: Grad-CAM ---")
    figure2()

    print("--- Figure 3: Variance bar chart ---")
    figure3()

    print(f"\nAll figures saved to: {OUT}\n")
