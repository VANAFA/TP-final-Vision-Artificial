"""YOLOP-backbone control fine-tuning: model + dataset for the Comma2k19 pipeline.

Loads the official YOLOP/lib/models/yolop.py architecture with the pretrained
YOLOP/weights/End-to-end.pth checkpoint (same loading pattern as
yolop_reproduction.ipynb), freezes it, and adds a small trainable head that fuses
pooled image features with current numeric telemetry (throttle, brake, steer, speed
-- already in BeamNG units) to predict corrected (throttle, brake, steer).
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent
YOLOP_REPO = ROOT / "YOLOP"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_yolop_backbone(weights_path: Path = YOLOP_REPO / "weights" / "End-to-end.pth", device="cpu"):
    """Loads the official YOLOP MCnet with pretrained weights. Mirrors yolop_reproduction.ipynb."""
    if str(YOLOP_REPO) not in sys.path:
        sys.path.insert(0, str(YOLOP_REPO))
    from lib.config import cfg
    from lib.models import get_net

    model = get_net(cfg)
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    return model.to(device)


def resize_unscale(img_rgb: np.ndarray, new_shape=(640, 640), color=114):
    """Same letterbox resize used by YOLOPDetector in yolop_beamng.py."""
    shape = img_rgb.shape[:2]
    canvas = np.full((new_shape[0], new_shape[1], 3), color, dtype=np.float32)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad_w, new_unpad_h = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = (new_shape[1] - new_unpad_w) // 2, (new_shape[0] - new_unpad_h) // 2
    if shape[::-1] != (new_unpad_w, new_unpad_h):
        img_rgb = cv2.resize(img_rgb, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_AREA)
    canvas[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :] = img_rgb
    return canvas


def preprocess_image_bgr(img_bgr: np.ndarray, size: int = 640) -> torch.Tensor:
    img_rgb = img_bgr[:, :, ::-1].copy()
    canvas = resize_unscale(img_rgb, (size, size))
    img = canvas / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = img.transpose(2, 0, 1).astype(np.float32)
    return torch.from_numpy(img)


class YOLOPControlNet(nn.Module):
    """Frozen YOLOP backbone + small trainable fusion head for control regression."""

    ENCODER_LAYER_INDEX = 16  # MCnet.model[16]: shared encoder output, 256ch, stride 16
    ENCODER_CHANNELS = 256
    NUMERIC_IN = 4   # throttle_t, brake_t, steer_t, speed_t
    NUMERIC_HIDDEN = 32
    OUT_DIM = 3      # throttle, brake, steer

    def __init__(self, weights_path: Path = YOLOP_REPO / "weights" / "End-to-end.pth", device="cpu"):
        super().__init__()
        self.backbone = load_yolop_backbone(weights_path, device=device)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

        self._encoder_feat = None
        self.backbone.model[self.ENCODER_LAYER_INDEX].register_forward_hook(self._capture_encoder_feat)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.numeric_branch = nn.Sequential(
            nn.Linear(self.NUMERIC_IN, self.NUMERIC_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Linear(self.NUMERIC_HIDDEN, self.NUMERIC_HIDDEN),
            nn.ReLU(inplace=True),
        )
        fused_dim = self.ENCODER_CHANNELS + self.NUMERIC_HIDDEN
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.OUT_DIM),
        )

    def _capture_encoder_feat(self, module, inputs, output):
        self._encoder_feat = output

    def trainable_parameters(self):
        return list(self.numeric_branch.parameters()) + list(self.fusion_head.parameters())

    def forward(self, image: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.backbone(image)
        img_feat = self.pool(self._encoder_feat).flatten(1)  # (B, 256)

        num_feat = self.numeric_branch(numeric)  # (B, 32)
        fused = torch.cat([img_feat, num_feat], dim=1)
        out = self.fusion_head(fused)

        steer = torch.tanh(out[:, 2:3])
        throttle = torch.sigmoid(out[:, 0:1])
        brake = torch.sigmoid(out[:, 1:2])
        return torch.cat([throttle, brake, steer], dim=1)


class Comma2k19ControlDataset(Dataset):
    """Reads the manifest produced by tools/prepare_comma2k19_control_dataset.py."""

    NUMERIC_COLS = ["throttle_t", "brake_t", "steer_t", "speed_t"]
    TARGET_COLS = ["throttle_target", "brake_target", "steer_target"]

    def __init__(self, manifest_path: Path, frames_root: Path, split: str, image_size: int = 640):
        df = pd.read_csv(manifest_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.frames_root = Path(frames_root)
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_bgr = cv2.imread(str(self.frames_root / row["image_path"]))
        image = preprocess_image_bgr(img_bgr, self.image_size)
        numeric = torch.tensor([row[c] for c in self.NUMERIC_COLS], dtype=torch.float32)
        target = torch.tensor([row[c] for c in self.TARGET_COLS], dtype=torch.float32)
        return image, numeric, target
