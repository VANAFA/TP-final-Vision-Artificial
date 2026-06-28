"""YOLOP-backbone control fine-tuning: model + dataset for the Comma2k19 pipeline.

Loads the official YOLOP/lib/models/yolop.py architecture with the pretrained
YOLOP/weights/End-to-end.pth checkpoint (same loading pattern as
yolop_reproduction.ipynb), freezes it, and adds a small trainable head that fuses
pooled image features with current numeric telemetry (throttle, brake, steer, speed
-- already in BeamNG units) to predict corrected (steer, brake).

Throttle/acceleration is intentionally NOT predicted by this network anymore: longitudinal
speed is handled by a separate, non-adaptive cruise-control routine outside this model (see
"2 Closed Loop BeamNG Control.ipynb"). The previous 3-output head (throttle, brake, steer)
shared every layer of `fusion_head` across all three targets, so comma2k19's "hard brake ->
wheel held straight" emergency-stop frames leaked a brake->steer correlation into the only
trainable part of the network. That is what collapsed steering to straight-ahead the moment
BeamNG's bridge/overpass shadows triggered the brake output. Splitting steer/brake into
separate heads with a gradient stop on the brake path (see
`YOLOPControlNet.detach_brake_grad`) removes that leakage path -- note the YOLOP backbone
itself was already frozen (`requires_grad_(False)`), so it was never the thing being
"flooded"; the small fusion MLP was.
"""

from __future__ import annotations

import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    """Aspect-ratio-preserving letterbox resize (uniform scale + pad, never anisotropic
    stretch). Same algorithm used by YOLOPDetector in yolop_beamng.py.

    This is what keeps lane-line *spacing* correct: a plain `cv2.resize(img, (640, 640))`
    would stretch a 4:3 BeamNG capture to 1:1 and squeeze every lane-line gap horizontally.
    Scaling both axes by a single ratio `r` and padding the leftover border (instead of
    stretching it to fill the target shape) is what prevents that distortion. With BeamNG's
    recommended 640x480 capture (aspect 1.333, matching comma2k19's 1164x874 source at
    aspect 1.332, see camera_calibration.py) `r` lands at ~1.0, so frames are only padded,
    never rescaled at all -- if you see lane lines looking "narrower" in-sim, that is the
    camera vertical-FOV mismatch (also handled in camera_calibration.py), not this resize.
    """
    shape = img_rgb.shape[:2]
    canvas = np.full((new_shape[0], new_shape[1], 3), color, dtype=np.float32)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad_w, new_unpad_h = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = (new_shape[1] - new_unpad_w) // 2, (new_shape[0] - new_unpad_h) // 2
    if shape[::-1] != (new_unpad_w, new_unpad_h):
        # INTER_AREA is only correct for downscaling (r < 1); upscaling with it aliases.
        interp = cv2.INTER_AREA if r < 1.0 else cv2.INTER_LINEAR
        img_rgb = cv2.resize(img_rgb, (new_unpad_w, new_unpad_h), interpolation=interp)
    canvas[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :] = img_rgb
    return canvas


def build_train_augmentations() -> A.Compose:
    """Photometric-only augmentations (no geometric warp, so steer/brake targets stay valid).

    Goal: break the spurious "dark patch in frame -> brake" correlation baked into
    comma2k19 (hard-brake frames are disproportionately under bridges/tunnels/overcast
    light), so the fine-tuned model stops treating BeamNG's bridge/overpass shadows as a
    brake trigger.

    - `RandomShadow`: soft-edged synthetic cast shadows crossing the lane -- the closest
      photometric analogue to BeamNG's bridge/overpass shadows.
    - `CoarseDropout` (Cutout): solid opaque rectangles over the road -- a harder, sharper
      occluder than RandomShadow; forces sane steer/brake even when part of the lane is
      fully gone from the frame.
    - `RandomBrightnessContrast` / `ColorJitter` / `RandomGamma`: global illumination and
      white-balance shifts, since BeamNG's lighting engine renders exposure differently
      from the comma EON's rolling-shutter camera.

    None of these touch image geometry, so `steer_target`/`brake_target` (read straight from
    the manifest) stay correct for the augmented image -- only the *visual* appearance of
    "this looks like an emergency-brake frame" is perturbed. This is the standard
    photometric-augmentation argument for out-of-distribution (OOD) robustness: lighting/
    occluder/color-cast variations the augmentation already covers at train time no longer
    count as "unseen" at test time, so behavior under BeamNG's different shadow/lighting
    model degrades gracefully instead of triggering a decision boundary the network only
    ever learned from comma2k19's particular lighting conditions.
    """
    return A.Compose([
        A.RandomShadow(shadow_roi=(0.0, 0.4, 1.0, 1.0), num_shadows_limit=(1, 3),
                        shadow_dimension=5, shadow_intensity_range=(0.4, 0.8), p=0.5),
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(0.08, 0.3),
                          hole_width_range=(0.08, 0.5), fill=0, p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.7),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.05, p=0.5),
        A.RandomGamma(gamma_limit=(60, 140), p=0.3),
    ])


def preprocess_image_bgr(img_bgr: np.ndarray, size: int = 640, augmentations: A.Compose | None = None) -> torch.Tensor:
    img_rgb = img_bgr[:, :, ::-1].copy()
    if augmentations is not None:
        # Augment the raw frame *before* letterboxing, so Cutout/shadow patches land on
        # real road pixels instead of on the gray padding bars.
        img_rgb = augmentations(image=img_rgb)["image"]
    canvas = resize_unscale(img_rgb, (size, size))
    img = canvas / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = img.transpose(2, 0, 1).astype(np.float32)
    return torch.from_numpy(img)


class YOLOPControlNet(nn.Module):
    """Frozen YOLOP backbone + small trainable shared trunk + separate steer/brake heads.

    Only two control outputs are predicted: `steer` and `brake`. Throttle/acceleration is
    deliberately excluded -- it is handled by an external non-adaptive cruise-control
    routine, so it has no business sharing gradients with the safety-critical steer output.

    `detach_brake_grad` (default True) implements the requested "gradient stop": `brake_trunk`
    (brake's own compression MLP, parallel to `shared_trunk`) reads a `.detach()`-ed copy of
    the *fused* pooled-image + numeric features, so backpropagating the brake loss only
    updates `brake_trunk`/`brake_head`'s own parameters and never touches `shared_trunk` or
    `numeric_branch`. The steer head still backprops through the shared trunk normally.

    Brake gets a *separate* trunk rather than reading `shared_trunk`'s 64-dim output (the
    original design): `shared_trunk` is only ever trained from steer's loss (brake's gradient
    is cut off before reaching it), so its 64-dim bottleneck is shaped purely to preserve
    steer-relevant structure. Feeding that bottleneck into `brake_head` starved brake of any
    visual signal it didn't share with steer -- confirmed by Grad-CAM producing near-identical
    heatmaps for both heads, and by `brake_head` collapsing to predicting close to the
    dataset's marginal mean regardless of the true brake target. Giving brake its own trunk
    off the *raw* fused features (still detached, so no gradient leaks back into steer's path)
    lets it learn its own compression instead of inheriting steer's.

    Road/lane attention (this revision): YOLOP's frozen forward pass already computes
    `da_seg_out` (drivable-area) and `ll_seg_out` (lane-line) segmentation maps on every call --
    earlier revisions discarded both and pooled `encoder_feat` uniformly over the whole frame,
    including hood/sky/background pixels that have nothing to do with the driving decision
    (confirmed by Grad-CAM lighting up guardrails/trees/sky for `steer`). `_road_mask` resizes
    those two maps down to the encoder's spatial grid and `_masked_pool` uses them as a soft
    spatial-attention weight, so `img_feat` is now a road+lane-weighted average instead of a
    flat one -- background pixels contribute close to nothing. This costs nothing extra from
    the backbone (already computed); it only changes how the existing frozen feature map gets
    pooled.

    Vehicle awareness (this revision): the frozen detection head's raw output is similarly
    already computed and was previously discarded. `_vehicle_features` runs the existing
    YOLOP NMS utility (frozen, no-grad bbox post-processing, not a new trainable component) and
    reduces the surviving boxes to a 3-dim summary (how many vehicles, how large/close the
    nearest one is, its lateral offset from frame center) that feeds `brake_trunk` alongside
    the pooled image feature -- `steer_head` does not get this, since lane-following doesn't
    need lead-vehicle distance and braking does. This detection head is single-class
    (vehicle-only; this YOLOP checkpoint has no separate pedestrian class).
    """

    ENCODER_LAYER_INDEX = 16  # MCnet.model[16]: shared encoder output, 256ch, stride 16
    ENCODER_CHANNELS = 256
    NUMERIC_IN = 4   # throttle_t, brake_t, steer_t, speed_t -- current-state *context*, not an output
    NUMERIC_HIDDEN = 32
    SHARED_HIDDEN = 64
    OUT_DIM = 2      # steer, brake
    VEHICLE_FEAT_DIM = 3  # [n_vehicles_norm, nearest_box_area_norm, nearest_box_lateral_offset]
    MASK_FLOOR = 0.05  # keeps masked pooling from degenerating to all-zero if segmentation
                        # fails outright on an out-of-distribution (BeamNG) frame
    DET_CONF_THRES = 0.3
    DET_IOU_THRES = 0.45

    def __init__(
        self,
        weights_path: Path = YOLOP_REPO / "weights" / "End-to-end.pth",
        device="cpu",
        detach_brake_grad: bool = True,
    ):
        super().__init__()
        self.detach_brake_grad = detach_brake_grad

        self.backbone = load_yolop_backbone(weights_path, device=device)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

        self._encoder_feat = None
        self.backbone.model[self.ENCODER_LAYER_INDEX].register_forward_hook(self._capture_encoder_feat)

        self.numeric_branch = nn.Sequential(
            nn.Linear(self.NUMERIC_IN, self.NUMERIC_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Linear(self.NUMERIC_HIDDEN, self.NUMERIC_HIDDEN),
            nn.ReLU(inplace=True),
        )
        fused_dim = self.ENCODER_CHANNELS + self.NUMERIC_HIDDEN
        self.shared_trunk = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.SHARED_HIDDEN),
            nn.ReLU(inplace=True),
        )
        self.steer_head = nn.Linear(self.SHARED_HIDDEN, 1)

        # Brake's own trunk, parallel to shared_trunk, fed from the same fused features (plus
        # vehicle_feat) but detached at the fusion point (not after shared_trunk -- see class
        # docstring) so it can learn its own compression instead of reusing steer's bottleneck.
        brake_fused_dim = fused_dim + self.VEHICLE_FEAT_DIM
        self.brake_trunk = nn.Sequential(
            nn.Linear(brake_fused_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.SHARED_HIDDEN),
            nn.ReLU(inplace=True),
        )
        self.brake_head = nn.Sequential(
            nn.Linear(self.SHARED_HIDDEN, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def _capture_encoder_feat(self, module, inputs, output):
        self._encoder_feat = output

    def trainable_parameters(self):
        return (
            list(self.numeric_branch.parameters())
            + list(self.shared_trunk.parameters())
            + list(self.steer_head.parameters())
            + list(self.brake_trunk.parameters())
            + list(self.brake_head.parameters())
        )

    def _road_mask(self, da_seg_out: torch.Tensor, ll_seg_out: torch.Tensor, spatial_hw) -> torch.Tensor:
        """Resizes YOLOP's drivable-area + lane-line maps (B,2,640,640 each, sigmoid already
        applied) down to the encoder feature map's spatial size and combines them into a
        single (B,1,h,w) soft attention weight -- `max` rather than `mean` so a thin lane line
        isn't washed out by the much larger drivable-area blob around it."""
        da_pos = da_seg_out[:, 1:2]  # channel 1 = "is drivable" positive class
        ll_pos = ll_seg_out[:, 1:2]  # channel 1 = "is lane line" positive class
        da_small = F.adaptive_avg_pool2d(da_pos, spatial_hw)
        ll_small = F.adaptive_avg_pool2d(ll_pos, spatial_hw)
        mask = torch.maximum(da_small, ll_small)
        return mask.clamp(min=self.MASK_FLOOR)

    @staticmethod
    def _masked_pool(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Masked average pool: feat (B,C,h,w), mask (B,1,h,w) -> (B,C). Pixels the mask is
        near-zero on (sky, hood, background) contribute almost nothing, instead of the flat
        per-pixel average `AdaptiveAvgPool2d` used to compute."""
        weighted = (feat * mask).sum(dim=(2, 3))
        denom = mask.sum(dim=(2, 3)).clamp(min=1e-6)
        return weighted / denom

    def _vehicle_features(self, det_raw) -> torch.Tensor:
        """`det_raw` is the `(z_cat, raw_per_scale)` tuple the frozen Detect head returns in
        eval mode; `z_cat` is (B, N, 6): xywh (pixel space, 640x640) + obj_conf + cls_conf for
        this checkpoint's single ("vehicle") class. Runs YOLOP's own NMS (frozen bbox
        post-processing, not a trainable step) and reduces surviving boxes to a fixed-size
        summary: how many vehicles, how large the nearest one is (a distance proxy -- closer
        vehicles occupy more pixels), and its lateral offset from frame center (is it in my
        lane). Frames with no detections get all-zero features (treated as "no vehicle seen")."""
        from lib.core.general import non_max_suppression  # YOLOP_REPO already on sys.path

        z_cat = det_raw[0].float()  # NMS/torchvision.ops.nms expects fp32; AMP gives fp16 here
        with torch.no_grad():
            preds = non_max_suppression(z_cat, conf_thres=self.DET_CONF_THRES, iou_thres=self.DET_IOU_THRES)

        feats = torch.zeros(len(preds), self.VEHICLE_FEAT_DIM, device=z_cat.device, dtype=torch.float32)
        for i, boxes in enumerate(preds):
            if boxes is None or boxes.shape[0] == 0:
                continue
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            best = torch.argmax(areas)
            nearest = boxes[best]
            feats[i, 0] = min(boxes.shape[0], 5) / 5.0
            feats[i, 1] = (areas[best] / (640.0 * 640.0)).float()
            feats[i, 2] = (((nearest[0] + nearest[2]) / 2.0 - 320.0) / 320.0).float()
        return feats

    def forward(self, image: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            det_out, da_seg_out, ll_seg_out = self.backbone(image)
        encoder_feat = self._encoder_feat  # (B, 256, h, w)

        mask = self._road_mask(da_seg_out, ll_seg_out, encoder_feat.shape[-2:])
        img_feat = self._masked_pool(encoder_feat, mask)  # (B, 256)

        num_feat = self.numeric_branch(numeric)  # (B, 32)
        fused = torch.cat([img_feat, num_feat], dim=1)

        shared = self.shared_trunk(fused)  # (B, 64)
        steer = torch.tanh(self.steer_head(shared))

        vehicle_feat = self._vehicle_features(det_out)  # (B, 3), no grad path at all
        brake_base = fused.detach() if self.detach_brake_grad else fused
        brake_in = torch.cat([brake_base, vehicle_feat], dim=1)
        brake_feat = self.brake_trunk(brake_in)  # (B, 64), brake's own compression
        brake = torch.sigmoid(self.brake_head(brake_feat))

        return torch.cat([steer, brake], dim=1)


class Comma2k19ControlDataset(Dataset):
    """Reads the manifest produced by tools/prepare_comma2k19_control_dataset.py.

    `throttle_target` is still written to the manifest by that script (kept for offline
    analysis/backward compatibility) but is no longer read as a training target here --
    `YOLOPControlNet` only predicts (steer, brake). `throttle_t` is still read as a
    *numeric input feature* (current-state context), since dropping a feature column is a
    separate decision from dropping an output head.
    """

    NUMERIC_COLS = ["throttle_t", "brake_t", "steer_t", "speed_t"]
    TARGET_COLS = ["steer_target", "brake_target"]

    def __init__(
        self,
        manifest_path: Path,
        frames_root: Path,
        split: str,
        image_size: int = 640,
        augment: bool | None = None,
    ):
        df = pd.read_csv(manifest_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.frames_root = Path(frames_root)
        self.image_size = image_size
        # Augment only the train split by default -- dev/test must stay clean to measure
        # real generalization, not augmentation-time noise.
        do_augment = (split == "train") if augment is None else augment
        self.augmentations = build_train_augmentations() if do_augment else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Each DataLoader worker is its own process with its own OpenCV thread pool; left at
        # cv2's default it spawns one thread per CPU core *per worker*, oversubscribing the
        # machine once num_workers > 1. Pin each worker to single-threaded OpenCV ops instead
        # (cheap/idempotent to call here -- no per-process worker_init_fn needed).
        cv2.setNumThreads(0)
        row = self.df.iloc[idx]
        img_bgr = cv2.imread(str(self.frames_root / row["image_path"]))
        image = preprocess_image_bgr(img_bgr, self.image_size, augmentations=self.augmentations)
        numeric = torch.tensor([row[c] for c in self.NUMERIC_COLS], dtype=torch.float32)
        target = torch.tensor([row[c] for c in self.TARGET_COLS], dtype=torch.float32)
        return image, numeric, target
