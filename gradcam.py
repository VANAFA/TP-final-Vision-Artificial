"""Grad-CAM for YOLOPControlNet's steer/brake heads.

`YOLOPControlNet.forward()` runs the frozen YOLOP backbone under `torch.no_grad()` (it's
frozen, no autograd needed there during normal training/inference), so the captured encoder
feature map (`model._encoder_feat`) has no grad history to backprop through. Grad-CAM needs
exactly that gradient, so `YOLOPControlGradCAM` re-runs the backbone with grad enabled for one
extra pass and reads the gradient off that same feature map -- it does not modify
`YOLOPControlNet` itself, just calls its submodules directly.

Visualizing `brake` deliberately ignores `detach_brake_grad`: that flag only blocks the brake
loss from updating `shared_trunk`/`numeric_branch` *during training* (brake reads a detached
copy of the fused features through its own `brake_trunk`). The forward computation itself is
identical either way, so for "which pixels drove this brake value" the detach would just zero
out the answer (no path back to the image) instead of changing it -- this module always uses
the non-detached path regardless of how the loaded checkpoint was trained.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class YOLOPControlGradCAM:
    """Grad-CAM against `YOLOPControlNet.ENCODER_LAYER_INDEX` (the same 256ch/stride-16
    feature map the control head pools from), for either the `steer` or `brake` output.

    Usage: `cam = YOLOPControlGradCAM(control_model)(image_tensor, numeric_tensor, target="brake")`
    where `image_tensor`/`numeric_tensor` are the same (unbatched-then-unsqueezed) tensors
    you'd pass to the model directly. Returns a single-image (Hf, Wf) array in [0, 1] --
    use `overlay_heatmap` to resize and blend it onto a frame.
    """

    def __init__(self, model):
        self.model = model

    def __call__(self, image: torch.Tensor, numeric: torch.Tensor, target: str = "steer") -> np.ndarray:
        if target not in ("steer", "brake"):
            raise ValueError(f"target must be 'steer' or 'brake', got {target!r}")

        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        image = image.clone().requires_grad_(True)
        with torch.enable_grad():
            det_out, da_seg_out, ll_seg_out = self.model.backbone(image)  # fires forward hook -> _encoder_feat
            feat = self.model._encoder_feat
            feat.retain_grad()

            # Mirrors YOLOPControlNet.forward()'s masked pooling: gradient through the road/lane
            # mask itself is irrelevant here (the mask doesn't depend on `feat`), but the masked
            # pool's per-pixel weighting still shapes d(out)/d(feat) the same way it does at
            # training/inference time, so the resulting heatmap should already look more
            # road/lane-focused than the old flat-average pooling did.
            mask = self.model._road_mask(da_seg_out, ll_seg_out, feat.shape[-2:])
            img_feat = self.model._masked_pool(feat, mask)
            num_feat = self.model.numeric_branch(numeric)
            fused = torch.cat([img_feat, num_feat], dim=1)

            if target == "steer":
                shared = self.model.shared_trunk(fused)
                out = torch.tanh(self.model.steer_head(shared))
            else:
                vehicle_feat = self.model._vehicle_features(det_out)
                brake_in = torch.cat([fused, vehicle_feat], dim=1)  # no detach, see module docstring
                brake_feat = self.model.brake_trunk(brake_in)
                out = torch.sigmoid(self.model.brake_head(brake_feat))

            out.sum().backward()

        grad = feat.grad
        act = feat.detach()
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * act).sum(dim=1))[0].cpu().numpy()

        cam -= cam.min()
        peak = cam.max()
        if peak > 1e-8:
            cam /= peak
        return cam


def overlay_heatmap(canvas_bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Resizes `cam` (any HxW, values in [0, 1]) up to `canvas_bgr`'s size and blends it on
    top as a JET colormap.

    Pass the *letterboxed* 640x640 frame (`resize_unscale`'s output, what `preprocess_image_bgr`
    actually fed the network) as `canvas_bgr`, not the raw BeamNG/dataset frame -- otherwise the
    heatmap won't line up with what the network looked at (including the gray padding bars).
    """
    h, w = canvas_bgr.shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(heatmap, alpha, canvas_bgr, 1 - alpha, 0)
