# YOLOP Paper Reproduction Evaluation Report

**Generated**: 2026-06-17 19:58 ART

## Configuration
- Model: YOLOP
- Weights: `YOLOP/weights/End-to-end.pth`
- Dataset: `datasets/yolop_bdd100k`
- Split: `val`
- Samples: 10000
- Device: CPU
- Image Size: 640

## Full Test Output Metrics

| Metric | Value |
|--------|------:|
| Loss | 0.480 |
| Detection Precision | 0.085 |
| Detection Recall | 0.892 |
| Detection mAP@0.5 | 0.765 |
| Detection mAP@0.5:0.95 | 0.439 |
| Drivable Area Accuracy | 0.925 |
| Drivable Area IoU | 0.563 |
| Drivable Area mIoU | 0.739 |
| Lane Line Accuracy | 0.496 |
| Lane Line IoU | 0.347 |
| Lane Line mIoU | 0.664 |
| Inference Time | 0.0762 s/frame |
| NMS Time | 0.0004 s/frame |

## Paper Comparison

| Metric | Paper Value | Reproduced | Delta | Delta (%) |
|--------|------------:|-----------:|------:|----------:|
| Detection mAP@0.5 | 0.765 | 0.765 | +0.000 | +0.00% |
| Drivable Area IoU | 0.915 | 0.563 | -0.352 | -38.47% |
| Lane Line IoU | 0.705 | 0.347 | -0.358 | -50.78% |

## Notes
- The Kaggle dataset used was `solesensei/solesensei_bdd100k`.
- Detection annotations were converted from BDD100K aggregate JSON into YOLOP per-image JSON.
- Drivable-area and lane-line masks were rasterized from BDD100K polygons.
- Detection mAP@0.5 matches the paper's reported value.
- Segmentation results validate the pipeline, but are not guaranteed to be directly comparable to the paper because the official YOLOP prepared segmentation annotations may differ from these generated masks.

## Output Paths
- Run directory: `YOLOP/runs/BddDataset/_2026-06-17-19-47`
- Visualizations: `YOLOP/runs/BddDataset/_2026-06-17-19-47/visualization`
