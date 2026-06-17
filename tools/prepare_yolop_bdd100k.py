from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw
from tqdm import tqdm


IMAGE_SIZE = (1280, 720)


def draw_poly(draw: ImageDraw.ImageDraw, vertices, fill: int, closed: bool, width: int = 8):
    points = [(float(x), float(y)) for x, y in vertices]
    if len(points) < 2:
        return
    if closed and len(points) >= 3:
        draw.polygon(points, fill=fill)
    else:
        draw.line(points, fill=fill, width=width, joint="curve")


def convert_item(item: dict, image_root: Path, out_root: Path, split: str) -> bool:
    image_name = item["name"]
    image_path = image_root / split / image_name
    if not image_path.exists():
        return False

    stem = Path(image_name).stem
    det_dir = out_root / "det_annotations" / split
    da_dir = out_root / "da_seg_annotations" / split
    ll_dir = out_root / "ll_seg_annotations" / split
    det_dir.mkdir(parents=True, exist_ok=True)
    da_dir.mkdir(parents=True, exist_ok=True)
    ll_dir.mkdir(parents=True, exist_ok=True)

    objects = item.get("labels", [])
    det_payload = {
        "name": image_name,
        "attributes": item.get("attributes", {}),
        "timestamp": item.get("timestamp"),
        "frames": [{"objects": objects}],
    }
    (det_dir / f"{stem}.json").write_text(json.dumps(det_payload), encoding="utf-8")

    da_mask = Image.new("L", IMAGE_SIZE, 0)
    ll_mask = Image.new("L", IMAGE_SIZE, 0)
    da_draw = ImageDraw.Draw(da_mask)
    ll_draw = ImageDraw.Draw(ll_mask)

    for obj in objects:
        category = obj.get("category")
        for poly in obj.get("poly2d", []):
            vertices = poly.get("vertices", [])
            closed = bool(poly.get("closed", False))
            if category == "drivable area":
                if obj.get("attributes", {}).get("areaType") == "direct":
                    draw_poly(da_draw, vertices, fill=255, closed=True)
            elif category == "lane":
                draw_poly(ll_draw, vertices, fill=255, closed=closed, width=8)

    da_mask.save(da_dir / f"{stem}.png")
    ll_mask.save(ll_dir / f"{stem}.png")
    return True


def prepare_split(source_root: Path, out_root: Path, split: str, limit: int | None):
    image_root = source_root / "bdd100k" / "bdd100k" / "images" / "100k"
    labels_path = (
        source_root
        / "bdd100k_labels_release"
        / "bdd100k"
        / "labels"
        / f"bdd100k_labels_images_{split}.json"
    )
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)

    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if limit is not None:
        labels = labels[:limit]

    images_link = out_root / "images"
    images_link.mkdir(parents=True, exist_ok=True)
    split_link = images_link / split
    if not split_link.exists():
        # A junction avoids duplicating the large image folder on Windows.
        import subprocess

        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(split_link), str(image_root / split)],
            check=True,
            capture_output=True,
            text=True,
        )

    written = 0
    skipped = 0
    for item in tqdm(labels, desc=f"Preparing {split}"):
        if convert_item(item, image_root, out_root, split):
            written += 1
        else:
            skipped += 1

    return written, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=Path("datasets/yolop_bdd100k"))
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    written, skipped = prepare_split(args.source_root, args.out_root, args.split, args.limit)
    print(f"Prepared {written} samples in {args.out_root} ({skipped} skipped).")


if __name__ == "__main__":
    main()
