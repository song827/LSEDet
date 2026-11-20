"""
Standard detection metrics via the official COCO API (pycocotools.COCOeval).

Reports AP@[.5:.95], AP50, AP75, AP_small / AP_medium / AP_large, AR, and per-class AP50.
No custom metric code: matching, interpolation and area ranges are pycocotools' own.

Two input modes
---------------
1) COCO JSON (recommended):
     python eval.py --gt-json gt.json --pred-json preds.json

   preds.json is a list of {"image_id", "category_id", "bbox": [x, y, w, h], "score"}
   in absolute pixels, with ids matching gt.json.

2) YOLO txt (converted to COCO in memory):
     python eval.py --images samples/images --gt-dir samples/labels_yolo --pred-dir runs/pred_labels

   GT   line: <cls> <cx> <cy> <w> <h>          (normalized 0-1)
   Pred line: <cls> <cx> <cy> <w> <h> <conf>   (normalized 0-1; same as Ultralytics save_txt + save_conf)
   Label files are matched to images by file stem.

Alternatively, an Ultralytics-format model can be validated with its built-in
pipeline (`yolo val model=... data=...`); note its mAP is computed with a
different interpolation and is not numerically identical to COCOeval.

Requirements: pip install pycocotools pillow numpy
"""

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

CLASS_NAMES = ["two-wheeler", "helmet", "without_helmet"]  
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def yolo_to_coco(images_dir: Path, gt_dir: Path, pred_dir: Path):
    images, annotations, detections = [], [], []
    ann_id = 1
    img_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not img_paths:
        raise SystemExit(f"No images found in {images_dir}")

    for img_id, img_path in enumerate(img_paths, start=1):
        with Image.open(img_path) as im:
            W, H = im.size
        images.append({"id": img_id, "file_name": img_path.name, "width": W, "height": H})

        for src, is_pred in ((gt_dir / f"{img_path.stem}.txt", False), (pred_dir / f"{img_path.stem}.txt", True)):
            if not src.exists():
                print(f"Warning: missing {'GT' if not is_pred else 'pred'} file for {img_path.name}")
                continue
            for line_no, line in enumerate(src.read_text().splitlines(), start=1):
                parts = line.split()
                if not parts:
                    continue
                expected = 6 if is_pred else 5
                if len(parts) != expected:
                    raise ValueError(f"{src}:{line_no}: expected {expected} fields, got {len(parts)}")
                cls = int(parts[0])
                if not 0 <= cls < len(CLASS_NAMES):
                    raise ValueError(f"{src}:{line_no}: class id {cls} out of range")
                cx, cy, w, h = (float(v) for v in parts[1:5])
                bw, bh = w * W, h * H
                bbox = [cx * W - bw / 2, cy * H - bh / 2, bw, bh]
                if is_pred:
                    detections.append({"image_id": img_id, "category_id": cls + 1,
                                       "bbox": bbox, "score": float(parts[5])})
                else:
                    annotations.append({"id": ann_id, "image_id": img_id, "category_id": cls + 1,
                                        "bbox": bbox, "area": bw * bh, "iscrowd": 0})
                    ann_id += 1

    gt = {"images": images, "annotations": annotations,
          "categories": [{"id": i, "name": n} for i, n in enumerate(CLASS_NAMES)]}
    return gt, detections


def evaluate(gt_dict, dets):
    coco_gt = COCO()
    coco_gt.dataset = gt_dict
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.createIndex()
    if not dets:
        raise SystemExit("No detections to evaluate.")
    coco_dt = coco_gt.loadRes(dets)

    E = COCOeval(coco_gt, coco_dt, iouType="bbox")
    E.evaluate()
    E.accumulate()
    E.summarize()

    # Per-class AP50 (area=all, maxDets=100). precision: [T, R, K, A, M]
    print("\nPer-class AP50:")
    precision = E.eval["precision"]
    t50 = int(np.where(np.isclose(E.params.iouThrs, 0.5))[0][0])
    for k, cat_id in enumerate(E.params.catIds):
        p = precision[t50, :, k, 0, -1]
        p = p[p > -1]
        ap = float(p.mean()) if p.size else float("nan")
        print(f"  {coco_gt.cats[cat_id]['name']:<16s} {ap:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-json", type=Path)
    ap.add_argument("--pred-json", type=Path)
    ap.add_argument("--images", type=Path)
    ap.add_argument("--gt-dir", type=Path)
    ap.add_argument("--pred-dir", type=Path)
    a = ap.parse_args()

    if a.gt_json and a.pred_json:
        gt = json.loads(a.gt_json.read_text())
        dets = json.loads(a.pred_json.read_text())
    elif a.images and a.gt_dir and a.pred_dir:
        gt, dets = yolo_to_coco(a.images, a.gt_dir, a.pred_dir)
    else:
        ap.error("use either --gt-json/--pred-json or --images/--gt-dir/--pred-dir")
    evaluate(gt, dets)


if __name__ == "__main__":
    main()
