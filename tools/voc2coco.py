"""
Convert LabelImg Pascal-VOC XML annotations to YOLO txt and a single COCO json.

    python tools/voc2yolo_coco.py --xml-dir samples/labels_voc \
        --yolo-dir samples/labels_yolo --coco-json samples/annotations_coco.json

Class names in the XML are mapped through NAME_MAP. Unknown names raise an error
instead of being silently dropped, so typos in the original annotations surface.
Edit NAME_MAP to match the exact strings used in the NVHD XML files.
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

CLASS_NAMES = ["two-wheeler", "helmet", "without_helmet"]
NAME_MAP = {  # XML <name> string -> class index. TODO: confirm against the real NVHD XML strings.
    "two-wheeler": 0,
    "helmet": 1,
    "without_helmet": 2,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml-dir", type=Path, required=True)
    ap.add_argument("--yolo-dir", type=Path, required=True)
    ap.add_argument("--coco-json", type=Path, required=True)
    a = ap.parse_args()
    a.yolo_dir.mkdir(parents=True, exist_ok=True)

    images, anns, ann_id = [], [], 1
    for img_id, xml_path in enumerate(sorted(a.xml_dir.glob("*.xml")), start=1):
        root = ET.parse(xml_path).getroot()
        W = int(root.findtext("size/width"))
        H = int(root.findtext("size/height"))
        fname = root.findtext("filename") or f"{xml_path.stem}.jpg"
        images.append({"id": img_id, "file_name": fname, "width": W, "height": H})

        lines = []
        for obj in root.iter("object"):
            name = obj.findtext("name").strip()
            if name not in NAME_MAP:
                raise ValueError(f"{xml_path}: unknown class name '{name}' (update NAME_MAP)")
            cls = NAME_MAP[name]
            b = obj.find("bndbox")
            x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            x1, x2 = max(0.0, x1), min(float(W), x2)
            y1, y2 = max(0.0, y1), min(float(H), y2)
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                raise ValueError(f"{xml_path}: degenerate box {x1, y1, x2, y2}")
            lines.append(f"{cls} {(x1 + w / 2) / W:.6f} {(y1 + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
            anns.append({"id": ann_id, "image_id": img_id, "category_id": cls + 1,
                         "bbox": [x1, y1, w, h], "area": w * h, "iscrowd": 0})
            ann_id += 1
        (a.yolo_dir / f"{xml_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    coco = {"images": images, "annotations": anns,
            "categories": [{"id": i + 1, "name": n} for i, n in enumerate(CLASS_NAMES)]}
    a.coco_json.write_text(json.dumps(coco, indent=2))
    print(f"{len(images)} images, {len(anns)} boxes -> {a.yolo_dir}, {a.coco_json}")


if __name__ == "__main__":
    main()
