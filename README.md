# LSEDet — core modules and evaluation (partial release)

This repository accompanies the paper's Section 3 and releases:

| Path | Content |
|---|---|
| `models/sfepn.py` | Layer definitions of **MSFC** (Sec. 3.2.1), **TFM** (Sec. 3.2.2) and the **Dynamic-IoU** loss (Sec. 3.3) |
| `eval.py` | COCO-style metric computation using the official `pycocotools.COCOeval` |
| `tools/voc2coco.py` | LabelImg VOC XML → COCO JSON conversion helper |
| `samples/` | De-identified NVHD test images with annotations in COCO format |

It is **not** the full training code: the backbone, neck wiring, detection head, label
assignment, LAMP pruning and OPED distillation are not included, and no pretrained
weights are provided. The NVHD dataset itself is private; `samples/` is only an
illustration of image conditions and the annotation format.

## Quick check

Install dependencies:
```bash
pip install -r requirements.txt
python models/sfepn.py          # shape sanity check of MSFC / TFM / loss
python eval.py --gt samples/annotations_coco.json --pred your_predictions.json
```

`eval.py` reports AP@[.5:.95], AP50, AP75, AP_S / AP_M / AP_L and per-class AP50. Area
ranges follow COCO (small < 32², medium 32²–96², large > 96² pixels, measured on the
original 1920 × 1080 frame).

## NVHD annotation protocol

**Source.** Videos recorded at multiple sites in China with a tripod-mounted camera at
an elevated position simulating the overhead view of traffic surveillance cameras,
under sunny, rainy and foggy conditions; frames extracted at 10 fps, 1920 × 1080. A
portion of the dataset was supplemented by web-crawled images. The full set contains
9,296 images split 8 : 1 : 1 into train / val / test.

**Tool and format.** Boxes were drawn with LabelImg and stored as Pascal-VOC XML
(axis-aligned, pixel coordinates `xmin, ymin, xmax, ymax`).

**Classes.**

| COCO id | Name | Definition |
|---|---|---|
| 0 | `two-wheeler` | Complete body of two-wheeled non-motorized vehicles (electric bikes, bicycles); the bounding box tightly encloses the vehicle outline. |
| 1 | `helmet` | Head region of a rider wearing a safety helmet; the bounding box tightly encloses the helmet edge. |
| 2 | `without_helmet` | Exposed head region of a rider not wearing a safety helmet; the bounding box tightly encloses the head outline. |

**Labelling rules.** 

- Only moving two-wheeler riders are annotated; parked two-wheelers and pedestrians wheeling a bike are not labelled.
- Targets with occlusion over 70% that cannot be judged for helmet-wearing are not annotated.
- Caps, hoods and non-safety headwear are counted as `without_helmet`.
- All annotations were completed independently by two annotators; inconsistent samples were reviewed by a third senior annotator, and annotation consistency (IoU > 0.9) reached over 98%.

**File formats in `samples/`.**

```
samples/
├── images/            # de-identified JPGs
└── annotations_coco.json   # bbox = [x, y, w, h] in pixels, category_id matches the COCO ids above
```

**De-identification.** Faces, licence plates and other identifying text were blurred manually; EXIF metadata was removed. Blurring changes pixels only, so box coordinates are unchanged. Because blurred faces can lie inside `without_helmet` / `helmet` boxes, the samples illustrate the data and format; they are not intended for reproducing reported metrics.

**Code access**

Due to pending patent applications and industrial confidentiality agreements, the full end-to-end training pipeline is not publicly released. For non-commercial academic research purposes, researchers may request the complete source code by contacting the corresponding author via email: `panlh@tyust.edu.cn`, after signing a code use agreement.
