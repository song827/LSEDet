"""
Core building blocks of the Small-object Feature Enhancement Pyramid Network (SFEPN)
used in LSEDet, plus the Dynamic-IoU bounding-box regression loss.

This file only contains self-contained layer / loss definitions (Sec. 3.2 and 3.3 of
the paper). It is NOT the full training pipeline: the backbone, neck wiring, detection
head, label assignment, pruning (LAMP) and distillation (OPED) code are not included.

Contents
--------
- ConvBNAct      : Conv2d + BN + SiLU helper
- MSFC           : Multi-scale Fusion and Context integrator (Sec. 3.2.1, Fig. 6)
- TFM            : Triple Feature Mix module (Sec. 3.2.2, Fig. 7)
- DynamicIoULoss : PIoU + Focaler linear-interval IoU remapping (Sec. 3.3, Eq. 1-6)

Note on naming: "Dynamic-IoU" is deliberately not abbreviated as "DIoU" here, because
DIoU conventionally refers to Distance-IoU (Zheng et al., 2020), a baseline in the paper.
"""

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ConvBNAct", "MSFC", "TFM", "DynamicIoULoss", "bbox_iou_xyxy"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class ConvBNAct(nn.Module):
    """Conv2d -> BatchNorm2d -> SiLU (YOLOv8-style basic conv)."""

    def __init__(self, c_in: int, c_out: int, k: int = 1, s: int = 1, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


# --------------------------------------------------------------------------- #
# 3.2.1 Multi-scale fusion and context integrator (MSFC)
# --------------------------------------------------------------------------- #
class MSFC(nn.Module):
    """
    Fuse adjacent pyramid levels (P2, P3, P4) with a 3D convolution over a
    (scale, H, W) volume.

    Steps (following Sec. 3.2.1):
      1. 1x1 conv aligns every input to `out_channels`.
      2. P3 / P4 are resized to the spatial size of P2 (nearest upsampling).
      3. Each map gets an extra "scale" axis and the maps are concatenated
         along it -> tensor of shape (B, C, S, H, W), S = number of levels.
      4. Conv3d -> BatchNorm3d -> LeakyReLU.
      5. MaxPool3d over the scale axis (kernel S x 1 x 1) -> (B, C, 1, H, W).
      6. The singleton scale axis is squeezed -> (B, C, H, W).

    Args:
        in_channels:  channels of the input maps, ordered from highest resolution
                      (P2) to lowest (P4).
        out_channels: channels after alignment and of the output.
        kernel_size:  Conv3d kernel as (scale, h, w). The paper motivates 3D conv
                      by local correlation in BOTH the spatial and the scale
                      neighbourhood, which requires a kernel > 1 in those axes;
                      hence the default (3, 3, 3). Set this to the value actually
                      used in training before releasing weights.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        kernel_size: Tuple[int, int, int] = (3, 3, 3),
        negative_slope: float = 0.1,
    ):
        super().__init__()
        self.num_levels = len(in_channels)
        self.align = nn.ModuleList(ConvBNAct(c, out_channels, 1) for c in in_channels)
        padding = tuple(k // 2 for k in kernel_size)
        self.conv3d = nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn3d = nn.BatchNorm3d(out_channels)
        self.act = nn.LeakyReLU(negative_slope, inplace=True)
        self.pool3d = nn.MaxPool3d(kernel_size=(self.num_levels, 1, 1))

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        assert len(feats) == self.num_levels, f"expected {self.num_levels} maps, got {len(feats)}"
        size = feats[0].shape[-2:]  # spatial size of the highest-resolution level (P2)
        volume = []
        for f, align in zip(feats, self.align):
            f = align(f)
            if f.shape[-2:] != size:
                f = F.interpolate(f, size=size, mode="nearest")
            volume.append(f.unsqueeze(2))  # (B, C, 1, H, W)
        x = torch.cat(volume, dim=2)       # (B, C, S, H, W)
        x = self.act(self.bn3d(self.conv3d(x)))
        x = self.pool3d(x)                 # (B, C, 1, H, W)
        return x.squeeze(2)                # (B, C, H, W)


# --------------------------------------------------------------------------- #
# 3.2.2 Triple feature mix (TFM)
# --------------------------------------------------------------------------- #
class TFM(nn.Module):
    """
    Resize a large (l, shallow), medium (m) and small (s, deep) map to the size of
    m and concatenate them along the channel axis.

      - l: conv reduces channels to `l_out_channels`, then hybrid pooling
           (adaptive max pool + adaptive avg pool, summed) down to m's size.
      - m: passed through unchanged.
      - s: conv adjusts channels to `s_out_channels`, then nearest-neighbour
           upsampling to m's size.
      - output: cat([l, m, s], dim=1)

    The paper text states that l is reduced "to a single channel", so the default
    `l_out_channels=1` follows the text literally. If the trained model kept more
    channels for l, set it here.

    Args:
        c_l, c_m, c_s:   input channels of the l, m, s maps.
        l_out_channels:  channels of l after reduction (paper text: 1).
        s_out_channels:  channels of s after adjustment (default: c_m).
    """

    def __init__(self, c_l: int, c_m: int, c_s: int, l_out_channels: int = 1, s_out_channels: int = None):
        super().__init__()
        s_out_channels = c_m if s_out_channels is None else s_out_channels
        self.l_reduce = ConvBNAct(c_l, l_out_channels, 1)
        self.s_adjust = ConvBNAct(c_s, s_out_channels, 1)
        self.out_channels = l_out_channels + c_m + s_out_channels

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        l, m, s = x
        size = m.shape[-2:]
        l = self.l_reduce(l)
        l = F.adaptive_max_pool2d(l, size) + F.adaptive_avg_pool2d(l, size)
        s = F.interpolate(self.s_adjust(s), size=size, mode="nearest")
        return torch.cat([l, m, s], dim=1)


# --------------------------------------------------------------------------- #
# 3.3 Dynamic-IoU loss
# --------------------------------------------------------------------------- #
def bbox_iou_xyxy(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Element-wise IoU of two (..., 4) tensors in (x1, y1, x2, y2) format."""
    inter_w = (torch.min(box1[..., 2], box2[..., 2]) - torch.max(box1[..., 0], box2[..., 0])).clamp(min=0)
    inter_h = (torch.min(box1[..., 3], box2[..., 3]) - torch.max(box1[..., 1], box2[..., 1])).clamp(min=0)
    inter = inter_w * inter_h
    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    return inter / (area1 + area2 - inter + eps)


class DynamicIoULoss(nn.Module):
    """
    L_Dynamic-IoU = L_PIoU + IoU - IoU_focaler                         (Eq. 6)

      P          = (dw1/w_gt + dw2/w_gt + dh1/h_gt + dh2/h_gt) / 4     (Eq. 1)
      f(x)       = 1 - exp(-x^2)                                       (Eq. 2)
      L_PIoU     = 1 - IoU + f(P)                                      (Eq. 3-4)
      IoU_focaler= 0 if IoU < d; (IoU-d)/(u-d) if d <= IoU <= u; 1 if IoU > u   (Eq. 5)

    dw1, dw2, dh1, dh2 are absolute distances between corresponding edges of the
    predicted and ground-truth boxes. Note that L_PIoU lies in [0, 2] (IoU in [0,1],
    f in [0,1)), so the total loss is not bounded by 1.

    Args:
        d, u:      interval of the Focaler linear mapping, 0 <= d < u <= 1.
                   The paper does not report the values used; the defaults
                   (d=0.0, u=0.95) are the Focaler-IoU paper's defaults and
                   should be replaced by the values used in the experiments.
        reduction: "none" | "mean" | "sum".

    Inputs:
        pred, target: (..., 4) boxes in (x1, y1, x2, y2), same coordinate frame.
    """

    def __init__(self, d: float = 0.0, u: float = 0.95, reduction: str = "none", eps: float = 1e-7):
        super().__init__()
        assert 0.0 <= d < u <= 1.0, "require 0 <= d < u <= 1"
        assert reduction in ("none", "mean", "sum")
        self.d, self.u, self.reduction, self.eps = d, u, reduction, eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        iou = bbox_iou_xyxy(pred, target, self.eps)

        w_gt = (target[..., 2] - target[..., 0]).clamp(min=self.eps)
        h_gt = (target[..., 3] - target[..., 1]).clamp(min=self.eps)
        dw1 = (pred[..., 0] - target[..., 0]).abs()
        dw2 = (pred[..., 2] - target[..., 2]).abs()
        dh1 = (pred[..., 1] - target[..., 1]).abs()
        dh2 = (pred[..., 3] - target[..., 3]).abs()
        P = (dw1 / w_gt + dw2 / w_gt + dh1 / h_gt + dh2 / h_gt) / 4.0

        loss_piou = 1.0 - iou + (1.0 - torch.exp(-P ** 2))
        iou_focaler = ((iou - self.d) / (self.u - self.d)).clamp(0.0, 1.0)
        loss = loss_piou + iou - iou_focaler

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# --------------------------------------------------------------------------- #
# Shape sanity check
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)
    p2, p3, p4 = torch.randn(2, 64, 160, 160), torch.randn(2, 128, 80, 80), torch.randn(2, 256, 40, 40)

    msfc = MSFC(in_channels=(64, 128, 256), out_channels=64)
    print("MSFC:", tuple(msfc([p2, p3, p4]).shape))           # (2, 64, 160, 160)

    tfm = TFM(c_l=64, c_m=128, c_s=256)
    print("TFM :", tuple(tfm([p2, p3, p4]).shape), "out_channels =", tfm.out_channels)  # (2, 257, 80, 80)

    gt = torch.tensor([[10.0, 10.0, 14.0, 14.0]])
    pr = gt + 2.0  # 4x4 box shifted 2 px along the diagonal (Fig. 8 example)
    print("IoU :", bbox_iou_xyxy(pr, gt).item())               # ~0.1429 (4/28)
    print("Loss:", DynamicIoULoss()(pr, gt).item())
