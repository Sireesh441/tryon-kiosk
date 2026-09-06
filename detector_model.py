"""
detector_model.py

A small anchor-free, single-class (person) object detector.

Architecture:
  - Lightweight CNN backbone, downsamples 512x512 -> 64x64 feature grid (stride 8)
  - Detection head: two branches on top of backbone features
      1. Heatmap branch: 1 channel, sigmoid -> "person center likelihood" per cell
      2. Box branch: 4 channels -> distance to left/top/right/bottom edges per cell

Loss:
  - Focal loss on the heatmap (handles background/foreground class imbalance)
  - IoU loss on box predictions, computed only at cells with a real person center

Usage (quick smoke test, random input, no real data needed):
    py detector_model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_channels, out_channels, stride=1):
    """Conv -> BatchNorm -> ReLU, the basic building block of the backbone."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class Backbone(nn.Module):
    """
    Downsamples 512x512x3 -> 64x64xC (stride 8 total: 2 * 2 * 2).
    Kept small deliberately -- this is a starting point to validate the
    pipeline, not a final architecture. We can deepen/widen later once
    training end-to-end is confirmed working.
    """

    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 32, stride=2),   # 512 -> 256
        )
        self.stage2 = nn.Sequential(
            conv_block(32, 64),
            conv_block(64, 64, stride=2),   # 256 -> 128
        )
        self.stage3 = nn.Sequential(
            conv_block(64, 128),
            conv_block(128, 128, stride=2),  # 128 -> 64
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x  # (B, 128, 64, 64)


class DetectionHead(nn.Module):
    """Two small conv branches: heatmap (1 channel) and box distances (4 channels)."""

    def __init__(self, in_channels=128):
        super().__init__()
        self.heatmap_branch = nn.Sequential(
            conv_block(in_channels, 64),
            nn.Conv2d(64, 1, kernel_size=1),  # raw logits, sigmoid applied outside
        )
        self.box_branch = nn.Sequential(
            conv_block(in_channels, 64),
            nn.Conv2d(64, 4, kernel_size=1),
            nn.ReLU(inplace=True),  # distances must be non-negative
        )

    def forward(self, features):
        heatmap_logits = self.heatmap_branch(features)  # (B, 1, 64, 64)
        box_preds = self.box_branch(features)            # (B, 4, 64, 64)
        return heatmap_logits, box_preds


class PersonDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.head = DetectionHead(in_channels=128)

    def forward(self, x):
        features = self.backbone(x)
        heatmap_logits, box_preds = self.head(features)
        return heatmap_logits, box_preds


def focal_loss(heatmap_logits, heatmap_targets, alpha=2.0, beta=4.0):
    """
    Standard heatmap focal loss (CornerNet/CenterNet-style).
    heatmap_logits: raw model output (B, 1, H, W), pre-sigmoid
    heatmap_targets: 0/1 target heatmap (B, 1, H, W)
    """
    pred = torch.sigmoid(heatmap_logits).clamp(min=1e-4, max=1 - 1e-4)

    pos_mask = (heatmap_targets == 1).float()
    neg_mask = (heatmap_targets != 1).float()

    pos_loss = -torch.log(pred) * torch.pow(1 - pred, alpha) * pos_mask
    neg_loss = -torch.log(1 - pred) * torch.pow(pred, alpha) * torch.pow(1 - heatmap_targets, beta) * neg_mask

    num_pos = pos_mask.sum().clamp(min=1)  # avoid divide-by-zero on images with no boxes
    return (pos_loss.sum() + neg_loss.sum()) / num_pos


def iou_loss(box_preds, box_targets, mask):
    """
    IoU loss between predicted and target box distances (left, top, right, bottom),
    computed only at cells where mask == 1 (i.e. a real person center).
    """
    pred_left, pred_top, pred_right, pred_bottom = box_preds.unbind(dim=1)
    tgt_left, tgt_top, tgt_right, tgt_bottom = box_targets.unbind(dim=1)

    pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)
    tgt_area = (tgt_left + tgt_right) * (tgt_top + tgt_bottom)

    inter_w = torch.min(pred_left, tgt_left) + torch.min(pred_right, tgt_right)
    inter_h = torch.min(pred_top, tgt_top) + torch.min(pred_bottom, tgt_bottom)
    inter_area = inter_w.clamp(min=0) * inter_h.clamp(min=0)

    union_area = pred_area + tgt_area - inter_area
    iou = inter_area / union_area.clamp(min=1e-6)

    loss = 1 - iou  # (B, H, W)
    mask = mask.squeeze(1)  # (B, H, W)

    num_pos = mask.sum().clamp(min=1)
    return (loss * mask).sum() / num_pos


def detector_loss(heatmap_logits, box_preds, heatmap_targets, box_targets, mask,
                   box_loss_weight=0.1):
    """
    box_loss_weight: down-weights the box/IoU loss relative to the heatmap
    loss. Needed because heatmap focal loss sums over all grid cells
    (thousands, mostly background) while box loss only sums over positive
    cells (a handful) -- without this weight, box loss barely influences
    training even when it's the dominant error source.

    0.1 is the standard starting point from CenterNet-style detectors;
    treat it as a hyperparameter to revisit once real training starts.
    """
    h_loss = focal_loss(heatmap_logits, heatmap_targets)
    b_loss = iou_loss(box_preds, box_targets, mask)
    total = h_loss + box_loss_weight * b_loss
    return total, h_loss, b_loss


if __name__ == "__main__":
    print("Running smoke test on PersonDetector...")

    model = PersonDetector()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created. Total parameters: {num_params:,}")

    # Fake a batch: 2 images, random pixels, random targets
    batch_size = 2
    fake_images = torch.rand(batch_size, 3, 512, 512)
    fake_heatmap = torch.zeros(batch_size, 1, 64, 64)
    fake_heatmap[:, 0, 32, 32] = 1.0  # pretend one person center per image
    fake_box_targets = torch.rand(batch_size, 4, 64, 64) * 50
    fake_mask = fake_heatmap.clone()

    heatmap_logits, box_preds = model(fake_images)
    print(f"heatmap_logits shape: {heatmap_logits.shape}")
    print(f"box_preds shape: {box_preds.shape}")

    total_loss, h_loss, b_loss = detector_loss(
        heatmap_logits, box_preds, fake_heatmap, fake_box_targets, fake_mask
    )
    print(f"total_loss: {total_loss.item():.4f}")
    print(f"heatmap_loss: {h_loss.item():.4f}")
    print(f"box_loss: {b_loss.item():.4f}")

    # Confirm backward pass works (gradients flow)
    total_loss.backward()
    print("\nBackward pass succeeded (gradients computed, no errors).")
    print("Smoke test passed.")
