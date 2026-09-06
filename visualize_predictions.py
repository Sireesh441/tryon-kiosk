"""
visualize_predictions.py

Loads a trained PersonDetector checkpoint, runs it on a single image,
decodes the heatmap+box predictions back into actual bounding boxes,
and draws them on the image so you can visually judge detection quality.

Usage:
    py visualize_predictions.py --checkpoint checkpoints/model_epoch5.pt --image data/coco/val2017/000000397133.jpg
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.io import read_image, write_png
from torchvision.transforms.functional import resize
from torchvision.utils import draw_bounding_boxes

from detector_model import PersonDetector

INPUT_SIZE = 512
STRIDE = 8
GRID_SIZE = INPUT_SIZE // STRIDE


def compute_iou(box_a, box_b):
    """IoU between two boxes in (x1, y1, x2, y2) format."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def non_max_suppression(detections, iou_threshold=0.5):
    """
    detections: list of (x1, y1, x2, y2, score), already sorted by score descending.
    Keeps the highest-scoring box in each overlapping cluster, discards the rest.
    """
    kept = []
    remaining = list(detections)

    while remaining:
        best = remaining.pop(0)
        kept.append(best)
        remaining = [
            d for d in remaining
            if compute_iou(best[:4], d[:4]) < iou_threshold
        ]

    return kept


def decode_predictions(heatmap_logits, box_preds, confidence_threshold=0.3, max_detections=50):
    """
    Converts raw model output into a list of (x1, y1, x2, y2, score) boxes
    in INPUT_SIZE (512x512) pixel coordinates.

    Simple approach: threshold the heatmap, take each surviving cell as a
    detection (no NMS yet -- fine for a first look at output quality; we
    can add NMS once we're seeing real predictions worth de-duplicating).
    """
    heatmap = torch.sigmoid(heatmap_logits)[0, 0]  # (64, 64), drop batch/channel dims
    boxes = box_preds[0]  # (4, 64, 64)

    detections = []
    grid_ys, grid_xs = torch.where(heatmap > confidence_threshold)

    for gy, gx in zip(grid_ys.tolist(), grid_xs.tolist()):
        score = heatmap[gy, gx].item()
        left, top, right, bottom = boxes[:, gy, gx].tolist()

        cell_center_x = gx * STRIDE + STRIDE / 2
        cell_center_y = gy * STRIDE + STRIDE / 2

        x1 = cell_center_x - left
        y1 = cell_center_y - top
        x2 = cell_center_x + right
        y2 = cell_center_y + bottom

        detections.append((x1, y1, x2, y2, score))

    # Keep only the top-N by confidence, in case of many low-quality detections early in training
    detections.sort(key=lambda d: d[4], reverse=True)
    detections = non_max_suppression(detections, iou_threshold=0.5)
    return detections[:max_detections]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to a saved model checkpoint (.pt)")
    parser.add_argument("--image", required=True, help="Path to an image to run detection on")
    parser.add_argument("--confidence", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--output", default="prediction_output.png", help="Where to save the visualized result")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading checkpoint from {args.checkpoint} ...")
    model = PersonDetector().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    print(f"Loading image from {args.image} ...")
    original_image = read_image(args.image)
    if original_image.shape[0] == 1:
        original_image = original_image.repeat(3, 1, 1)

    image_for_model = original_image.float() / 255.0
    image_for_model = resize(image_for_model, [INPUT_SIZE, INPUT_SIZE], antialias=True)
    image_for_model = image_for_model.unsqueeze(0).to(device)  # add batch dim

    print("Running inference...")
    with torch.no_grad():
        heatmap_logits, box_preds = model(image_for_model)

    detections = decode_predictions(heatmap_logits, box_preds, confidence_threshold=args.confidence)
    print(f"Found {len(detections)} detections above confidence {args.confidence}")

    if len(detections) == 0:
        print("No detections above threshold -- try lowering --confidence, or the model may need more training.")

    # Draw boxes on the resized 512x512 image (matches the coordinate space of our predictions)
    display_image = (resize(original_image, [INPUT_SIZE, INPUT_SIZE], antialias=True))

    if detections:
        boxes_tensor = torch.tensor([[x1, y1, x2, y2] for x1, y1, x2, y2, score in detections])
        labels = [f"{score:.2f}" for _, _, _, _, score in detections]
        display_image = draw_bounding_boxes(display_image, boxes_tensor, labels=labels, colors="red", width=2)

    write_png(display_image, args.output)
    print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    main()
