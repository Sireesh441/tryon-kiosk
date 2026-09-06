"""
coco_person_dataset.py

PyTorch Dataset for the person-only COCO val2017 subset.

Loads an image + its person bounding boxes, resizes both to a fixed
size, and builds the training targets our anchor-free detector head
expects:
  - a heatmap: 1 at each box center, 0 elsewhere (on a downsampled grid)
  - a box-size map: distance from each "inside-a-box" grid cell to the
    box's left/top/right/bottom edges

Usage (quick smoke test):
    py coco_person_dataset.py
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.functional import resize

# --- Config ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
IMAGES_DIR = PROJECT_ROOT / "data" / "coco" / "val2017"
ANNOTATIONS_PATH = PROJECT_ROOT / "data" / "coco" / "annotations" / "instances_val2017_person.json"

INPUT_SIZE = 512          # resize every image to INPUT_SIZE x INPUT_SIZE
STRIDE = 8                # how much the backbone downsamples (grid = INPUT_SIZE / STRIDE)
GRID_SIZE = INPUT_SIZE // STRIDE  # 64x64 grid of prediction cells


class CocoPersonDataset(Dataset):
    def __init__(self, images_dir=IMAGES_DIR, annotations_path=ANNOTATIONS_PATH):
        self.images_dir = Path(images_dir)

        with open(annotations_path, "r") as f:
            coco = json.load(f)

        # Map image_id -> image metadata (filename, width, height)
        self.image_id_to_info = {img["id"]: img for img in coco["images"]}

        # Group annotations by image_id: image_id -> list of [x, y, w, h] boxes
        self.image_id_to_boxes = {}
        for ann in coco["annotations"]:
            self.image_id_to_boxes.setdefault(ann["image_id"], []).append(ann["bbox"])

        # Final ordered list of image_ids we'll actually iterate over
        self.image_ids = sorted(self.image_id_to_boxes.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        info = self.image_id_to_info[image_id]
        orig_w, orig_h = info["width"], info["height"]

        # --- Load and resize image ---
        image_path = self.images_dir / info["file_name"]
        image = read_image(str(image_path)).float() / 255.0  # (C, H, W), normalized 0-1

        if image.shape[0] == 1:  # some COCO images are grayscale
            image = image.repeat(3, 1, 1)

        image = resize(image, [INPUT_SIZE, INPUT_SIZE], antialias=True)

        # --- Scale boxes from original image size to INPUT_SIZE ---
        scale_x = INPUT_SIZE / orig_w
        scale_y = INPUT_SIZE / orig_h

        boxes_xyxy = []
        for (x, y, w, h) in self.image_id_to_boxes[image_id]:
            x1, y1 = x * scale_x, y * scale_y
            x2, y2 = (x + w) * scale_x, (y + h) * scale_y
            boxes_xyxy.append([x1, y1, x2, y2])

        # --- Build training targets on the downsampled grid ---
        heatmap = torch.zeros((1, GRID_SIZE, GRID_SIZE))
        box_targets = torch.zeros((4, GRID_SIZE, GRID_SIZE))  # left, top, right, bottom distances
        mask = torch.zeros((1, GRID_SIZE, GRID_SIZE))  # 1 where a box center falls

        for (x1, y1, x2, y2) in boxes_xyxy:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            grid_x, grid_y = int(cx / STRIDE), int(cy / STRIDE)

            if 0 <= grid_x < GRID_SIZE and 0 <= grid_y < GRID_SIZE:
                heatmap[0, grid_y, grid_x] = 1.0
                mask[0, grid_y, grid_x] = 1.0

                # distances from this cell's center (in input-image pixels) to each edge
                cell_center_x = grid_x * STRIDE + STRIDE / 2
                cell_center_y = grid_y * STRIDE + STRIDE / 2
                box_targets[0, grid_y, grid_x] = cell_center_x - x1  # left
                box_targets[1, grid_y, grid_x] = cell_center_y - y1  # top
                box_targets[2, grid_y, grid_x] = x2 - cell_center_x  # right
                box_targets[3, grid_y, grid_x] = y2 - cell_center_y  # bottom

        return {
            "image": image,              # (3, 512, 512)
            "heatmap": heatmap,           # (1, 64, 64) - target for confidence head
            "box_targets": box_targets,   # (4, 64, 64) - target for box-size head
            "mask": mask,                 # (1, 64, 64) - which cells have a real box (for loss)
        }


if __name__ == "__main__":
    print("Running smoke test on CocoPersonDataset...")
    dataset = CocoPersonDataset()
    print(f"Dataset size: {len(dataset)} images")

    sample = dataset[0]
    print(f"image shape: {sample['image'].shape}")
    print(f"heatmap shape: {sample['heatmap'].shape}, sum: {sample['heatmap'].sum().item()}")
    print(f"box_targets shape: {sample['box_targets'].shape}")
    print(f"mask shape: {sample['mask'].shape}, active cells: {sample['mask'].sum().item()}")

    # Sanity check a few samples for reasonable heatmap activity
    total_active_cells = 0
    for i in range(min(20, len(dataset))):
        total_active_cells += dataset[i]["mask"].sum().item()
    print(f"\nAcross first 20 images: {total_active_cells} total active grid cells (should be > 0)")

    print("\nSmoke test passed if no errors above and active cells > 0.")
