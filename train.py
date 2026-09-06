"""
train.py

Wires together CocoPersonDataset + PersonDetector + detector_loss into
an actual training loop.

This is meant to be run in two modes:
  1. CPU smoke run (default): trains on a SMALL subset for a few epochs,
     just to confirm the loss goes down and nothing is broken. Slow but
     cheap -- do this before spending RunPod GPU credits.
  2. Full run (later): once the smoke run confirms things work, we'll
     adjust NUM_EPOCHS / SUBSET_SIZE and move this to the GPU.

Usage:
    py train.py
"""

import time

import torch
from torch.utils.data import DataLoader, Subset

from coco_person_dataset import CocoPersonDataset
from detector_model import PersonDetector, detector_loss

# --- Config ---------------------------------------------------------------
SUBSET_SIZE = 64        # small subset for a fast CPU smoke run
BATCH_SIZE = 4
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print(f"Using device: {DEVICE}")

    print("Loading dataset...")
    full_dataset = CocoPersonDataset()
    subset_indices = list(range(min(SUBSET_SIZE, len(full_dataset))))
    dataset = Subset(full_dataset, subset_indices)
    print(f"Training on a subset of {len(dataset)} images (out of {len(full_dataset)} total)")

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = PersonDetector().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\nStarting training for {NUM_EPOCHS} epochs...\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        model.train()

        total_loss_sum = 0.0
        heatmap_loss_sum = 0.0
        box_loss_sum = 0.0
        num_batches = 0

        for batch in dataloader:
            images = batch["image"].to(DEVICE)
            heatmap_targets = batch["heatmap"].to(DEVICE)
            box_targets = batch["box_targets"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)

            heatmap_logits, box_preds = model(images)
            total_loss, h_loss, b_loss = detector_loss(
                heatmap_logits, box_preds, heatmap_targets, box_targets, mask
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_loss_sum += total_loss.item()
            heatmap_loss_sum += h_loss.item()
            box_loss_sum += b_loss.item()
            num_batches += 1

        avg_total = total_loss_sum / num_batches
        avg_heatmap = heatmap_loss_sum / num_batches
        avg_box = box_loss_sum / num_batches
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch}/{NUM_EPOCHS} | "
            f"total_loss: {avg_total:.4f} | "
            f"heatmap_loss: {avg_heatmap:.4f} | "
            f"box_loss: {avg_box:.4f} | "
            f"time: {epoch_time:.1f}s"
        )

    print("\nTraining smoke run complete.")
    print("Check above: total_loss should trend downward across epochs.")
    print("If it does, the pipeline (data -> model -> loss -> optimizer) is verified working.")

    import os
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "epoch": NUM_EPOCHS,
        "model_state_dict": model.state_dict(),
    }, f"checkpoints/model_epoch{NUM_EPOCHS}.pt")
    print(f"Saved checkpoint to checkpoints/model_epoch{NUM_EPOCHS}.pt")


if __name__ == "__main__":
    main()
