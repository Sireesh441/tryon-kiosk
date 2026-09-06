"""
train.py

Wires together CocoPersonDataset + PersonDetector + detector_loss into
an actual training loop.

This is meant to be run in two modes:
  1. CPU smoke run: trains on a SMALL subset (SUBSET_SIZE) for a few
     epochs, just to confirm the loss goes down and nothing is broken.
     Slow but cheap -- do this before spending RunPod GPU credits.
  2. Full run (current default): SUBSET_SIZE=None trains on the entire
     person-filtered dataset (2,693 val2017 images) for NUM_EPOCHS=50 on
     the GPU. 50 is a starting point for a small (~450K param) detector
     on a few thousand images -- CenterNet-style anchor-free detectors
     this size typically need tens of epochs to show real convergence;
     revisit upward if loss is still dropping steadily at epoch 50.

Usage:
    py train.py
"""

import os
import time

import torch
from torch.utils.data import DataLoader, Subset

from coco_person_dataset import CocoPersonDataset
from detector_model import PersonDetector, detector_loss

# --- Config ---------------------------------------------------------------
SUBSET_SIZE = None      # None = full dataset; set to an int for a fast CPU smoke run
BATCH_SIZE = 4
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
CHECKPOINT_EVERY = 10   # also save a checkpoint every N epochs, not just at the end
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print(f"Using device: {DEVICE}")

    print("Loading dataset...")
    full_dataset = CocoPersonDataset()
    if SUBSET_SIZE is None:
        dataset = full_dataset
    else:
        subset_indices = list(range(min(SUBSET_SIZE, len(full_dataset))))
        dataset = Subset(full_dataset, subset_indices)
    print(f"Training on {len(dataset)} images (out of {len(full_dataset)} total)")

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = PersonDetector().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    os.makedirs("checkpoints", exist_ok=True)

    def save_checkpoint(epoch):
        path = f"checkpoints/model_epoch{epoch}.pt"
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, path)
        print(f"Saved checkpoint to {path}")

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

        if epoch % CHECKPOINT_EVERY == 0:
            save_checkpoint(epoch)

    print("\nTraining run complete.")
    print("Check above: total_loss should trend downward across epochs.")
    print("If it does, the pipeline (data -> model -> loss -> optimizer) is verified working.")

    save_checkpoint(NUM_EPOCHS)


if __name__ == "__main__":
    main()
