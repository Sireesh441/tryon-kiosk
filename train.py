"""
train.py

Wires together CocoPersonDataset + PersonDetector + detector_loss into
an actual training loop.

This is meant to be run in three modes, toggled via SPLIT/SUBSET_SIZE/
NUM_EPOCHS below:
  1. CPU smoke run: SPLIT="val2017", small SUBSET_SIZE, few epochs --
     confirms the loss goes down and nothing is broken. Slow but cheap --
     do this before spending RunPod GPU credits.
  2. val2017 full run: SPLIT="val2017", SUBSET_SIZE=None -- trains on the
     full 2,693-image person-filtered val set. Good for validating the
     pipeline end-to-end on GPU before committing to the much larger
     train2017 run.
  3. train2017 full run (current default): SPLIT="train2017",
     SUBSET_SIZE=None -- the real, full-scale training run.

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
SPLIT = "train2017"     # "val2017" or "train2017" -- which CocoPersonDataset split to train on
SUBSET_SIZE = None      # None = full dataset; set to an int for a fast CPU smoke run
BATCH_SIZE = 4
NUM_EPOCHS = 15         # see the printed time estimate before committing to a full train2017 run
LEARNING_RATE = 1e-3
CHECKPOINT_EVERY = 1    # save a checkpoint every N epochs, not just at the end
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print(f"Using device: {DEVICE}")

    print(f"Loading dataset (split={SPLIT})...")
    full_dataset = CocoPersonDataset(split=SPLIT)
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
        percent_complete = 100 * epoch / NUM_EPOCHS

        print(
            f"Epoch {epoch}/{NUM_EPOCHS} ({percent_complete:.1f}% complete) | "
            f"total_loss: {avg_total:.4f} | "
            f"heatmap_loss: {avg_heatmap:.4f} | "
            f"box_loss: {avg_box:.4f} | "
            f"time: {epoch_time:.1f}s"
        )

        if epoch == 1:
            estimated_total_seconds = epoch_time * NUM_EPOCHS
            print(
                f"  -> Based on epoch 1's time, estimated total training time: "
                f"{estimated_total_seconds / 3600:.1f} hours "
                f"({estimated_total_seconds / 60:.0f} minutes) for all {NUM_EPOCHS} epochs."
            )

        if epoch % CHECKPOINT_EVERY == 0:
            save_checkpoint(epoch)

    print("\nTraining run complete.")
    print("Check above: total_loss should trend downward across epochs.")
    print("If it does, the pipeline (data -> model -> loss -> optimizer) is verified working.")

    save_checkpoint(NUM_EPOCHS)


if __name__ == "__main__":
    main()
