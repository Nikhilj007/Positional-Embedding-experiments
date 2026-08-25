"""
Training script for Modified ViT (2D Rotary Position Embedding) on CIFAR-100.

Usage:
    python modified_vit/train.py [--epochs 200] [--batch_size 128] [--lr 1e-3]
"""

import os
import sys
import argparse
import math
import time
import torch
import torch.nn as nn
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modified_vit.model import ViT, count_parameters
from original_vit.utils import (
    set_seed, get_cifar100_loaders, Mixup, SoftTargetCrossEntropy,
    MetricLogger, accuracy
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Modified ViT (2D RoPE) on CIFAR-100")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=10, help="Warmup epochs")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--save_dir", type=str, default="./results", help="Save directory")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--dry_run", action="store_true", help="Quick test run")
    return parser.parse_args()


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs

        progress = (
            epoch - warmup_epochs
        ) / (total_epochs - warmup_epochs)

        return 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda
    )


def train_one_epoch(model, loader, criterion, optimizer, mixup_fn, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)

        # Apply Mixup/CutMix
        images_mixed, targets_mixed = mixup_fn(images, targets)

        # Forward pass
        outputs = model(images_mixed)
        loss = criterion(outputs, targets_mixed)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        # Accuracy on original (un-mixed) targets
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / total
    avg_acc = 100.0 * correct / total
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate on validation/test set."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / total
    avg_acc = 100.0 * correct / total
    return avg_loss, avg_acc


def main():
    args = parse_args()
    set_seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Data
    print("Loading CIFAR-100...")
    train_loader, val_loader, test_loader = get_cifar100_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers, data_dir=args.data_dir
    )
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val:   {len(val_loader.dataset)} samples")
    print(f"  Test:  {len(test_loader.dataset)} samples")

    # Model
    model = ViT(
        img_size=32, patch_size=4, in_channels=3, num_classes=100,
        embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, dropout=args.dropout
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"\nModified ViT (2D Rotary Position Embedding)")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Override epochs early so scheduler sees the correct total
    if args.dry_run:
        args.epochs = 2

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_epochs, args.epochs)

    # Loss and augmentation
    criterion = SoftTargetCrossEntropy()
    mixup_fn = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, num_classes=100, label_smoothing=0.1)

    # Metric logger
    logger = MetricLogger()
    best_val_acc = 0.0
    os.makedirs(args.save_dir, exist_ok=True)

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, mixup_fn, device
        )

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, device)

        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Log metrics
        logger.update({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "lr": current_lr,
        })

        epoch_time = time.time() - epoch_start

        # Print progress
        # NOTE: train_acc is approximate — predictions are from mixed inputs
        # but compared against original (un-mixed) labels.
        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc (approx): {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            }, os.path.join(args.save_dir, "modified_vit_best.pth"))
            print(f"  -> New best val accuracy: {val_acc:.2f}%")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.1f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

    # Load best model and evaluate on test set
    print("\nEvaluating best model on test set...")
    checkpoint = torch.load(
        os.path.join(args.save_dir, "modified_vit_best.pth"),
        map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc = evaluate(model, test_loader, device)
    logger.set_test_acc(test_acc)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy (Top-1): {test_acc:.2f}%")

    # Save metrics
    logger.save(os.path.join(args.save_dir, "modified_metrics.json"))

    print("\nDone!")


if __name__ == "__main__":
    main()
