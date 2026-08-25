"""
Comparison script: Generate overlaid loss curves and print test accuracies.

Usage:
    python compare.py [--results_dir ./results]
"""

import os
import json
import argparse
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for cluster
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Compare Original vs Modified ViT results")
    parser.add_argument("--results_dir", type=str, default="./results", help="Results directory")
    return parser.parse_args()


def load_metrics(path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    args = parse_args()

    # Load metrics
    original_path = os.path.join(args.results_dir, "original_metrics.json")
    modified_path = os.path.join(args.results_dir, "modified_metrics.json")

    if not os.path.exists(original_path):
        print(f"ERROR: {original_path} not found. Train the original ViT first.")
        return
    if not os.path.exists(modified_path):
        print(f"ERROR: {modified_path} not found. Train the modified ViT first.")
        return

    original = load_metrics(original_path)
    modified = load_metrics(modified_path)

    epochs = range(1, len(original["train_loss"]) + 1)

    # ================================================================
    # Print Test Accuracies
    # ================================================================
    print("=" * 60)
    print("              ViT Positional Embedding Comparison")
    print("=" * 60)
    print(f"  Original ViT (1D Learned)     — Top-1 Test Acc: {original['test_acc']:.2f}%")
    print(f"  Modified ViT (2D RoPE)        — Top-1 Test Acc: {modified['test_acc']:.2f}%")
    print(f"  Difference                    — {modified['test_acc'] - original['test_acc']:+.2f}%")
    print("=" * 60)

    # ================================================================
    # Plot Loss Curves
    # ================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Training Loss ---
    ax = axes[0]
    ax.plot(epochs, original["train_loss"], label="Original ViT (1D Learned)", color="#2196F3",
            linewidth=1.5, alpha=0.9)
    ax.plot(epochs, modified["train_loss"], label="Modified ViT (2D RoPE)", color="#FF5722",
            linewidth=1.5, alpha=0.9)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Training Loss", fontsize=12)
    ax.set_title("Training Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- Validation Loss ---
    ax = axes[1]
    ax.plot(epochs, original["val_loss"], label="Original ViT (1D Learned)", color="#2196F3",
            linewidth=1.5, alpha=0.9)
    ax.plot(epochs, modified["val_loss"], label="Modified ViT (2D RoPE)", color="#FF5722",
            linewidth=1.5, alpha=0.9)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation Loss", fontsize=12)
    ax.set_title("Validation Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        "ViT Positional Embedding Comparison on CIFAR-100\n"
        f"Original (1D Learned): {original['test_acc']:.2f}%  |  "
        f"Modified (2D RoPE): {modified['test_acc']:.2f}%",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    # Save
    plot_path = os.path.join(args.results_dir, "loss_curves.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nLoss curves saved to: {plot_path}")

    # ================================================================
    # Also plot accuracy curves
    # ================================================================
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes2[0]
    ax.plot(epochs, original["train_acc"], label="Original ViT (1D Learned)", color="#2196F3",
            linewidth=1.5, alpha=0.9)
    ax.plot(epochs, modified["train_acc"], label="Modified ViT (2D RoPE)", color="#FF5722",
            linewidth=1.5, alpha=0.9)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Training Accuracy (%)", fontsize=12)
    ax.set_title("Training Accuracy", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes2[1]
    ax.plot(epochs, original["val_acc"], label="Original ViT (1D Learned)", color="#2196F3",
            linewidth=1.5, alpha=0.9)
    ax.plot(epochs, modified["val_acc"], label="Modified ViT (2D RoPE)", color="#FF5722",
            linewidth=1.5, alpha=0.9)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=12)
    ax.set_title("Validation Accuracy", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        "ViT Accuracy Comparison on CIFAR-100",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    acc_path = os.path.join(args.results_dir, "accuracy_curves.png")
    fig2.savefig(acc_path, dpi=150, bbox_inches="tight")
    print(f"Accuracy curves saved to: {acc_path}")


if __name__ == "__main__":
    main()
