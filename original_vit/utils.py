"""
Utility functions shared between Original ViT and Modified ViT.
Includes: data loading, augmentation, Mixup/CutMix, training helpers, metric logging.
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cifar100_loaders(batch_size=128, num_workers=4, val_split=0.1, data_dir="./data"):
    """
    Create CIFAR-100 train/val/test data loaders.

    Train set is split into train (90%) and validation (10%).
    Test set is kept separate for final evaluation.
    """
    # Training augmentations
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        ),
    ])

    # Validation/test transforms (no augmentation)
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        ),
    ])

    # Download and load CIFAR-100
    full_train_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=eval_transform
    )

    # Split training set into train and validation
    num_train = len(full_train_dataset)
    num_val = int(num_train * val_split)
    num_train = num_train - num_val

    # Create a validation dataset with eval transforms
    val_dataset_base = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=False, transform=eval_transform
    )

    # Use same indices for the split
    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices = random_split(
        range(len(full_train_dataset)), [num_train, num_val], generator=generator
    )

    train_subset = torch.utils.data.Subset(full_train_dataset, train_indices.indices)
    val_subset = torch.utils.data.Subset(val_dataset_base, val_indices.indices)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


class Mixup:
    """Mixup and CutMix augmentation for training."""

    def __init__(self, mixup_alpha=0.8, cutmix_alpha=1.0, prob=0.5, num_classes=100,
                 label_smoothing=0.1):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

    def __call__(self, images, targets):
        # Convert targets to one-hot with label smoothing
        targets_onehot = torch.zeros(targets.size(0), self.num_classes, device=targets.device)
        targets_onehot.fill_(self.label_smoothing / self.num_classes)
        targets_onehot.scatter_(
            1, targets.unsqueeze(1),
            1.0 - self.label_smoothing + self.label_smoothing / self.num_classes
        )

        if random.random() > self.prob:
            # Apply CutMix
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            rand_index = torch.randperm(images.size(0), device=images.device)

            bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
            images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]

            # Adjust lambda based on actual area ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))
            targets_onehot = lam * targets_onehot + (1 - lam) * targets_onehot[rand_index]
        else:
            # Apply Mixup
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            rand_index = torch.randperm(images.size(0), device=images.device)

            images = lam * images + (1 - lam) * images[rand_index]
            targets_onehot = lam * targets_onehot + (1 - lam) * targets_onehot[rand_index]

        return images, targets_onehot

    @staticmethod
    def _rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2


class SoftTargetCrossEntropy(nn.Module):
    """Cross-entropy loss for soft/mixed targets."""

    def forward(self, logits, targets):
        loss = -targets * torch.nn.functional.log_softmax(logits, dim=-1)
        return loss.sum(dim=-1).mean()


class MetricLogger:
    """Log and save training metrics."""

    def __init__(self):
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "test_acc": None,
            "lr": [],
        }

    def update(self, epoch_metrics):
        for key in ["train_loss", "val_loss", "train_acc", "val_acc", "lr"]:
            if key in epoch_metrics:
                self.metrics[key].append(epoch_metrics[key])

    def set_test_acc(self, acc):
        self.metrics["test_acc"] = acc

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Metrics saved to {path}")

    def load(self, path):
        with open(path, "r") as f:
            self.metrics = json.load(f)
        return self.metrics


def accuracy(output, target, topk=(1,)):
    """Compute top-k accuracy."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res
