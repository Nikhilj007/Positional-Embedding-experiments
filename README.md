# ViT Positional Embedding Comparison: Original vs. 2D RoPE

## Overview

This project trains two variants of Vision Transformer (ViT) from scratch on **CIFAR-100** and compares their performance:

1. **Original ViT** — 1D learnable positional embeddings (Dosovitskiy et al., ICLR 2021)
2. **Modified ViT** — 2D Rotary Position Embedding (RoPE)

**Reference Paper:** Dosovitskiy et al., *"An Image Is Worth 16×16 Words: Transformers for Image Recognition at Scale"*, ICLR 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

---

## Architecture (Identical for Both Variants)

| Parameter       | Value          |
|----------------|----------------|
| Image Size      | 32×32          |
| Patch Size      | 4×4            |
| Num Patches     | 64 (8×8 grid)  |
| Embedding Dim   | 192            |
| Depth           | 12 layers      |
| Attention Heads | 3              |
| MLP Ratio       | 4× (hidden=768)|
| Dropout         | 0.1            |
| Num Classes     | 100            |

---

## Project Structure

```
Assignment1/
├── original_vit/
│   ├── model.py          # ViT with 1D learned positional embedding (218 lines)
│   ├── train.py          # Training script
│   └── utils.py          # Data loading, augmentation, metrics
├── modified_vit/
│   ├── model.py          # ViT with 2D RoPE (ONLY file that differs) (485 lines)
│   └── train.py          # Training script (identical logic)
├── compare.py            # Generate comparison plots
├── run_original.sh       # SLURM script: train original ViT
├── run_modified.sh       # SLURM script: train modified ViT
├── run_compare.sh        # SLURM script: generate plots + report
├── README.md             # This file
├── CHANGES_README.md     # Line-by-line changes: original vs modified ViT
├── .gitignore            # Git ignore rules
└── results/              # Output directory (created during training)
    ├── original_metrics.json
    ├── modified_metrics.json
    ├── loss_curves.png
    ├── accuracy_curves.png
    ├── justification.pdf # Why 2D RoPE was chosen (≤1 page)
    └── discussion.pdf    # Did results match expectations? (≤1 page)
```

---

## How to Run

### Step 1: Train Original ViT
```bash
sbatch run_original.sh
```

### Step 2: Train Modified ViT
```bash
sbatch run_modified.sh
```

Both can be submitted simultaneously — they are independent.

### Step 3: Generate Comparison (after both jobs complete)
```bash
sbatch run_compare.sh
```

Or run directly:
```bash
conda activate hicom_bw
python compare.py --results_dir ./results
```

---

## Changes Made: Modified ViT vs. Original ViT

The **only file that differs** between the two codebases is `model.py`. Below is a line-by-line summary of all changes.

### File: `modified_vit/model.py` vs `original_vit/model.py`

#### 1. Added `grid_size` attribute to `PatchEmbedding` (Line 28)
```diff
  # In PatchEmbedding.__init__():
  self.num_patches = (img_size // patch_size) ** 2
+ self.grid_size = img_size // patch_size            # 8
```
**Why:** The RoPE module needs to know the grid dimensions to compute 2D position indices.

---

#### 2. Added `RotaryPositionEmbedding2D` class (Lines 44–131)
```diff
+ class RotaryPositionEmbedding2D(nn.Module):
+     """
+     2D Rotary Position Embedding for Vision Transformers.
+     Splits head dimension into two halves:
+       - First half: encodes row (y) position
+       - Second half: encodes column (x) position
+     """
+     def __init__(self, head_dim, grid_size=8, theta_base=10000.0):
+         ...  # Precomputes sinusoidal frequencies for 2D grid
+
+     def forward(self, q, k, has_cls_token=True):
+         ...  # Applies rotation to Q and K tensors
```
**Why:** This is the core replacement for the 1D learned positional embedding. Instead of adding a position vector to token embeddings, it rotates Q and K vectors in attention using position-dependent rotation matrices.

**Key details:**
- Lines 73–80: Computes inverse frequencies for sinusoidal encoding
- Lines 83–89: Creates 2D grid of angles (row × col)
- Lines 92–94: Precomputes cos/sin buffers (no learnable parameters)
- Lines 108–120: Applies rotation formula: `x_rot = x1*cos - x2*sin, x2*cos + x1*sin`
- Lines 121–123: Skips CLS token (it has no spatial position)

---

#### 3. Modified `MultiHeadSelfAttention` to accept and apply RoPE (Lines 140, 155–160)
```diff
  class MultiHeadSelfAttention(nn.Module):
-     def __init__(self, embed_dim=192, num_heads=3, dropout=0.1):
+     def __init__(self, embed_dim=192, num_heads=3, dropout=0.1, rope=None):
          ...
+         self.rope = rope

      def forward(self, x):
          ...
          q, k, v = qkv.unbind(0)
+
+         # Apply 2D RoPE rotation to Q and K
+         if self.rope is not None:
+             q, k = self.rope(q, k, has_cls_token=True)
+
          attn = (q @ k.transpose(-2, -1)) * self.scale
```
**Why:** RoPE is applied inside the attention mechanism (to Q and K before computing attention scores), unlike the original approach which adds position at the input.

---

#### 4. Modified `TransformerBlock` to pass RoPE to attention (Lines 179, 185)
```diff
  class TransformerBlock(nn.Module):
-     def __init__(self, embed_dim=192, num_heads=3, mlp_ratio=4, dropout=0.1):
+     def __init__(self, embed_dim=192, num_heads=3, mlp_ratio=4, dropout=0.1, rope=None):
          ...
-         self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
+         self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout, rope=rope)
```

---

#### 5. Modified `ViT.__init__`: Replaced `pos_embed` with RoPE (Lines 221–229)
```diff
  class ViT(nn.Module):
      def __init__(self, ...):
          ...
+         # Create shared 2D RoPE module (replaces self.pos_embed)
+         self.rope = RotaryPositionEmbedding2D(head_dim, grid_size)
+
          self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
-         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
          ...
          self.blocks = nn.Sequential(*[
-             TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
+             TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, rope=self.rope)
              for _ in range(depth)
          ])
```

---

#### 6. Modified `ViT._init_weights`: Removed pos_embed initialization (Line 241)
```diff
  def _init_weights(self):
-     nn.init.trunc_normal_(self.pos_embed, std=0.02)
      nn.init.trunc_normal_(self.cls_token, std=0.02)
```

---

#### 7. Modified `ViT.forward`: Removed pos_embed addition (Line 268)
```diff
  def forward(self, x):
      ...
      x = torch.cat([cls_tokens, x], dim=1)
-     x = x + self.pos_embed
      x = self.pos_drop(x)
```
**Why:** Position is now encoded via RoPE rotations inside each attention layer, not via additive embedding at the input.

---

## Summary of Differences

| Aspect | Original ViT | Modified ViT |
|--------|-------------|--------------|
| **File changed** | `original_vit/model.py` | `modified_vit/model.py` |
| **Positional encoding type** | 1D Learned (additive) | 2D RoPE (rotational) |
| **Where applied** | Once at input (before Transformer) | Every attention layer (Q, K rotation) |
| **Extra parameters** | 12,480 (65 × 192) | 0 (precomputed sin/cos) |
| **Spatial awareness** | Implicit (1D sequence) | Explicit (2D row/col) |
| **Position type** | Absolute | Relative |
| **New classes added** | — | `RotaryPositionEmbedding2D` |
| **Lines changed** | — | ~100 lines added/modified |

---

## Training Configuration

| Hyperparameter   | Value                    |
|-----------------|--------------------------|
| Optimizer        | AdamW                    |
| Learning Rate    | 1e-3                     |
| Weight Decay     | 0.05                     |
| LR Schedule      | Cosine with 10-ep warmup |
| Batch Size       | 128                      |
| Epochs           | 200                      |
| Augmentation     | RandomCrop(32,pad=4), HFlip, AutoAugment, Mixup(α=0.8), CutMix(α=1.0) |
| Label Smoothing  | 0.1                      |
| Gradient Clipping| Max norm = 1.0           |
| Seed             | 42                       |
