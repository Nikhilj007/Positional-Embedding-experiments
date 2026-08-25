# Changes Made: Modified ViT vs. Original ViT

## Overview

The **only file** that differs between the two codebases is `model.py`. All other files
(`train.py`, `utils.py`) contain identical training logic. The modification replaces the
**1D Learnable Positional Embedding** with **2D Rotary Position Embedding (RoPE)**.

**Source files:**
- Original: `original_vit/model.py` (218 lines)
- Modified: `modified_vit/model.py` (485 lines)

---

## Change 1: Added `grid_size` attribute to `PatchEmbedding`

| | Original (`original_vit/model.py`) | Modified (`modified_vit/model.py`) |
|---|---|---|
| **Line** | Line 26 | Lines 28–29 |
| **Type** | ADDED (1 line) | |

**Original (Line 26):**
```python
self.num_patches = (img_size // patch_size) ** 2  # 64 for 32x32 with 4x4 patches
```

**Modified (Lines 28–29):**
```python
self.num_patches = (img_size // patch_size) ** 2  # 64 for 32x32 with 4x4 patches
self.grid_size = img_size // patch_size            # 8    # <-- NEW
```

**Why:** RoPE needs the 2D grid dimensions (8×8) to compute row and column position indices.

---

## Change 2: Added `RotaryPositionEmbedding2D` class (NEW)

| | Original (`original_vit/model.py`) | Modified (`modified_vit/model.py`) |
|---|---|---|
| **Lines** | Does not exist | Lines 44–277 |
| **Type** | NEW CLASS (~234 lines) | |

This entirely new class implements 2D Rotary Position Embedding. Key sections:

| Modified Lines | Purpose |
|---|---|
| 61–77 | Class definition and docstring |
| 79–101 | `__init__`: Compute inverse frequencies for sinusoidal encoding |
| 104–106 | Position indices: `0, 1, ..., grid_size-1` |
| 109–134 | Create 2D coordinate grid of angles (row × col) |
| 143–159 | Precompute and register cos/sin buffers (no learnable parameters) |
| 161–187 | `rotate_pairs()`: Static method applying rotation formula: `x1*cos - x2*sin`, `x2*cos + x1*sin` |
| 189–277 | `forward()`: Splits head dim into row/col halves, applies rotation to Q and K, skips CLS token |

**Why:** This is the core replacement for the 1D learned positional embedding. Instead of adding a fixed position vector to token embeddings at the input, RoPE encodes position by rotating Q and K vectors inside every attention layer.

---

## Change 3: Modified `MultiHeadSelfAttention.__init__` to accept RoPE

| | Original (`original_vit/model.py`) | Modified (`modified_vit/model.py`) |
|---|---|---|
| **Line** | Line 44 | Line 288 |
| **Type** | MODIFIED (signature) | |

**Original (Line 44):**
```python
def __init__(self, embed_dim=192, num_heads=3, dropout=0.1):
```

**Modified (Line 288):**
```python
def __init__(self, embed_dim=192, num_heads=3, dropout=0.1, rope=None):
```

---

## Change 4: Added `self.rope` storage in `MultiHeadSelfAttention`

| | Original (`original_vit/model.py`) | Modified (`modified_vit/model.py`) |
|---|---|---|
| **Lines** | Does not exist (after Line 53) | Line 302 |
| **Type** | ADDED (1 line) | |

**Modified (Line 302):**
```python
self.rope = rope
```

---

## Change 5: Added RoPE application in `MultiHeadSelfAttention.forward`

| | Original (`original_vit/model.py`) | Modified (`modified_vit/model.py`) |
|---|---|---|
| **Lines** | Between Lines 59–61 | Lines 310–314 |
| **Type** | ADDED (5 lines) | |

**Original (Lines 59–61):**
```python
q, k, v = qkv.unbind(0)

attn = (q @ k.transpose(-2, -1)) * self.scale
```

**Modified (Lines 308–316):**
```python
q, k, v = qkv.unbind(0)

# CHANGE: Apply 2D RoPE rotation to Q and K
if self.rope is not None:
    q, k = self.rope(q, k, has_cls_token=True)

attn = (q @ k.transpose(-2, -1)) * self.scale
```

**Why:** RoPE is applied inside the attention mechanism (to Q and K before computing attention scores), unlike the original approach which adds position information at the input.

---

## Change 6: Modified `TransformerBlock.__init__` to accept and pass RoPE

| | Original (`original_vit/model.py`) | Modified (`modified_vit/model.py`) |
|---|---|---|
| **Lines** | Lines 94, 97 | Lines 349, 355 |
| **Type** | MODIFIED (2 lines) | |

**Original (Lines 94, 97):**
```python
def __init__(self, embed_dim=192, num_heads=3, mlp_ratio=4, dropout=0.1):
    ...
    self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
```

**Modified (Lines 349, 355):**
```python
def __init__(self, embed_dim=192, num_heads=3, mlp_ratio=4, dropout=0.1, rope=None):
    ...
    self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout, rope=rope)
```

---

## Change 7: Replaced `self.pos_embed` with `self.rope` in `ViT.__init__`

| | Original (`original_vit/model.py`) | Modified (`modified_vit/model.py`) |
|---|---|---|
| **Lines** | Lines 131–138 | Lines 394–405 |
| **Type** | MODIFIED + DELETED + ADDED | |

**Original (Lines 131–138):**
```python
num_patches = self.patch_embed.num_patches  # 64

# POSITIONAL EMBEDDING: 1D Learnable (Original ViT)
# A learnable vector for each position (CLS + 64 patches = 65)
self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
```

**Modified (Lines 394–405):**
```python
num_patches = self.patch_embed.num_patches  # 64
grid_size = self.patch_embed.grid_size       # 8       # <-- NEW
head_dim = embed_dim // num_heads            # 64      # <-- NEW

# CHANGE: Create shared 2D RoPE module (replaces self.pos_embed)
self.rope = RotaryPositionEmbedding2D(head_dim, grid_size)  # <-- NEW

self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
# REMOVED: self.pos_embed = nn.Parameter(...)               # <-- DELETED
```

---

## Change 8: Pass RoPE to each `TransformerBlock`

| | Original (`original_vit/model.py`) | Modified (`modified_vit/model.py`) |
|---|---|---|
| **Lines** | Lines 143–144 | Lines 410–411 |
| **Type** | MODIFIED (1 line) | |

**Original (Lines 143–144):**
```python
self.blocks = nn.Sequential(*[
    TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
```

**Modified (Lines 410–411):**
```python
self.blocks = nn.Sequential(*[
    TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, rope=self.rope)
```

---

## Change 9: Removed `pos_embed` initialization in `_init_weights`

| | Original (`original_vit/model.py`) | Modified (`modified_vit/model.py`) |
|---|---|---|
| **Lines** | Lines 157–158 | Line 424 |
| **Type** | DELETED (1 line) | |

**Original (Lines 157–158):**
```python
def _init_weights(self):
    # Initialize positional embedding with truncated normal
    nn.init.trunc_normal_(self.pos_embed, std=0.02)     # <-- DELETED in modified
    nn.init.trunc_normal_(self.cls_token, std=0.02)
```

**Modified (Lines 423–425):**
```python
def _init_weights(self):
    # REMOVED: nn.init.trunc_normal_(self.pos_embed, std=0.02)
    nn.init.trunc_normal_(self.cls_token, std=0.02)
```

---

## Change 10: Removed positional embedding addition in `forward`

| | Original (`original_vit/model.py`) | Modified (`modified_vit/model.py`) |
|---|---|---|
| **Lines** | Lines 183–187 | Lines 449–454 |
| **Type** | DELETED (1 line) | |

**Original (Lines 183–187):**
```python
# ADD POSITIONAL EMBEDDING (Original ViT approach)
x = x + self.pos_embed  # Add learned positional embeddings       # <-- DELETED
x = self.pos_drop(x)
```

**Modified (Lines 449–454):**
```python
# CHANGE: NO positional embedding addition here
# Position is encoded via RoPE rotations inside attention layers
# REMOVED: x = x + self.pos_embed
x = self.pos_drop(x)    # Dropout only, no pos_embed addition
```

**Why:** Position is now encoded via RoPE rotations inside each attention layer, not via an additive embedding at the input.

---

## Summary Table

| # | Change Type | Original Line(s) | Modified Line(s) | Description |
|---|---|---|---|---|
| 1 | ADDED | After 26 | 29 | `self.grid_size` in PatchEmbedding |
| 2 | NEW CLASS | — | 44–277 | `RotaryPositionEmbedding2D` (234 lines) |
| 3 | MODIFIED | 44 | 288 | Added `rope=None` to MHSA `__init__` |
| 4 | ADDED | After 53 | 302 | `self.rope = rope` in MHSA |
| 5 | ADDED | After 59 | 310–314 | RoPE application to Q, K in MHSA `forward` |
| 6 | MODIFIED | 94, 97 | 349, 355 | Added `rope=None` to TransformerBlock, pass to MHSA |
| 7 | REPLACED | 131–138 | 394–405 | Replaced `self.pos_embed` with `self.rope` in ViT `__init__` |
| 8 | MODIFIED | 143–144 | 410–411 | Pass `rope=self.rope` to TransformerBlock |
| 9 | DELETED | 157–158 | 424 | Removed `pos_embed` initialization |
| 10 | DELETED | 186 | 454 | Removed `x = x + self.pos_embed` in forward |

**Total:** 3 lines deleted, ~240 lines added, 5 lines modified. All changes are in `model.py` only.

---

## Unchanged Components

The following are **identical** between original and modified ViT:

- `PatchEmbedding` class (except the added `grid_size` attribute)
- `MLP` class (Lines 71–88 original → Lines 326–343 modified)
- `TransformerBlock.forward()` method
- `ViT._init_module_weights()` method
- `ViT.forward()` logic flow (patch → CLS → dropout → blocks → norm → head)
- `train.py` — entire training script is identical
- `utils.py` — data loading, augmentation, metrics are identical
