"""
Vision Transformer (ViT) — Modified Implementation
Positional Encoding: 2D Rotary Position Embedding (RoPE)

Architecture is IDENTICAL to the original ViT except:
  - REMOVED: Learnable 1D positional embeddings (self.pos_embed)
  - ADDED:   2D Rotary Position Embedding applied to Q and K in each attention layer

All other hyperparameters (embed_dim, depth, num_heads, mlp_ratio, patch_size) are the same.

References:
  - RoFormer (Su et al., 2021): https://arxiv.org/abs/2104.09864
  - EVA-02 (Fang et al., 2023): uses 2D RoPE for ViT
"""

import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    """Split image into patches and project to embedding dimension."""

    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 64 for 32x32 with 4x4 patches
        self.grid_size = img_size // patch_size            # 8

        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W) -> (B, embed_dim, H/P, W/P) -> (B, num_patches, embed_dim)
        x = self.projection(x)              # (B, embed_dim, 8, 8)
        x = x.flatten(2)                     # (B, embed_dim, 64)
        x = x.transpose(1, 2)               # (B, 64, embed_dim)
        return x


# ============================================================
# 2D ROTARY POSITION EMBEDDING (RoPE)
# This replaces the learnable 1D positional embedding.
#
# Instead of adding a position vector to the token embeddings,
# RoPE encodes position by rotating the query and key vectors
# in the self-attention layer. For 2D images, we split the
# head dimension in half: the first half encodes row position,
# the second half encodes column position.
#
# Key differences from Original ViT:
#   1. No learnable position parameters (zero extra params)
#   2. Encodes RELATIVE position (rotation depends on distance)
#   3. Explicitly models 2D spatial structure (row, col)
#   4. Applied inside attention (to Q, K) rather than at input
# ============================================================

class RotaryPositionEmbedding2D(nn.Module):
    """
    2D Rotary Position Embedding for Vision Transformers.

    The attention head dimension is divided into two parts:

        First half  -> row (Y) position
        Second half -> column (X) position

    Within each half, dimensions are paired and rotated using
    sinusoidal frequencies.

    For head_dim = 64:
        32 dims -> row
        32 dims -> column
        16 rotation pairs per axis
    """

    def __init__(self, head_dim, grid_size=8, theta_base=10000.0):
        super().__init__()

        assert head_dim % 4 == 0, \
            "head_dim must be divisible by 4 for 2D RoPE"

        self.head_dim = head_dim
        self.grid_size = grid_size

        # Half of the head dimension is allocated to each axis.
        axis_dim = head_dim // 2

        # Each rotary pair contains 2 dimensions.
        pair_dim = axis_dim // 2

        # Inverse frequencies for rotary dimensions.
        freq_seq = torch.arange(
            pair_dim, dtype=torch.float32
        )

        inv_freq = 1.0 / (
            theta_base ** (freq_seq / pair_dim)
        )

        # Position indices: 0 ... grid_size-1
        positions = torch.arange(
            grid_size, dtype=torch.float32
        )

        # Angles for each row/column position.
        row_angles = torch.einsum(
            "i,j->ij",
            positions,
            inv_freq
        )

        col_angles = torch.einsum(
            "i,j->ij",
            positions,
            inv_freq
        )

        # Create 2D coordinate grid.
        row_grid = (
            row_angles
            .unsqueeze(1)
            .expand(-1, grid_size, -1)
            .reshape(-1, pair_dim)
        )

        col_grid = (
            col_angles
            .unsqueeze(0)
            .expand(grid_size, -1, -1)
            .reshape(-1, pair_dim)
        )

        # Shape:
        # (num_patches, pair_dim)
        #
        # row_grid = 16 dimensions
        # col_grid = 16 dimensions
        #
        # Combined = 32 angles
        angles = torch.cat(
            [row_grid, col_grid],
            dim=-1
        )

        # Each angle corresponds to one rotation pair.
        self.register_buffer(
            "cos_cached",
            angles.cos().unsqueeze(0).unsqueeze(0),
            persistent=False
        )

        self.register_buffer(
            "sin_cached",
            angles.sin().unsqueeze(0).unsqueeze(0),
            persistent=False
        )

    @staticmethod
    def rotate_pairs(x, cos, sin):
        """
        Apply rotary transformation to paired dimensions.

        x:   (..., 2 * num_pairs)
        cos: (..., num_pairs)
        sin: (..., num_pairs)
        """

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_even_rot = (
            x_even * cos
            - x_odd * sin
        )

        x_odd_rot = (
            x_odd * cos
            + x_even * sin
        )

        return torch.stack(
            [x_even_rot, x_odd_rot],
            dim=-1
        ).flatten(-2)

    def forward(self, q, k, has_cls_token=True):

        if has_cls_token:
            q_cls = q[:, :, :1, :]
            q_patches = q[:, :, 1:, :]

            k_cls = k[:, :, :1, :]
            k_patches = k[:, :, 1:, :]
        else:
            q_patches = q
            k_patches = k

        # ---------------------------------------------------------
        # Split head dimension into row and column portions.
        # ---------------------------------------------------------

        axis_dim = self.head_dim // 2

        q_row = q_patches[..., :axis_dim]
        q_col = q_patches[..., axis_dim:]

        k_row = k_patches[..., :axis_dim]
        k_col = k_patches[..., axis_dim:]

        # ---------------------------------------------------------
        # Cached angles.
        #
        # First half -> row
        # Second half -> column
        # ---------------------------------------------------------

        cos_row = self.cos_cached[..., :axis_dim // 2]
        sin_row = self.sin_cached[..., :axis_dim // 2]

        cos_col = self.cos_cached[..., axis_dim // 2:]
        sin_col = self.sin_cached[..., axis_dim // 2:]

        # ---------------------------------------------------------
        # Apply independent rotary transformations.
        # ---------------------------------------------------------

        q_row = self.rotate_pairs(
            q_row,
            cos_row,
            sin_row
        )

        k_row = self.rotate_pairs(
            k_row,
            cos_row,
            sin_row
        )

        q_col = self.rotate_pairs(
            q_col,
            cos_col,
            sin_col
        )

        k_col = self.rotate_pairs(
            k_col,
            cos_col,
            sin_col
        )

        # Recombine row + column dimensions.
        q_rotated = torch.cat(
            [q_row, q_col],
            dim=-1
        )

        k_rotated = torch.cat(
            [k_row, k_col],
            dim=-1
        )

        # CLS token is left unrotated.
        if has_cls_token:
            q_rotated = torch.cat(
                [q_cls, q_rotated],
                dim=2
            )

            k_rotated = torch.cat(
                [k_cls, k_rotated],
                dim=2
            )

        return q_rotated, k_rotated


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention WITH 2D Rotary Position Embedding.

    CHANGED vs Original ViT:
      - Accepts a RoPE module and applies it to Q and K before attention.
    """

    def __init__(self, embed_dim=192, num_heads=3, dropout=0.1, rope=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # ============================================================
        # CHANGE: Store reference to RoPE module
        # ============================================================
        self.rope = rope

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        # ============================================================
        # CHANGE: Apply 2D RoPE rotation to Q and K
        # ============================================================
        if self.rope is not None:
            q, k = self.rope(q, k, has_cls_token=True)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, embed_dim=192, mlp_ratio=4, dropout=0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Single Transformer encoder block: LayerNorm -> MHSA -> LayerNorm -> MLP."""

    def __init__(self, embed_dim=192, num_heads=3, mlp_ratio=4, dropout=0.1, rope=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        # ============================================================
        # CHANGE: Pass RoPE module to attention layer
        # ============================================================
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout, rope=rope)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    """
    Vision Transformer with 2D Rotary Position Embedding (RoPE).

    CHANGES vs Original ViT:
      1. REMOVED: self.pos_embed (learnable 1D positional embedding)
      2. REMOVED: pos_embed initialization in _init_weights()
      3. REMOVED: x = x + self.pos_embed in forward()
      4. ADDED:   RotaryPositionEmbedding2D module
      5. ADDED:   RoPE is passed to each TransformerBlock -> MultiHeadSelfAttention
      6. ADDED:   Q and K are rotated inside attention before computing scores

    All other components (patch embedding, CLS token, MLP, depth, width) are IDENTICAL.
    """

    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=100,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        dropout=0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches  # 64
        grid_size = self.patch_embed.grid_size       # 8
        head_dim = embed_dim // num_heads            # 64

        # ============================================================
        # CHANGE: Create shared 2D RoPE module (replaces self.pos_embed)
        # No learnable parameters — positions encoded via rotations
        # ============================================================
        self.rope = RotaryPositionEmbedding2D(head_dim, grid_size)  # Line 221

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # REMOVED: self.pos_embed = nn.Parameter(...)                # Line 224 (deleted)

        self.pos_drop = nn.Dropout(dropout)

        # Transformer encoder blocks — each receives the shared RoPE module
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, rope=self.rope)  # Line 229
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # REMOVED: nn.init.trunc_normal_(self.pos_embed, std=0.02)  # Line 241 (deleted)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Initialize linear layers and layer norms
        self.apply(self._init_module_weights)

    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, 64, 192)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 65, 192)

        # ============================================================
        # CHANGE: NO positional embedding addition here
        # Position is encoded via RoPE rotations inside attention layers
        # REMOVED: x = x + self.pos_embed
        # ============================================================
        x = self.pos_drop(x)  # Line 268: Dropout only, no pos_embed addition

        # Transformer encoder
        x = self.blocks(x)
        x = self.norm(x)

        # Classification: use CLS token output
        cls_output = x[:, 0]
        logits = self.head(cls_output)
        return logits


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    model = ViT()
    total, trainable = count_parameters(model)
    print(f"Modified ViT (2D Rotary Position Embedding)")
    print(f"  Total parameters:     {total:,}")
    print(f"  Trainable parameters: {trainable:,}")

    # Test forward pass
    dummy = torch.randn(2, 3, 32, 32)
    out = model(dummy)
    print(f"  Input shape:  {dummy.shape}")
    print(f"  Output shape: {out.shape}")
