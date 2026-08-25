"""
Vision Transformer (ViT) — Original Implementation
Positional Encoding: 1D Learnable Positional Embeddings (as in Dosovitskiy et al., ICLR 2021)

Architecture:
    - Image Size: 32x32 (CIFAR-100)
    - Patch Size: 4x4
    - Embedding Dim: 192
    - Depth: 12 Transformer blocks
    - Attention Heads: 3
    - MLP Ratio: 4x
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


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention."""

    def __init__(self, embed_dim=192, num_heads=3, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

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

    def __init__(self, embed_dim=192, num_heads=3, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    """
    Vision Transformer with 1D Learnable Positional Embeddings.

    The positional embedding is a learnable parameter of shape (1, num_patches+1, embed_dim)
    that is added to the patch embeddings (including the CLS token) before being fed
    into the Transformer encoder. This is the standard approach from the original
    ViT paper (Dosovitskiy et al., ICLR 2021).
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

        # ============================================================
        # POSITIONAL EMBEDDING: 1D Learnable (Original ViT)
        # A learnable vector for each position (CLS + 64 patches = 65)
        # ============================================================
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))  # Line 113

        self.pos_drop = nn.Dropout(dropout)

        # Transformer encoder blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize positional embedding with truncated normal
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
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
        # ADD POSITIONAL EMBEDDING (Original ViT approach)
        # ============================================================
        x = x + self.pos_embed  # Line 153: Add learned positional embeddings
        x = self.pos_drop(x)

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
    print(f"Original ViT (1D Learned Positional Embedding)")
    print(f"  Total parameters:     {total:,}")
    print(f"  Trainable parameters: {trainable:,}")

    # Test forward pass
    dummy = torch.randn(2, 3, 32, 32)
    out = model(dummy)
    print(f"  Input shape:  {dummy.shape}")
    print(f"  Output shape: {out.shape}")
