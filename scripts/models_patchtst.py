"""PatchTST model for C-MAPSS RUL prediction.

Reference: Nie et al. "A Time Series is Worth 64 Words" (ICLR 2023).
Simplified implementation: no tsai dependency, pure PyTorch.
"""
import torch
import torch.nn as nn
import math


class PatchTST_RUL(nn.Module):
    """PatchTST for RUL regression.

    Args:
        n_vars: number of input channels (sensors)
        seq_len: input sequence length (window size)
        patch_len: patch length
        stride: stride between patches (overlap = patch_len - stride)
        d_model: transformer latent dimension
        n_heads: number of attention heads
        n_layers: number of transformer encoder layers
        dropout: dropout rate
    """
    def __init__(self, n_vars, seq_len=30, patch_len=6, stride=3,
                 d_model=128, n_heads=4, n_layers=3, dropout=0.2):
        super().__init__()
        self.n_vars = n_vars
        self.patch_len = patch_len
        self.stride = stride
        self.seq_len = seq_len

        # Number of patches
        self.n_patches = ((seq_len - patch_len) // stride) + 1

        # Patch embedding (per variable, shared weights)
        self.patch_embed = nn.Linear(patch_len, d_model)

        # Positional encoding (per patch position, shared across vars)
        self.pos_embed = nn.Parameter(torch.randn(1, 1, self.n_patches, d_model) * 0.02)

        # Transformer encoder (applied per variable independently)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Layer norm before head
        self.norm = nn.LayerNorm(d_model)

        # Regression head: global average pooling over patches + linear
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        """x: (B, seq_len, n_vars)"""
        B, L, V = x.shape

        # Create patches: unfold each variable independently
        # x_unfold: (B, V, n_patches, patch_len)
        x_unfold = x.unfold(dimension=1, size=self.patch_len, step=self.stride)  # (B, n_patches, V, patch_len)
        x_unfold = x_unfold.permute(0, 2, 1, 3)  # (B, V, n_patches, patch_len)

        # Embed patches: (B, V, n_patches, d_model)
        x_embed = self.patch_embed(x_unfold)

        # Add positional encoding
        x_embed = x_embed + self.pos_embed

        # Reshape for per-variable transformer: (B*V, n_patches, d_model)
        x_tfm = x_embed.reshape(B * V, self.n_patches, -1)

        # Transformer (per variable independently)
        x_tfm = self.transformer(x_tfm)  # (B*V, n_patches, d_model)

        # Reshape back: (B, V, n_patches, d_model)
        x_tfm = x_tfm.reshape(B, V, self.n_patches, -1)

        # Global average pooling over patches + variables
        x_pool = x_tfm.mean(dim=(1, 2))  # (B, d_model)

        # Layer norm + head
        x_out = self.norm(x_pool)
        return self.head(x_out).squeeze(-1)  # (B,)
