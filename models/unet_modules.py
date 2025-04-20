import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = (
            torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        )  # shape: (d_model//2,)
        emb = torch.exp(-emb)  # shape: (d_model//2,)
        pos = torch.arange(T).float()  # shape: (T,)
        emb = pos[:, None] * emb[None, :]  # shape: (T, d_model//2)
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack(
            [torch.sin(emb), torch.cos(emb)], dim=-1
        )  # shape: (T, d_model//2, 2)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model).contiguous()  # shape: (T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb=None, cemb=None):
        x = self.main(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb=None, cemb=None):
        _, _, H, W = x.shape
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.main(x)
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C).contiguous()
        k = k.view(B, C, H * W).contiguous()
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C).contiguous()
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        h = self.proj(h)

        return x + h


class CrossAttnBlock(nn.Module):
    def __init__(self, in_ch, c_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.class_norm = nn.LayerNorm(c_ch)  # Normalization for class embeddings
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Linear(c_ch, in_ch)
        self.proj_v = nn.Linear(c_ch, in_ch)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            if isinstance(module, nn.Conv2d):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, x, class_emb):
        B, C, H, W = x.shape

        # Normalize input
        h = self.group_norm(x)

        # Normalize class embeddings
        class_emb = self.class_norm(class_emb)

        # Query from input feature map
        q = self.proj_q(h)
        q = q.permute(0, 2, 3, 1).view(B, H * W, C).contiguous()

        # Key and Value from class embeddings
        k = self.proj_k(class_emb).unsqueeze(1)  # Shape: (B, 1, in_ch)
        v = self.proj_v(class_emb).unsqueeze(1)  # Shape: (B, 1, in_ch)

        # Attention weights
        w = torch.bmm(q, k.permute(0, 2, 1)) * (int(C) ** (-0.5))  # Shape: (B, H*W, 1)
        w = F.softmax(w, dim=-1)  # Shape: (B, H*W, 1)

        # Attention output
        h = torch.bmm(w, v)  # Shape: (B, H*W, in_ch)
        h = (
            h.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        )  # Shape: (B, in_ch, H, W)

        # Final projection and skip connection
        h = self.proj(h)

        return x + h


class ResBlock(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        tdim,
        dropout,
        attn=False,
        cross_attn=False,
        cdim=None,
        conditional=False,
    ):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()
        if cross_attn:
            assert cdim is not None
            self.cross_attn = CrossAttnBlock(out_ch, cdim)
        else:
            self.cross_attn = nn.Identity()

        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb, cemb=None):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        if isinstance(self.attn, AttnBlock):
            h = self.attn(h)
        if isinstance(self.cross_attn, CrossAttnBlock):
            h = self.cross_attn(h, cemb)
        return h


class AdaGN_ResBlock(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        tdim,
        dropout,
        attn=False,
        cross_attn=False,
        cdim=None,
        conditional=False,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.tdim = tdim  # Store tdim
        self.cdim = cdim if conditional else None

        # Block 1
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.silu1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)

        # Determine the embedding dimension for AdaGN projections
        # If class conditioning is configured (cdim is not None), AdaGN will expect combined embedding.
        self.ada_embed_dim = self.tdim
        if conditional:
            self.ada_embed_dim += self.cdim

        # AdaGN Projection Layer for Block 1 (and 2)
        # Uses the determined ada_embed_dim
        self.adaGN_proj1 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                self.ada_embed_dim, 2 * in_ch
            ),  # Output: scale and shift for out_ch channels
        )
        self.adaGN_proj2 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                self.ada_embed_dim, 2 * out_ch
            ),  # Output: scale and shift for out_ch channels
        )

        # Block 2
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.silu2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1)

        # Shortcut connection
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()

        # Attention layers
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()

        # Cross Attention - only instantiate if needed
        if cross_attn:
            if self.cdim is None:
                raise ValueError("cdim must be provided if cross_attn is True")
            self.cross_attn = CrossAttnBlock(out_ch, self.cdim)
        else:
            self.cross_attn = nn.Identity()  # Keep as identity otherwise

        self.initialize()

    def initialize(self):
        # Initialize conv layers
        for module in [self.conv1, self.conv2, self.shortcut]:
            if isinstance(module, nn.Conv2d):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

        # Initialize AdaGN projections: zero out the final Linear layer
        init.zeros_(self.adaGN_proj1[-1].weight)
        init.zeros_(self.adaGN_proj1[-1].bias)
        init.zeros_(self.adaGN_proj2[-1].weight)
        init.zeros_(self.adaGN_proj2[-1].bias)

        # Initialize attention projection layers
        if isinstance(self.attn, AttnBlock):
            # Add initialize method to AttnBlock if it doesn't have one
            if hasattr(self.attn, "initialize") and callable(self.attn.initialize):
                self.attn.initialize()
        if isinstance(self.cross_attn, CrossAttnBlock):
            # Add initialize method to CrossAttnBlock if it doesn't have one
            if hasattr(self.cross_attn, "initialize") and callable(
                self.cross_attn.initialize
            ):
                self.cross_attn.initialize()

        # Zero-out the last conv layer in the residual block
        init.zeros_(
            self.conv2.weight
        )  # Zero init often preferred for last layer in resblock
        init.zeros_(self.conv2.bias)

    def forward(self, x, temb, cemb=None):
        # Input x: (B, C_in, H, W)
        # Input temb: (B, tdim)
        # Input cemb: (B, cdim) or None

        # --- Shortcut ---
        shortcut_output = self.shortcut(x)  # (B, C_out, H, W)

        # --- Prepare Embedding for AdaGN ---
        # Start with time embedding
        combined_emb = temb

        # If class conditioning is configured for the U-Net (cdim was passed to __init__)
        # AND class embedding is actually provided in this forward pass, concatenate them.
        if self.cdim is not None:
            if cemb is None:
                print(f"cdim: {self.cdim}, but cemb is None")
                raise ValueError(
                    f"ResBlock configured with cdim={self.cdim} but cemb is None in forward pass."
                )

            # Concatenate time and class embeddings
            combined_emb = torch.cat([temb, cemb], dim=-1)
            # Verify the dimension matches what adaGN_proj expects
            if combined_emb.shape[1] != self.ada_embed_dim:
                raise ValueError(
                    f"Combined embedding dim {combined_emb.shape[1]} does not match expected ada_embed_dim {self.ada_embed_dim}"
                )

        # --- Block 1 ---
        h = self.norm1(x)

        # Project for scale and shift (Block 1)
        ada_params1 = self.adaGN_proj1(combined_emb)  # (B, 2 * C_out)
        scale1, shift1 = torch.chunk(ada_params1, 2, dim=1)  # (B, C_out), (B, C_out)

        # Apply AdaGN 1
        h = (
            h * (1 + scale1[:, :, None, None]) + shift1[:, :, None, None]
        )  # Modulate after norm

        h = self.silu1(h)
        h = self.conv1(h)  # (B, C_out, H, W)

        # --- Block 2 ---
        h = self.norm2(h)

        # Project for scale and shift (Block 2)
        ada_params2 = self.adaGN_proj2(combined_emb)  # (B, 2 * C_out)
        scale2, shift2 = torch.chunk(ada_params2, 2, dim=1)  # (B, C_out), (B, C_out)

        # Apply AdaGN 2
        h = (
            h * (1 + scale2[:, :, None, None]) + shift2[:, :, None, None]
        )  # Modulate after norm

        h = self.silu2(h)
        h = self.dropout(h)
        h = self.conv2(h)  # (B, C_out, H, W)

        # --- Final Addition + Attention ---
        h = h + shortcut_output  # Add shortcut

        # Apply self-attention if applicable
        if isinstance(self.attn, AttnBlock):
            h = self.attn(h)

        # Apply cross-attention if applicable (only if self.use_cross_attn was True)
        if isinstance(self.cross_attn, CrossAttnBlock):
            if (
                cemb is None
            ):  # Double check, though checked earlier when creating combined_emb
                raise ValueError(
                    "Class embedding `cemb` cannot be None when cross_attn is enabled."
                )
            h = self.cross_attn(h, cemb)

        return h


class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization
    """

    def __init__(self, embedding_dim, cond_embedding_dim):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)
        # Projection layer for generating scale and shift parameters
        self.projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_embedding_dim, 2 * embedding_dim),
        )
        self.initialize()

    def initialize(self):
        # Zero-initialize the final projection layer for stability
        init.zeros_(self.projection[-1].weight)
        init.zeros_(self.projection[-1].bias)

    def forward(self, x, cond_embedding):
        # x: (B, N, D) or (N, B, D)
        # cond_embedding: (B, cond_embedding_dim)

        # Project conditional embedding to get scale and shift
        scale_shift = self.projection(cond_embedding)  # (B, 2*D)
        scale, shift = torch.chunk(scale_shift, 2, dim=1)  # (B, D), (B, D)

        # Reshape scale and shift for broadcasting
        # If x is (B, N, D), need (B, 1, D)
        # If x is (N, B, D), need (1, B, D)
        if x.dim() == 3 and x.shape[0] == scale.shape[0]:  # batch_first=True expected
            scale = scale.unsqueeze(1)  # (B, 1, D)
            shift = shift.unsqueeze(1)  # (B, 1, D)
        elif x.dim() == 3 and x.shape[1] == scale.shape[0]:  # batch_first=False
            scale = scale.unsqueeze(0)  # (1, B, D)
            shift = shift.unsqueeze(0)  # (1, B, D)
        else:
            # Handle potential other cases or raise error
            raise ValueError(f"Unexpected input shape {x.shape} for AdaLN")

        # Apply normalization
        x_norm = self.norm(x)

        # Apply adaptive modulation
        return x_norm * (1 + scale) + shift


class TransformerBlock(nn.Module):
    """
    A Transformer block with Adaptive Layer Normalization.
    Uses batch_first=True convention: (Batch, Sequence, Feature)
    """

    def __init__(
        self, embed_dim, num_heads, cond_embed_dim, mlp_ratio=4.0, dropout=0.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.cond_embed_dim = cond_embed_dim

        # Adaptive Layer Norm + Multi-Head Self-Attention
        self.norm1 = AdaLN(embed_dim, cond_embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Adaptive Layer Norm + MLP
        self.norm2 = AdaLN(embed_dim, cond_embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.SiLU(),  # Changed from GELU to SiLU to match UNet style
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.initialize()

    def initialize(self):
        # Initialize MLP layers
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.zeros_(layer.bias)

    def forward(self, x, cond_embedding):
        # x: (B, N, D)
        # cond_embedding: (B, cond_embed_dim)

        # Self-Attention part
        residual = x
        x_norm = self.norm1(x, cond_embedding)  # shape: (B, H*W, C)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = residual + attn_output

        # MLP part
        residual = x
        x_norm = self.norm2(x, cond_embedding)
        mlp_output = self.mlp(x_norm)
        x = residual + mlp_output

        return x
