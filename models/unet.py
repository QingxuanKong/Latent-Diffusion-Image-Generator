import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from .unet_modules import (
    TimeEmbedding,
    DownSample,
    UpSample,
    ResBlock,
    AdaGN_ResBlock,
    TransformerBlock,
)


class UNet(nn.Module):
    def __init__(
        self,
        input_size,
        input_ch,
        T,
        ch,
        ch_mult,
        attn,
        num_res_blocks,
        dropout=0.0,
        conditional=False,
        c_dim=None,
        # --- New arguments for AdaGN ResNet ---
        use_adagn_resblock=False,
        # --- New arguments for Transformer Bottleneck ---
        use_transformer_bottleneck=False,  # Flag to enable transformer
        transformer_depth=1,  # Number of transformer blocks
        transformer_num_heads=8,  # Number of attention heads
        # --------------------------------------------
    ):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), "attn index out of bound"

        self.input_size = input_size
        self.input_ch = input_ch
        self.T = T
        self.conditional = conditional
        self.c_dim = c_dim if conditional else 0

        BlockClass = AdaGN_ResBlock if use_adagn_resblock else ResBlock

        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)

        # --- Determine conditional embedding dimension ---
        self.cond_embed_dim = tdim
        if self.conditional:
            if self.c_dim is None:
                raise ValueError("c_dim must be provided if conditional=True")
            self.cond_embed_dim += self.c_dim
        # -----------------------------------------------

        self.stem = nn.Conv2d(input_ch, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(
                    BlockClass(
                        in_ch=now_ch,
                        out_ch=out_ch,
                        tdim=tdim,
                        dropout=dropout,
                        attn=(
                            i in attn and not use_transformer_bottleneck
                        ),  # Disable conv attn if transformer used
                        cross_attn=conditional
                        and (i in attn),  # Keep cross-attn for ResBlocks if needed
                        cdim=c_dim,
                        conditional=conditional,
                    )
                )
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        # --- Middle Blocks: Either ResNet/Attn or Transformer ---
        self.use_transformer_bottleneck = use_transformer_bottleneck
        if use_transformer_bottleneck:
            bottleneck_resolution = input_size // (2 ** (len(ch_mult) - 1))
            num_patches = bottleneck_resolution * bottleneck_resolution
            self.pos_emb = nn.Parameter(
                torch.zeros(1, num_patches, now_ch)
            )  # Learnable Positional Embedding
            self.middleblocks = nn.ModuleList(
                [
                    TransformerBlock(
                        embed_dim=now_ch,
                        num_heads=transformer_num_heads,
                        cond_embed_dim=self.cond_embed_dim,  # Pass combined dim
                        dropout=dropout,
                    )
                    for _ in range(transformer_depth)
                ]
            )
            print(
                f"[INFO] Using Transformer bottleneck with {transformer_depth} layers, {transformer_num_heads} heads."
            )
        else:
            self.middleblocks = nn.ModuleList(
                [
                    BlockClass(
                        now_ch,
                        now_ch,
                        tdim,
                        dropout,
                        attn=True,
                        cross_attn=conditional,
                        cdim=c_dim,
                        conditional=conditional,
                    ),
                    BlockClass(
                        now_ch,
                        now_ch,
                        tdim,
                        dropout,
                        attn=False,
                        cross_attn=False,
                        cdim=c_dim,
                        conditional=conditional,
                    ),
                ]
            )
            print("[INFO] Using ResNet/Attention bottleneck.")
        # -------------------------------------------------------

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(
                    BlockClass(
                        in_ch=chs.pop() + now_ch,
                        out_ch=out_ch,
                        tdim=tdim,
                        dropout=dropout,
                        attn=(i in attn),
                        cross_attn=conditional and (i in attn),
                        cdim=c_dim,
                        conditional=conditional,
                    )
                )
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.head = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            nn.SiLU(),
            nn.Conv2d(now_ch, input_ch, 1, stride=1, padding=0),
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.stem.weight)
        init.zeros_(self.stem.bias)
        init.xavier_uniform_(self.head[-1].weight, gain=1e-5)
        init.zeros_(self.head[-1].bias)
        if hasattr(self, "pos_emb"):
            init.normal_(self.pos_emb, std=0.02)

    def forward(self, x, t, c=None):
        # Time embedding
        if not torch.is_tensor(t):
            t = torch.tensor([t], dtype=torch.long, device=x.device)
        elif torch.is_tensor(t) and len(t.shape) == 0:
            t = t[None].to(x.device)
        t = t * torch.ones(x.shape[0], dtype=t.dtype, device=t.device)
        temb = self.time_embedding(t)  # shape: (batch_size, tdim)

        # --- Prepare combined conditional embedding ---
        cond_emb = temb
        if self.conditional:
            if c is None:
                raise ValueError("Conditional=True but class embedding c is None")
            if c.shape[1] != self.c_dim:
                raise ValueError(
                    f"Class embedding dim {c.shape[1]} does not match c_dim {self.c_dim}"
                )
            cond_emb = torch.cat([temb, c], dim=1)  # (B, tdim + c_dim)
        # -------------------------------------------

        # Downsampling
        h = self.stem(x)  # shape: (batch_size, ch, h, w)
        hs = [h]
        for i, layer in enumerate(self.downblocks):
            if isinstance(layer, (ResBlock, AdaGN_ResBlock)):
                h = layer(h, temb, c)
            else:
                h = layer(h)
            hs.append(h)  # shape: (batch_size, out_ch, h, w)

        # Middle Blocks
        if self.use_transformer_bottleneck:
            B, C, H_mid, W_mid = h.shape
            h = h.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
            h = h + self.pos_emb  # Add positional embedding
            for layer in self.middleblocks:
                h = layer(h, cond_emb)  # TransformerBlock takes combined embedding
            h = h.permute(0, 2, 1).view(B, C, H_mid, W_mid)  # Reshape back
        else:
            for i, layer in enumerate(self.middleblocks):
                h = layer(h, temb, c)  # shape: (batch_size, now_channel, h, w)

        # Upsampling
        for i, layer in enumerate(self.upblocks):
            if isinstance(layer, (ResBlock, AdaGN_ResBlock)):
                h = torch.cat(
                    [h, hs.pop()], dim=1
                )  # shape: (batch_size, reverse downsampling + last upsampling, h, w)
                h = layer(h, temb, c)
            else:
                h = layer(h, temb, c)

        h = self.head(h)  # (B, C_in, H_in, W_in)

        assert len(hs) == 0
        return h
