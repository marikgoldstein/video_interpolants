import math
from typing import Tuple
import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import torch
from torch import nn

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x: torch.Tensor):
        b, _, h, w = x.shape
        not_mask = torch.ones([b, h, w], dtype=torch.float32, device=x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(512, num_pos_feats)
        self.col_embed = nn.Embedding(512, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight, a=-1., b=1.)
        nn.init.uniform_(self.col_embed.weight, a=-1., b=1.)

    def forward(self, x: torch.Tensor):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


class PositionEmbeddingLearnedFlat(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.embed = nn.Embedding(500, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.embed.weight, a=-1., b=1.)

    def forward(self, x: torch.Tensor):
        n = x.shape[-2]
        i = torch.arange(n, device=x.device)
        emb = self.embed(i)
        pos = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
        return pos


def build_position_encoding(hidden_dim: int, position_embedding_name: str):
    N_steps = hidden_dim // 2
    if position_embedding_name in ('v2', 'sine'):
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif position_embedding_name in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    elif position_embedding_name in ('v4', 'learned_flat'):
        position_embedding = PositionEmbeddingLearnedFlat(hidden_dim)
    else:
        raise ValueError(f"not supported")

    return position_embedding


def timestamp_embedding(timesteps, dim, scale=200, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param scale: a premultiplier of timesteps
    :param max_period: controls the minimum frequency of the embeddings.
    :param repeat_only: whether to repeat only the values in timesteps along the 2nd dim
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = scale * timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(scale * timesteps, 'b -> b d', d=dim)
    return embedding


class VectorFieldRegressor(nn.Module):
    def __init__(self, 
        state_size = 4, 
        state_res = [8, 8], 
        inner_dim = 768, 
        depth = 4, 
        mid_depth = 5, 
        out_norm = 'ln'
    ):
        super(VectorFieldRegressor, self).__init__()
        self.reference = True # whether to use the random context frame.
        self.state_size = state_size
        self.state_res = state_res
        self.state_height = self.state_res[0]
        self.state_width = self.state_res[1]
        self.inner_dim = inner_dim
        self.depth = depth
        self.mid_depth = mid_depth
        self.out_norm = out_norm
        self.position_encoding = build_position_encoding(self.inner_dim, position_embedding_name="learned")
        self.project_in = nn.Sequential(
            Rearrange("b c h w -> b (h w) c"),
            nn.Linear(3 * self.state_size if self.reference else 2 * self.state_size, self.inner_dim)
        )
        self.time_projection = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Linear(256, self.inner_dim)
        )

        def build_layer(d_model: int):
            return nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=4 * d_model,
                dropout=0.05,
                activation="gelu",
                norm_first=True,
                batch_first=True)

        self.in_blocks = nn.ModuleList()
        self.mid_blocks = nn.Sequential(*[build_layer(self.inner_dim) for _ in range(self.mid_depth)])
        self.out_blocks = nn.ModuleList()
        for i in range(self.depth):
            self.in_blocks.append(build_layer(self.inner_dim))
            self.out_blocks.append(nn.ModuleList([
                nn.Linear(2 * self.inner_dim, self.inner_dim),
                build_layer(self.inner_dim)]))

        if self.out_norm == "ln":
            self.project_out = nn.Sequential(
                nn.Linear(self.inner_dim, self.inner_dim),
                nn.GELU(),
                nn.LayerNorm(self.inner_dim),
                Rearrange("b (h w) c -> b c h w", h=self.state_height),
                nn.Conv2d(self.inner_dim, self.state_size, kernel_size=3, stride=1, padding=1),
            )
        elif self.out_norm == "bn":
            self.project_out = nn.Sequential(
                nn.Linear(self.inner_dim, self.inner_dim),
                Rearrange("b (h w) c -> b c h w", h=self.state_height),
                nn.GELU(),
                nn.BatchNorm2d(self.inner_dim),
                nn.Conv2d(self.inner_dim, self.state_size, kernel_size=3, stride=1, padding=1),
            )
        else:
            raise NotImplementedError


    def forward(self, xt, t, ref, cond, gap):
        """

        :param input_latents: [b, c, h, w]
        :param reference_latents: [b, c, h, w]
        :param conditioning_latents: [b, c, h, w]
        :param index_distances: [b]
        :param timestamps: [b]
        :return: [b, c, h, w]
        """
        pos_enc = self.position_encoding
        p_in = self.project_in
        p_out = self.project_out 
        gap_embedder = self.time_projection




        # Fetch timestamp tokens
        t = timestamp_embedding(t, dim=self.inner_dim)[:, None] # was unsqueeze 1

        # Calculate position embedding
        pos = pos_enc(xt)
        pos = rearrange(pos, "b c h w -> b (h w) c")

        # Calculate distance embeddings
        gap_embedding = gap_embedder(torch.log(gap))[:, None]

        # Build input tokens
        if self.reference:
            x = torch.cat([xt, ref, cond], dim = 1)
        else:
            x = torch.cat([xt, cond], dim = 1)

        x = p_in(x)
        x = x + pos + gap_embedding
        x = torch.cat([t, x], dim=1)

        # Propagate through the main network
        hs = []
        for b in self.in_blocks:
            x = b(x)
            hs.append(x.clone())
        x = self.mid_blocks(x)
        for i, b in enumerate(self.out_blocks):
            x = b[1](b[0](torch.cat([hs[-i - 1], x], dim=-1)))

        # Project to output
        out = p_out(x[:, 1:])

        return out
