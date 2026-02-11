# Anima 2B model - based on Cosmos Predict2 MiniTrainDIT architecture
# with LLMAdapter for text conditioning via Qwen3 0.6B
# Adapted for Wan2GP from ComfyUI's implementation

import math
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from shared.attention import pay_attention


logger = logging.getLogger(__name__)


# ---------------------- Rotary Position Embeddings -----------------------

def apply_rotary_pos_emb_predict2(t, freqs):
    t_ = t.reshape(*t.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2).float()
    t_out = freqs[..., 0] * t_[..., 0] + freqs[..., 1] * t_[..., 1]
    t_out = t_out.movedim(-1, -2).reshape(*t.shape).type_as(t)
    return t_out


class VideoRopePosition3DEmb(nn.Module):
    def __init__(
        self,
        model_channels,
        len_h,
        len_w,
        len_t,
        head_dim,
        max_fps=30,
        min_fps=1,
        is_learnable=False,
        interpolation="crop",
        h_extrapolation_ratio=1.0,
        w_extrapolation_ratio=1.0,
        t_extrapolation_ratio=1.0,
        enable_fps_modulation=True,
        device=None,
        **kwargs,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.len_h = len_h
        self.len_w = len_w
        self.len_t = len_t
        self.h_extrapolation_ratio = h_extrapolation_ratio
        self.w_extrapolation_ratio = w_extrapolation_ratio
        self.t_extrapolation_ratio = t_extrapolation_ratio

        dim = head_dim
        h_dim = dim // 6 * 2
        w_dim = dim // 6 * 2
        t_dim = dim - h_dim - w_dim

        self.h_dim = h_dim
        self.w_dim = w_dim
        self.t_dim = t_dim

    def _get_freqs(self, length, dim, theta=10000.0, extrapolation_ratio=1.0, device=None):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
        freqs = freqs / extrapolation_ratio
        t = torch.arange(length, device=device).float()
        freqs = torch.outer(t, freqs)
        freqs_cos = freqs.cos()
        freqs_sin = freqs.sin()
        freqs = torch.stack([freqs_cos, -freqs_sin, freqs_sin, freqs_cos], dim=-1)
        freqs = rearrange(freqs, "... (r1 r2) -> ... r1 r2", r1=2, r2=2)
        return freqs

    def forward(self, x_B_T_H_W_D, fps=None, device=None):
        B, T, H, W, D = x_B_T_H_W_D.shape
        device = device or x_B_T_H_W_D.device

        freqs_h = self._get_freqs(H, self.h_dim, extrapolation_ratio=self.h_extrapolation_ratio, device=device)
        freqs_w = self._get_freqs(W, self.w_dim, extrapolation_ratio=self.w_extrapolation_ratio, device=device)
        freqs_t = self._get_freqs(T, self.t_dim, extrapolation_ratio=self.t_extrapolation_ratio, device=device)

        freqs_h = freqs_h[None, None, :, None, :, :]  # 1, 1, H, 1, dim, 2x2
        freqs_w = freqs_w[None, None, None, :, :, :]  # 1, 1, 1, W, dim, 2x2
        freqs_t = freqs_t[None, :, None, None, :, :]  # 1, T, 1, 1, dim, 2x2

        freqs_h = freqs_h.expand(1, T, H, W, -1, -1)
        freqs_w = freqs_w.expand(1, T, H, W, -1, -1)
        freqs_t = freqs_t.expand(1, T, H, W, -1, -1)

        freqs = torch.cat([freqs_t, freqs_h, freqs_w], dim=-2)  # 1, T, H, W, head_dim/2, 2x2
        freqs = rearrange(freqs, "b t h w d r1 r2 -> b (t h w) d r1 r2")
        return freqs


# ---------------------- Feed Forward Network -----------------------

class GPT2FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.activation = nn.GELU()
        self.layer1 = nn.Linear(d_model, d_ff, bias=False)
        self.layer2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.layer2(self.activation(self.layer1(x)))


# ---------------------- Attention -----------------------

class Predict2Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, n_heads=8, head_dim=64):
        super().__init__()
        self.is_selfattn = context_dim is None
        context_dim = query_dim if context_dim is None else context_dim
        inner_dim = head_dim * n_heads

        self.n_heads = n_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.output_proj = nn.Linear(inner_dim, query_dim, bias=False)

    def forward(self, x, context=None, rope_emb=None):
        context = x if context is None else context

        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)

        q, k, v = map(
            lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim),
            (q, k, v),
        )

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.is_selfattn and rope_emb is not None:
            q = apply_rotary_pos_emb_predict2(q, rope_emb)
            k = apply_rotary_pos_emb_predict2(k, rope_emb)

        # pay_attention expects [batches, tokens, heads, head_features]
        out = pay_attention([q, k, v], cross_attn=not self.is_selfattn)
        out = rearrange(out, "b s h d -> b s (h d)")

        return self.output_proj(out)


# ---------------------- Timestep Embeddings -----------------------

class Timesteps(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps_B_T):
        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        timesteps = timesteps_B_T.flatten().float()
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) / half_dim
        emb = timesteps[:, None].float() * torch.exp(exponent)[None, :]
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
        return rearrange(emb, "(b t) d -> b t d", b=timesteps_B_T.shape[0], t=timesteps_B_T.shape[1])


class TimestepEmbedding(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear_1 = nn.Linear(in_features, out_features, bias=True)
        self.activation = nn.SiLU()
        self.linear_2 = nn.Linear(out_features, out_features, bias=False)

    def forward(self, sample):
        emb = self.linear_2(self.activation(self.linear_1(sample)))
        return emb, None  # (emb, adaln_lora=None)


# ---------------------- Patch Embed -----------------------

class PatchEmbed(nn.Module):
    def __init__(self, spatial_patch_size, temporal_patch_size, in_channels=3, out_channels=768):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.proj = nn.Sequential(
            Rearrange(
                "b c (t r) (h m) (w n) -> b t h w (c r m n)",
                r=temporal_patch_size, m=spatial_patch_size, n=spatial_patch_size,
            ),
            nn.Linear(
                in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size,
                out_channels, bias=False
            ),
        )

    def forward(self, x):
        return self.proj(x)


# ---------------------- Final Layer -----------------------

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, spatial_patch_size, temporal_patch_size, out_channels):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size,
            spatial_patch_size * spatial_patch_size * temporal_patch_size * out_channels,
            bias=False,
        )
        self.hidden_size = hidden_size
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=False),
        )

    def forward(self, x_B_T_H_W_D, emb_B_T_D, adaln_lora_B_T_3D=None):
        shift, scale = self.adaln_modulation(emb_B_T_D).chunk(2, dim=-1)
        shift = rearrange(shift, "b t d -> b t 1 1 d")
        scale = rearrange(scale, "b t d -> b t 1 1 d")
        x_B_T_H_W_D = self.layer_norm(x_B_T_H_W_D) * (1 + scale) + shift
        return self.linear(x_B_T_H_W_D)


# ---------------------- Transformer Block -----------------------

class Block(nn.Module):
    def __init__(self, x_dim, context_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.x_dim = x_dim

        self.layer_norm_self_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = Predict2Attention(x_dim, None, num_heads, x_dim // num_heads)

        self.layer_norm_cross_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = Predict2Attention(x_dim, context_dim, num_heads, x_dim // num_heads)

        self.layer_norm_mlp = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = GPT2FeedForward(x_dim, int(x_dim * mlp_ratio))

        self.adaln_modulation_self_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
        self.adaln_modulation_cross_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
        self.adaln_modulation_mlp = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))

    def forward(self, x_B_T_H_W_D, emb_B_T_D, crossattn_emb, rope_emb_L_1_1_D=None, **kwargs):
        residual_dtype = x_B_T_H_W_D.dtype
        compute_dtype = emb_B_T_D.dtype

        shift_sa, scale_sa, gate_sa = self.adaln_modulation_self_attn(emb_B_T_D).chunk(3, dim=-1)
        shift_ca, scale_ca, gate_ca = self.adaln_modulation_cross_attn(emb_B_T_D).chunk(3, dim=-1)
        shift_mlp, scale_mlp, gate_mlp = self.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)

        # Reshape for broadcasting: (B, T, D) -> (B, T, 1, 1, D)
        shift_sa, scale_sa, gate_sa = [rearrange(t, "b t d -> b t 1 1 d") for t in [shift_sa, scale_sa, gate_sa]]
        shift_ca, scale_ca, gate_ca = [rearrange(t, "b t d -> b t 1 1 d") for t in [shift_ca, scale_ca, gate_ca]]
        shift_mlp, scale_mlp, gate_mlp = [rearrange(t, "b t d -> b t 1 1 d") for t in [shift_mlp, scale_mlp, gate_mlp]]

        B, T, H, W, D = x_B_T_H_W_D.shape

        def _adaln(x, norm, scale, shift):
            return norm(x) * (1 + scale) + shift

        # Self-attention
        normed = _adaln(x_B_T_H_W_D, self.layer_norm_self_attn, scale_sa, shift_sa)
        sa_out = rearrange(
            self.self_attn(
                rearrange(normed.to(compute_dtype), "b t h w d -> b (t h w) d"),
                rope_emb=rope_emb_L_1_1_D,
            ),
            "b (t h w) d -> b t h w d", t=T, h=H, w=W,
        )
        x_B_T_H_W_D = x_B_T_H_W_D + gate_sa.to(residual_dtype) * sa_out.to(residual_dtype)

        # Cross-attention
        normed = _adaln(x_B_T_H_W_D, self.layer_norm_cross_attn, scale_ca, shift_ca)
        ca_out = rearrange(
            self.cross_attn(
                rearrange(normed.to(compute_dtype), "b t h w d -> b (t h w) d"),
                crossattn_emb,
            ),
            "b (t h w) d -> b t h w d", t=T, h=H, w=W,
        )
        x_B_T_H_W_D = x_B_T_H_W_D + gate_ca.to(residual_dtype) * ca_out.to(residual_dtype)

        # MLP
        normed = _adaln(x_B_T_H_W_D, self.layer_norm_mlp, scale_mlp, shift_mlp)
        mlp_out = self.mlp(normed.to(compute_dtype))
        x_B_T_H_W_D = x_B_T_H_W_D + gate_mlp.to(residual_dtype) * mlp_out.to(residual_dtype)

        return x_B_T_H_W_D


# ---------------------- MiniTrainDIT (Cosmos Predict2 base) -----------------------

class MiniTrainDIT(nn.Module):
    def __init__(
        self,
        max_img_h=1024,
        max_img_w=1024,
        max_frames=1,
        in_channels=16,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        concat_padding_mask=True,
        model_channels=2048,
        num_blocks=28,
        num_heads=32,
        mlp_ratio=4.0,
        crossattn_emb_channels=1024,
        pos_emb_cls="rope3d",
        pos_emb_learnable=False,
        pos_emb_interpolation="crop",
        min_fps=1,
        max_fps=30,
        rope_h_extrapolation_ratio=1.0,
        rope_w_extrapolation_ratio=1.0,
        rope_t_extrapolation_ratio=1.0,
        extra_per_block_abs_pos_emb=False,
        image_model=None,
        **kwargs,
    ):
        super().__init__()
        self.max_img_h = max_img_h
        self.max_img_w = max_img_w
        self.max_frames = max_frames
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.model_channels = model_channels
        self.concat_padding_mask = concat_padding_mask
        self.pos_emb_cls = pos_emb_cls

        # Positional embeddings
        self.pos_embedder = VideoRopePosition3DEmb(
            model_channels=model_channels,
            len_h=max_img_h // patch_spatial,
            len_w=max_img_w // patch_spatial,
            len_t=max_frames // patch_temporal,
            head_dim=model_channels // num_heads,
            max_fps=max_fps,
            min_fps=min_fps,
            is_learnable=pos_emb_learnable,
            interpolation=pos_emb_interpolation,
            h_extrapolation_ratio=rope_h_extrapolation_ratio,
            w_extrapolation_ratio=rope_w_extrapolation_ratio,
            t_extrapolation_ratio=rope_t_extrapolation_ratio,
        )

        # Timestep embedding
        self.t_embedder = nn.Sequential(
            Timesteps(model_channels),
            TimestepEmbedding(model_channels, model_channels),
        )

        # Patch embedding
        actual_in_channels = in_channels + 1 if concat_padding_mask else in_channels
        self.x_embedder = PatchEmbed(
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            in_channels=actual_in_channels,
            out_channels=model_channels,
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                x_dim=model_channels,
                context_dim=crossattn_emb_channels,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(num_blocks)
        ])

        # Final layer
        self.final_layer = FinalLayer(
            hidden_size=model_channels,
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            out_channels=out_channels,
        )

        self.t_embedding_norm = nn.RMSNorm(model_channels, eps=1e-6)

    def prepare_embedded_sequence(self, x_B_C_T_H_W, fps=None, padding_mask=None):
        if self.concat_padding_mask:
            if padding_mask is None:
                padding_mask = torch.zeros(
                    x_B_C_T_H_W.shape[0], 1, x_B_C_T_H_W.shape[3], x_B_C_T_H_W.shape[4],
                    dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device,
                )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)],
                dim=1,
            )

        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)

        if "rope" in self.pos_emb_cls.lower():
            return x_B_T_H_W_D, self.pos_embedder(x_B_T_H_W_D, fps=fps, device=x_B_C_T_H_W.device)

        return x_B_T_H_W_D, None

    def unpatchify(self, x_B_T_H_W_M):
        return rearrange(
            x_B_T_H_W_M,
            "B T H W (p1 p2 t C) -> B C (T t) (H p1) (W p2)",
            p1=self.patch_spatial,
            p2=self.patch_spatial,
            t=self.patch_temporal,
        )

    def forward(self, x, timesteps, context, fps=None, padding_mask=None, **kwargs):
        orig_shape = list(x.shape)

        # Pad to patch size
        ps = self.patch_spatial
        pt = self.patch_temporal
        _, _, T, H, W = x.shape
        pad_h = (ps - H % ps) % ps
        pad_w = (ps - W % ps) % ps
        pad_t = (pt - T % pt) % pt
        if pad_h > 0 or pad_w > 0 or pad_t > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t))

        x_B_T_H_W_D, rope_emb = self.prepare_embedded_sequence(x, fps=fps, padding_mask=padding_mask)

        if timesteps.ndim == 1:
            timesteps = timesteps.unsqueeze(1)

        t_emb, adaln_lora = self.t_embedder[1](self.t_embedder[0](timesteps).to(x_B_T_H_W_D.dtype))
        t_emb = self.t_embedding_norm(t_emb)

        rope_emb_expanded = rope_emb.unsqueeze(1).unsqueeze(0) if rope_emb is not None else None

        # fp16 residual stream fix
        if x_B_T_H_W_D.dtype == torch.float16:
            x_B_T_H_W_D = x_B_T_H_W_D.float()

        for block in self.blocks:
            x_B_T_H_W_D = block(x_B_T_H_W_D, t_emb, context, rope_emb_L_1_1_D=rope_emb_expanded)

        x_out = self.final_layer(x_B_T_H_W_D.to(context.dtype), t_emb)
        x_out = self.unpatchify(x_out)[:, :, :orig_shape[-3], :orig_shape[-2], :orig_shape[-1]]
        return x_out


# ---------------------- Anima LLM Adapter -----------------------

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_adapter(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


class AdapterRotaryEmbedding(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.rope_theta = 10000
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


class AdapterAttention(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, head_dim):
        super().__init__()
        inner_dim = head_dim * n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.o_proj = nn.Linear(inner_dim, query_dim, bias=False)

    def forward(self, x, mask=None, context=None, position_embeddings=None, position_embeddings_context=None):
        context = x if context is None else context
        input_shape = x.shape[:-1]
        q_shape = (*input_shape, self.n_heads, self.head_dim)
        context_shape = context.shape[:-1]
        kv_shape = (*context_shape, self.n_heads, self.head_dim)

        query_states = self.q_norm(self.q_proj(x).view(q_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(context).view(kv_shape)).transpose(1, 2)
        value_states = self.v_proj(context).view(kv_shape).transpose(1, 2)

        if position_embeddings is not None and position_embeddings_context is not None:
            cos, sin = position_embeddings
            query_states = apply_rotary_pos_emb_adapter(query_states, cos, sin)
            cos, sin = position_embeddings_context
            key_states = apply_rotary_pos_emb_adapter(key_states, cos, sin)

        attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=mask)
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output)


class AdapterTransformerBlock(nn.Module):
    def __init__(self, source_dim, model_dim, num_heads=16, mlp_ratio=4.0, use_self_attn=False):
        super().__init__()
        self.use_self_attn = use_self_attn

        if self.use_self_attn:
            self.norm_self_attn = nn.RMSNorm(model_dim, eps=1e-6)
            self.self_attn = AdapterAttention(model_dim, model_dim, num_heads, model_dim // num_heads)

        self.norm_cross_attn = nn.RMSNorm(model_dim, eps=1e-6)
        self.cross_attn = AdapterAttention(model_dim, source_dim, num_heads, model_dim // num_heads)

        self.norm_mlp = nn.RMSNorm(model_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, int(model_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(model_dim * mlp_ratio), model_dim),
        )

    def forward(self, x, context, target_attention_mask=None, source_attention_mask=None,
                position_embeddings=None, position_embeddings_context=None):
        if self.use_self_attn:
            normed = self.norm_self_attn(x)
            x = x + self.self_attn(normed, mask=target_attention_mask,
                                   position_embeddings=position_embeddings,
                                   position_embeddings_context=position_embeddings)

        normed = self.norm_cross_attn(x)
        x = x + self.cross_attn(normed, mask=source_attention_mask, context=context,
                                position_embeddings=position_embeddings,
                                position_embeddings_context=position_embeddings_context)

        x = x + self.mlp(self.norm_mlp(x))
        return x


class LLMAdapter(nn.Module):
    def __init__(self, source_dim=1024, target_dim=1024, model_dim=1024,
                 num_layers=6, num_heads=16, use_self_attn=True, **kwargs):
        super().__init__()
        self.embed = nn.Embedding(32128, target_dim)
        if model_dim != target_dim:
            self.in_proj = nn.Linear(target_dim, model_dim)
        else:
            self.in_proj = nn.Identity()
        self.rotary_emb = AdapterRotaryEmbedding(model_dim // num_heads)
        self.blocks = nn.ModuleList([
            AdapterTransformerBlock(source_dim, model_dim, num_heads=num_heads, use_self_attn=use_self_attn)
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(model_dim, target_dim)
        self.norm = nn.RMSNorm(target_dim, eps=1e-6)

    def forward(self, source_hidden_states, target_input_ids, target_attention_mask=None, source_attention_mask=None):
        if target_attention_mask is not None:
            target_attention_mask = target_attention_mask.to(torch.bool)
            if target_attention_mask.ndim == 2:
                target_attention_mask = target_attention_mask.unsqueeze(1).unsqueeze(1)

        if source_attention_mask is not None:
            source_attention_mask = source_attention_mask.to(torch.bool)
            if source_attention_mask.ndim == 2:
                source_attention_mask = source_attention_mask.unsqueeze(1).unsqueeze(1)

        x = self.in_proj(self.embed(target_input_ids))
        context = source_hidden_states
        position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        position_ids_context = torch.arange(context.shape[1], device=x.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(x, position_ids)
        position_embeddings_context = self.rotary_emb(x, position_ids_context)
        for block in self.blocks:
            x = block(x, context,
                      target_attention_mask=target_attention_mask,
                      source_attention_mask=source_attention_mask,
                      position_embeddings=position_embeddings,
                      position_embeddings_context=position_embeddings_context)
        return self.norm(self.out_proj(x))


# ---------------------- Anima Model -----------------------

class AnimaModel(MiniTrainDIT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_adapter = LLMAdapter()

    def preprocess_text_embeds(self, text_embeds, text_ids):
        if text_ids is not None:
            out = self.llm_adapter(text_embeds, text_ids)
            if out.shape[1] < 512:
                out = F.pad(out, (0, 0, 0, 512 - out.shape[1]))
            return out
        return text_embeds

    def forward(self, x, timesteps, context, **kwargs):
        t5xxl_ids = kwargs.pop("t5xxl_ids", None)
        if t5xxl_ids is not None:
            context = self.preprocess_text_embeds(context, t5xxl_ids)
        return super().forward(x, timesteps, context, **kwargs)
