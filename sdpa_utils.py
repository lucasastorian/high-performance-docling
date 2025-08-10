import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional

def _split_heads_tbd(x: Tensor, B: int, T: int, H: int, Dh: int) -> Tensor:
    """Transform [T,B,D] -> [B,H,T,Dh]"""
    return x.transpose(0,1).reshape(B, T, H, Dh).permute(0,2,1,3).contiguous()

def _merge_heads_bhtd(x: Tensor) -> Tensor:
    """Transform [B,H,T,Dh] -> [T,B,D]"""
    B,H,T,Dh = x.shape
    return x.permute(0,2,1,3).contiguous().reshape(B, T, H*Dh).transpose(0,1).contiguous()

def _merge_masks_for_sdpa(B: int, H: int, T: int, S: int,
                          key_padding_mask: Optional[Tensor],
                          attn_mask: Optional[Tensor]) -> Optional[Tensor]:
    """
    Return a boolean mask broadcastable to [B*H, T, S]; True = masked.
    key_padding_mask: [B,S] for keys (True=pad)
    attn_mask: [T,S] or [B*H,T,S] (boolean)
    """
    merged = None
    if key_padding_mask is not None:
        kpm = key_padding_mask[:, None, :].expand(B, T, S).reshape(B*H, T, S)
        merged = kpm
    if attn_mask is not None:
        if attn_mask.dim() == 2:
            attn_mask = attn_mask[None, :, :].expand(B*H, -1, -1)
        merged = attn_mask if merged is None else (merged | attn_mask)
    return merged

def mha_sdpa_forward(query: Tensor, key: Tensor, value: Tensor,
                     mha: nn.MultiheadAttention,
                     attn_mask: Optional[Tensor] = None,
                     key_padding_mask: Optional[Tensor] = None,
                     is_causal: bool = False,
                     training: bool = False) -> Tensor:
    """
    SDPA path that REUSES weights from an nn.MultiheadAttention without adding params.
    Shapes: query [Tq,B,D], key/value [Sk,B,D]. Returns [Tq,B,D].
    """
    Tq, B, D = query.shape
    Sk = key.shape[0]
    H  = mha.num_heads
    Dh = D // H

    # ---- Projections using existing tensors (no new modules) ----
    # PyTorch has two layouts:
    #  (A) single in_proj_weight (packed), (B) separate q_proj_weight/k_proj_weight/v_proj_weight.
    if getattr(mha, "_qkv_same_embed_dim", True) and hasattr(mha, "in_proj_weight"):
        # Packed weights
        W = mha.in_proj_weight      # [3D, D]
        b = mha.in_proj_bias        # [3D] or None
        W_q, W_k, W_v = W[:D, :], W[D:2*D, :], W[2*D:, :]
        b_q = b[:D]   if b is not None else None
        b_k = b[D:2*D] if b is not None else None
        b_v = b[2*D:]  if b is not None else None
        q = F.linear(query, W_q, b_q)  # [Tq,B,D]
        k = F.linear(key,   W_k, b_k)
        v = F.linear(value, W_v, b_v)
    else:
        # Try separate weights (available in some PyTorch builds)
        try:
            W_q = mha.q_proj_weight
            W_k = mha.k_proj_weight
            W_v = mha.v_proj_weight
            b = mha.in_proj_bias  # still packed bias in many builds
            b_q = b[:D]   if b is not None else None
            b_k = b[D:2*D] if b is not None else None
            b_v = b[2*D:]  if b is not None else None
            q = F.linear(query, W_q, b_q)
            k = F.linear(key,   W_k, b_k)
            v = F.linear(value, W_v, b_v)
        except AttributeError:
            # Fallback to packed weights
            W = mha.in_proj_weight      # [3D, D]
            b = mha.in_proj_bias        # [3D] or None
            W_q, W_k, W_v = W[:D, :], W[D:2*D, :], W[2*D:, :]
            b_q = b[:D]   if b is not None else None
            b_k = b[D:2*D] if b is not None else None
            b_v = b[2*D:]  if b is not None else None
            q = F.linear(query, W_q, b_q)  # [Tq,B,D]
            k = F.linear(key,   W_k, b_k)
            v = F.linear(value, W_v, b_v)

    # ---- Heads -> SDPA ----
    qh = _split_heads_tbd(q, B, Tq, H, Dh)  # [B,H,Tq,Dh]
    kh = _split_heads_tbd(k, B, Sk, H, Dh)  # [B,H,Sk,Dh]
    vh = _split_heads_tbd(v, B, Sk, H, Dh)  # [B,H,Sk,Dh]

    qf = qh.reshape(B*H, Tq, Dh)
    kf = kh.reshape(B*H, Sk, Dh)
    vf = vh.reshape(B*H, Sk, Dh)

    mask = _merge_masks_for_sdpa(B, H, Tq, Sk, key_padding_mask, attn_mask)

    y = F.scaled_dot_product_attention(
        qf, kf, vf,
        attn_mask=mask,                       # bool mask: True = blocked
        dropout_p=mha.dropout if training else 0.0,
        is_causal=is_causal
    )  # [B*H,Tq,Dh]

    y = y.reshape(B, H, Tq, Dh)
    y = _merge_heads_bhtd(y)                 # [Tq,B,D]

    # ---- Output projection (reuse existing Linear) ----
    y = mha.out_proj(y)                      # [Tq,B,D]
    return y
