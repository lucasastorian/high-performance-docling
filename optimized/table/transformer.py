# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT

import math
import os
import logging
from typing import Optional, List

import torch
from torch import nn, Tensor
import torch.nn.functional as F

import docling_ibm_models.tableformer.utils.utils as u

LOG_LEVEL = logging.INFO


def _enable_fast_sdp():
    """Prefer Flash/mem-efficient SDPA kernels on CUDA."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal PE with an eval-time fast path that adds PE only to the *last* token.
    - Input can be [T, B, D] or [B, T, D]; we forward it untouched except for the PE add.
    - For training, or if you want full PE, set last_only=False.
    """
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 4096, last_only: bool = True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.last_only = last_only

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        # Shape: [max_len, 1, d_model] to broadcast over batch for [T, B, D] inputs.
        self.register_buffer("pe_tbd", pe.unsqueeze(1))  # [L,1,D]

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [T,B,D] or [B,T,D]
        Returns x with PE added (in-place add on the last step path to minimize copies).
        """
        if x.dim() != 3:
            return x  # safety

        if self.training or not self.last_only:
            # Add PE to the whole prefix
            if x.shape[0] < x.shape[1]:  # probably [B,T,D] -> transpose view for add
                B, T, D = x.shape
                # make a [T,B,D] view without copy if possible
                x_tbd = x.transpose(0, 1)  # [T,B,D]
                x_tbd = x_tbd + self.pe_tbd[:T]  # broadcast over B
                return self.dropout(x_tbd.transpose(0, 1))
            else:
                T, B, D = x.shape
                x = x + self.pe_tbd[:T]
                return self.dropout(x)

        # Eval fast-path: add PE to only the last position
        if x.shape[0] < x.shape[1]:  # [B,T,D]
            B, T, D = x.shape
            x[:, T - 1:T, :] = x[:, T - 1:T, :] + self.pe_tbd[T - 1:T]  # broadcast over B
            return self.dropout(x)
        else:  # [T,B,D]
            T, B, D = x.shape
            x[T - 1:T, :, :] = x[T - 1:T, :, :] + self.pe_tbd[T - 1:T]
            return self.dropout(x)


class KVCache:
    """Per-layer self-attention KV cache (projected, head-split tensors)."""
    def __init__(self):
        self.k: Optional[Tensor] = None  # [B, H, T, Dh]
        self.v: Optional[Tensor] = None  # [B, H, T, Dh]


def _shape_for_heads(x: Tensor, n_heads: int) -> Tensor:
    # x: [B, T, D] -> [B, H, T, Dh]
    B, T, D = x.shape
    Dh = D // n_heads
    return x.view(B, T, n_heads, Dh).transpose(1, 2)


def _merge_heads(x: Tensor) -> Tensor:
    # x: [B, H, T, Dh] -> [B, T, D]
    B, H, T, Dh = x.shape
    return x.transpose(1, 2).contiguous().view(B, T, H * Dh)


def _project_qkv(mha: nn.MultiheadAttention, x: Tensor):
    """
    Project QKV using mha.in_proj_weight/bias. x is [B,T,D] (batch_first).
    Returns (q, k, v) as [B,H,T,Dh] each.
    """
    W = mha.in_proj_weight  # [3D, D]
    b = mha.in_proj_bias    # [3D]
    q, k, v = F.linear(x, W, b).chunk(3, dim=-1)  # [B,T,D] each
    H = mha.num_heads
    return _shape_for_heads(q, H), _shape_for_heads(k, H), _shape_for_heads(v, H)


class FastDecoderLayer(nn.Module):
    """
    Pre-LN Transformer decoder layer optimized for last-token generation.
    - Batch-first throughout: inputs/outputs are [B, T, D] (we produce only the last step).
    - Real KV cache for self-attention (stores projected K/V).
    - Uses Flash/SDPA via scaled_dot_product_attention when available.
    """
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.0, use_sdpa: bool = True):
        super().__init__()
        self.use_sdpa = use_sdpa

        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=0.0)

        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=0.0)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.SiLU(),
            nn.Linear(dim_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

        # Length thresholds where SDPA gives benefit (measured on CUDA)
        self.self_len_threshold = 64
        self.cross_len_threshold = 64

    def forward(
        self,
        tgt: Tensor,                 # [B, T, D] (we will use only last token as query)
        memory: Optional[Tensor],    # [B, S, D] or None
        cache: Optional[KVCache],    # per-layer KV cache
    ) -> (Tensor, KVCache):          # returns ([B, 1, D], cache)
        if cache is None:
            cache = KVCache()

        # --- Self-attention (last-token query) ---
        x = tgt                              # [B,T,D]
        y = self.norm1(x)                    # Pre-LN
        q_last = y[:, -1:, :]                # [B,1,D]

        # Project q/k/v for the last token and update cache K/V
        qh, kh_new, vh_new = _project_qkv(self.self_attn, q_last)  # [B,H,1,Dh] each
        if cache.k is None:
            # First step: also project K/V for the (same) last token (length=1)
            cache.k, cache.v = kh_new, vh_new                       # [B,H,1,Dh]
        else:
            # Append only the latest key/value
            cache.k = torch.cat([cache.k, kh_new], dim=2)           # [B,H,T,Dh]
            cache.v = torch.cat([cache.v, vh_new], dim=2)           # [B,H,T,Dh]

        # SDPA over heads (Flash path if enabled)
        T_kv = cache.k.size(2)
        use_sdp_self = self.use_sdpa and (T_kv >= self.self_len_threshold)
        if use_sdp_self:
            attn = F.scaled_dot_product_attention(  # [B,H,1,Dh]
                qh, cache.k, cache.v, is_causal=False)
        else:
            # Fallback: call MHA on batch-first last token vs whole prefix
            # (still efficient on PyTorch >=2.1)
            attn, _ = self.self_attn(q_last, y, y, need_weights=False)
            attn = _shape_for_heads(attn, self.self_attn.num_heads)  # [B,H,1,Dh]

        sa_out = _merge_heads(attn)                                   # [B,1,D]
        h = x[:, -1:, :] + self.dropout(sa_out)                       # residual

        # --- Cross-attention (optional) ---
        if memory is not None:
            z = self.norm2(h)
            S_mem = memory.size(1)
            use_sdp_cross = self.use_sdpa and (S_mem >= self.cross_len_threshold)
            if use_sdp_cross:
                # Project q/k/v ourselves to use SDPA directly
                qh, _, _ = _project_qkv(self.cross_attn, z)          # [B,H,1,Dh]
                # For keys/vals we project the full memory once (no cache — memory is static)
                # Use the MHA's input projection matrices:
                W = self.cross_attn.in_proj_weight
                b = self.cross_attn.in_proj_bias
                # Only take K,V blocks from in_proj (PyTorch packs as [Q;K;V]):
                D = z.size(-1)
                Wk, Wv = W[D:2*D, :], W[2*D:, :]
                bk, bv = b[D:2*D], b[2*D:]
                kv = F.linear(memory, torch.cat([Wk, Wv], dim=0), torch.cat([bk, bv], dim=0))
                kh, vh = kv.split(D, dim=-1)
                kh = _shape_for_heads(kh, self.cross_attn.num_heads)  # [B,H,S,Dh]
                vh = _shape_for_heads(vh, self.cross_attn.num_heads)  # [B,H,S,Dh]

                ca = F.scaled_dot_product_attention(qh, kh, vh, is_causal=False)  # [B,H,1,Dh]
                ca = _merge_heads(ca)                                             # [B,1,D]
            else:
                ca, _ = self.cross_attn(z, memory, memory, need_weights=False)    # [B,1,D]
            h = h + self.dropout(ca)

        # --- FFN (fused) ---
        out = h + self.dropout(self.ffn(h))  # [B,1,D]
        return out, cache


class FastTransformerDecoder(nn.Module):
    """
    Stack of FastDecoderLayer with per-layer KV caches.
    forward() returns only the last token embedding ([1,B,D] after transpose for compatibility)
    plus updated caches.
    """
    def __init__(self, d_model: int, nhead: int, dim_ff: int, num_layers: int, dropout: float = 0.0, use_sdpa: bool = True):
        super().__init__()
        self.layers = nn.ModuleList([
            FastDecoderLayer(d_model, nhead, dim_ff, dropout=dropout, use_sdpa=use_sdpa)
        for _ in range(num_layers)])

    def forward(
        self,
        tgt: Tensor,                 # [T,B,D] or [B,T,D]
        memory: Optional[Tensor] = None,  # [S,B,D] or [B,S,D]
        cache: Optional[List[KVCache]] = None,
        memory_mask: Optional[Tensor] = None,               # ignored
        tgt_key_padding_mask: Optional[Tensor] = None,      # ignored
        memory_key_padding_mask: Optional[Tensor] = None,   # ignored
    ):
        # Normalize shapes to batch-first
        if tgt.shape[0] < tgt.shape[1]:
            # [B,T,D]
            x_btd = tgt
        else:
            # [T,B,D] -> [B,T,D]
            x_btd = tgt.transpose(0, 1)

        mem_bsd = None
        if memory is not None:
            mem_bsd = memory.transpose(0, 1) if memory.shape[0] > memory.shape[1] else memory  # [B,S,D]

        # Setup caches
        if cache is None:
            caches = [KVCache() for _ in range(len(self.layers))]
        else:
            caches = cache

        # Run layers (last-token only)
        h = x_btd
        new_caches: List[KVCache] = []
        for layer, kv in zip(self.layers, caches):
            h, new_kv = layer(h, mem_bsd, kv)
            new_caches.append(new_kv)

        # Return [1,B,D] to match existing "decoded[-1, :, :]" pattern
        out_1bd = h.transpose(0, 1)  # [1,B,D]
        return out_1bd, new_caches


class Tag_Transformer(nn.Module):
    """
    Optimized tag transformer
    - Batch-first internally (encoder & decoder).
    - Encoder: stock TransformerEncoder (batch_first=True).
    - Decoder: FastTransformerDecoder (Pre-LN, fused FFN, KV cache, last-token only).
    - `_positional_encoding` adds PE only to last token during eval to avoid O(T) PE adds.
    - Exposes same public attributes used elsewhere: `_embedding`, `_positional_encoding`,
      `_input_filter`, `_encoder`, `_decoder`, `_fc`.
    """
    def __init__(
        self,
        device: str,
        vocab_size: int,
        td_encode,                   # kept for API parity (not used directly here)
        embed_dim: int,
        encoder_layers: int,
        decoder_layers: int,
        enc_image_size: int,
        dropout: float = 0.0,
        n_heads: int = 4,
        dim_ff: int = 1024,
        use_sdpa: bool = True,
        compile_decoder: bool = True,
    ):
        super().__init__()
        _enable_fast_sdp()

        self._device = device
        self._n_heads = n_heads
        self._decoder_dim = embed_dim
        self._enc_image_size = enc_image_size

        # Embedding + PE (PE adds only to last token during eval)
        self._embedding = nn.Embedding(vocab_size, embed_dim)
        self._positional_encoding = PositionalEncoding(embed_dim, dropout=0.0, max_len=4096, last_only=True)

        # Image input filter stays NCHW -> NCHW (you already optimized around this)
        self._input_filter = u.resnet_block(stride=1)

        # Transformer encoder (batch_first=True). We accept [S,B,D] at call sites and
        # convert once per call inside.
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads,
                                               dim_feedforward=dim_ff, dropout=0.0,
                                               batch_first=True)
        self._encoder = nn.TransformerEncoder(enc_layer, num_layers=encoder_layers,
                                              enable_nested_tensor=False)

        # Fast decoder stack
        dec = FastTransformerDecoder(embed_dim, n_heads, dim_ff, decoder_layers,
                                     dropout=dropout, use_sdpa=use_sdpa)
        if compile_decoder and hasattr(torch, "compile"):
            try:
                dec = torch.compile(dec, mode="reduce-overhead", fullgraph=False)
            except Exception:
                pass
        self._decoder = dec

        # Final classifier
        self._fc = nn.Linear(embed_dim, vocab_size)

        # Inference-only: don’t waste time on autograd book-keeping
        for p in self.parameters():
            p.requires_grad_(False)

        self.eval()

    # -------- Convenience helpers you already rely on in your pipeline --------

    def encode_memory(self, enc_hwc: Tensor) -> Tensor:
        """
        Take encoder features [B,h,w,C] (NHWC) and produce transformer memory [S,B,C]
        to match existing call sites.
        """
        B, h, w, C = enc_hwc.shape
        mem_bsc = enc_hwc.reshape(B, h * w, C)     # [B,S,C]
        mem_sbc = mem_bsc.transpose(0, 1).contiguous()  # [S,B,C] for compatibility
        return mem_sbc

    # -------- Typical usage patterns --------
    # 1) Encoder path (as you do now):
    #    x_nchw = _input_filter(enc_out_nchw)              # [B,C,h,w]
    #    x_nhwc = x_nchw.permute(0,2,3,1)                  # [B,h,w,C]
    #    mem_sbc = _encoder(x_nhwc)                        # [S,B,C] (wrapper below)
    #
    # 2) Decoder step in your AR loop:
    #    tgt = _positional_encoding(_embedding(decoded_tags[:t+1]))  # [T,B,D]
    #    decoded, cache = _decoder(tgt, memory=mem_sbc, cache=cache) # decoded -> [1,B,D]
    #    last_H = decoded[-1, :, :]                                   # [B,D]
    #
    # -------------------------------------------------------------------------

    def _encode_sbc(self, mem_sbc: Tensor) -> Tensor:
        """
        Wrapper to keep external shape [S,B,C] while running encoder batch-first.
        """
        if mem_sbc.dim() != 3:
            return mem_sbc
        # [S,B,C] -> [B,S,C]
        mem_bsc = mem_sbc.transpose(0, 1).contiguous()
        enc_bsc = self._encoder(mem_bsc)            # batch_first=True
        # [B,S,C] -> [S,B,C]
        return enc_bsc.transpose(0, 1).contiguous()

    # Keep the names you already use externally:
    # - call sites typically do: mem_enc = self._tag_transformer._encoder(mem, mask=None)
    # So we expose a callable that accepts [S,B,C] and returns [S,B,C].
    def _encoder(self, mem_sbc: Tensor, mask: Optional[Tensor] = None) -> Tensor:  # type: ignore
        # mask is ignored intentionally (dense features; last-token decoding needs no causal mask)
        return self._encode_sbc(mem_sbc)

    # Training path (teacher-forced, full-seq) isn’t the hot path and is omitted here.
    # If you need it later, iterate time-steps and concatenate decoder outputs (uses KV cache).

    # Optional: a compact inference utility mirroring your old `.inference(...)`
    def inference(self, enc_inputs_nhwc: Tensor, tags: Tensor, tag_lens: Tensor, num_cells: int):
        """
        Compatibility stub for batch teacher-forcing (kept minimal).
        Encodes the image, then runs the decoder autoregressively over the gold tags.
        Returns per-step logits shaped like the original.
        """
        # Flatten [B,h,w,C] -> [S,B,C], encode
        mem_sbc = self.encode_memory(enc_inputs_nhwc)
        mem_enc = self._encode_sbc(mem_sbc)  # [S,B,C]

        B = tags.size(0)
        T = tags.size(1)

        # AR over teacher tokens (slow path; used only for training/eval loops, not prod decode)
        cache: Optional[List[KVCache]] = None
        logits_steps: List[Tensor] = []
        for t in range(1, T + 1):
            # Embed up to t, add PE only to last (fast path in eval)
            tgt = self._embedding(tags[:, :t])           # [B,t,D]
            tgt = tgt.transpose(0, 1)                    # [t,B,D] to match upstream usage
            tgt = self._positional_encoding(tgt)         # [t,B,D] with last-only PE
            # Decode last step
            decoded, cache = self._decoder(tgt, memory=mem_enc, cache=cache)  # [1,B,D]
            step_logits = self._fc(decoded[-1])          # [B,D_vocab]
            logits_steps.append(step_logits)

        # [B,T,V]
        predictions = torch.stack(logits_steps, dim=1)
        decode_lengths = (tag_lens - 1).tolist()
        return predictions, decode_lengths
