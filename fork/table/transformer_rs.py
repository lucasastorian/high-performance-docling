#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#


import logging
import math
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import docling_ibm_models.tableformer.utils.utils as u

LOG_LEVEL = logging.INFO


# LOG_LEVEL = logging.DEBUG


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)  # Keep as float32 for checkpoint compatibility

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TMTransformerDecoder(nn.TransformerDecoder):
    def forward(  # type: ignore
            self,
            tgt: Tensor,
            memory: Optional[Tensor] = None,
            cache: Optional[Tensor] = None,  # Unused now, kept for compatibility
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            memory_kv=None,  # NEW
            sa_kv_cache=None,  # NEW: self-attention KV cache list
            max_pred_len: Optional[int] = None,  # For capacity hint
    ):
        """
        Args:
            tgt (Tensor): encoded tags. (tags_len,bsz,hidden_dim)
            memory (Tensor): encoded image (enc_image_size,bsz,hidden_dim)
            cache (Optional[Tensor]): None during training, only used during inference.
        Returns:
            output (Tensor): (tags_len,bsz,hidden_dim)
        """

        # Always use incremental path now (Checkpoint C)
        # tgt must be [1, B, D] (single token with PE already added)
        tgt_last = tgt  # Should be [1, B, D]
        new_sa_kv_cache = []  # Always create KV list

        for i, mod in enumerate(self.layers):
            # Get KV cache for this layer
            layer_self_kv = None
            if sa_kv_cache is not None and i < len(sa_kv_cache):
                layer_self_kv = sa_kv_cache[i]

            # Call layer with single token (pass capacity hint for better initial allocation)
            result = mod(
                tgt_last, memory,  # tgt_last is [1, B, D]
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_kv=None if memory_kv is None else memory_kv[i],
                self_kv=layer_self_kv,
                max_pred_len=max_pred_len,  # Pass through capacity hint
            )

            # Handle return format
            if isinstance(result, tuple):
                tgt_last, layer_kv_new = result  # [1, B, D], (K, V)
            else:
                tgt_last, layer_kv_new = result, None

            # Collect new KV cache
            new_sa_kv_cache.append(layer_kv_new)

        # Return: no tag cache, just the final last token output
        return tgt_last, new_sa_kv_cache  # [1, B, D], list of KV


class TMTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def _flash_debug_guard(self, q, k, v, is_causal, where=""):
        """Debug function to check Flash Attention eligibility"""
        def info(t): 
            return (t.device.type, t.dtype, t.shape, t.is_contiguous(), t.stride())
        B,H,S_q,Dh = q.shape
        _,_,S_k,_ = k.shape
        problems = []
        if q.device.type != "cuda" or k.device.type != "cuda" or v.device.type != "cuda":
            problems.append("not on CUDA")
        if not (q.dtype == k.dtype == v.dtype):
            problems.append(f"dtype mismatch: {q.dtype},{k.dtype},{v.dtype}")
        if q.dtype not in (torch.float16, torch.bfloat16):
            problems.append(f"bad dtype: {q.dtype}")
        if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
            problems.append("non-contiguous q/k/v")
        if Dh % 8 != 0:
            problems.append(f"Dh % 8 != 0 (Dh={Dh})")
        if min(B,H,S_q,S_k,Dh) <= 0:
            problems.append(f"zero-sized dim: B={B},H={H},S_q={S_q},S_k={S_k},Dh={Dh}")
        if problems:
            print(f"[FLASH ELIGIBILITY FAIL @ {where}] {problems}")
            print("  q:", info(q))
            print("  k:", info(k))
            print("  v:", info(v))
            return False
        else:
            print(f"[FLASH OK @ {where}] B={B},H={H},S_q={S_q},S_k={S_k},Dh={Dh}")
        return True
    
    def _sa_kv_step(
            self,
            last_in: torch.Tensor,  # [1, B, D]  (layer input for the *last* token)
            kv_prev,  # Either (K_prev, V_prev) or (K_buf, V_buf, t, cap) for O(1) growth
            cap_hint: Optional[int] = None,  # Capacity hint for initial allocation
    ):
        """
        Incremental self-attention with O(1) KV cache growth and SDPA.
        Flash-compatible: keeps 4D tensors and uses is_causal=True.
        Return:
          sa_out: [1, B, D]   (self-attention output, before residual)
          kv_new: (K_buf, V_buf, t, cap) with O(1) growth buffer
        """
        mha = self.self_attn
        E = mha.embed_dim
        H = mha.num_heads
        Dh = E // H
        B = last_in.size(1)

        # Fused QKV projection: ONE matmul instead of three
        x = last_in.squeeze(0)  # [1,B,D] -> [B,D] to avoid permutes
        W = mha.in_proj_weight  # [3E, E]
        b = mha.in_proj_bias  # [3E] or None

        qkv = F.linear(x, W, b)  # [B, 3E] - single GEMM!
        q, k, v = qkv.split(E, dim=-1)  # each [B, E]

        # Keep 4D for Flash Attention (no reshape to B*H)
        q = q.view(B, H, 1, Dh)  # [B,H,1,Dh]
        k = k.view(B, H, 1, Dh)  # [B,H,1,Dh]
        v = v.view(B, H, 1, Dh)  # [B,H,1,Dh]

        # Interpret/initialize cache (always 4-tuple now)
        if kv_prev is None:
            # New cache: allocate EXACT capacity upfront (no reallocation ever!)
            cap = cap_hint if cap_hint is not None else 128
            K_buf = k.new_empty((B, H, cap, Dh))
            V_buf = v.new_empty((B, H, cap, Dh))
            t = 0
        else:
            # Existing cache: unpack 4-tuple
            K_buf, V_buf, t, cap = kv_prev

        # No growing needed - we allocated exact capacity upfront!
        # Just assert for safety in debug mode
        assert t < cap, f"KV cache overflow: t={t}, cap={cap}"

        # O(1) in-place append using copy_ for better performance
        K_buf[:, :, t:t + 1, :].copy_(k)
        V_buf[:, :, t:t + 1, :].copy_(v)
        t += 1

        # Keep 4D for Flash Attention
        k_sdp = K_buf[:, :, :t, :]  # [B,H,t,Dh]
        v_sdp = V_buf[:, :, :t, :]  # [B,H,t,Dh]

        # Ensure bf16 and contiguous for Flash
        q = q.contiguous()
        k_sdp = k_sdp.contiguous()
        v_sdp = v_sdp.contiguous()

        # Debug: Check Flash eligibility
        self._flash_debug_guard(q, k_sdp, v_sdp, True, "self-attn")

        # Flash-compatible call: no mask, causal=True for self-attention
        ctx = F.scaled_dot_product_attention(
            q, k_sdp, v_sdp,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True  # Changed to True for Flash compatibility
        )  # [B,H,1,Dh]

        # Merge heads -> [1,B,E]
        ctx = ctx.permute(0, 2, 1, 3).contiguous().view(B, 1, E).transpose(0, 1)

        # Out projection
        sa_out = mha.out_proj(ctx)  # [1,B,E]

        return sa_out, (K_buf, V_buf, t, cap)

    def forward(  # type: ignore
            self,
            tgt: Tensor,
            memory: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            memory_kv=None,  # NEW: (K_mem, V_mem) or None
            self_kv=None,  # NEW: self-attn KV cache (now flexible tuple)
            max_pred_len: Optional[int] = None,  # For capacity hint
    ) -> Tuple[Tensor, Optional[Tuple]]:
        """
        Args:
            same as TMTransformerDecoder
        Returns:
            Tuple[Tensor, Optional[Tuple[Tensor,Tensor]]]:
                - embedding of last tag: (1,bsz,hidden_dim)
                - self-attention KV cache (always None for now in Step 3)
        """

        # From PyTorch but modified to only use the last tag
        tgt_last_tok = tgt[-1:, :, :]

        # Detect incremental mode vs full sequence mode
        if tgt.size(0) == 1:
            # Incremental mode: always use KV cache (seed if self_kv is None)
            # Use max_pred_len for EXACT capacity allocation (no reallocs!)
            cap_hint = max_pred_len + 1 if max_pred_len is not None else 128
            sa_out, self_kv_new = self._sa_kv_step(tgt_last_tok, self_kv, cap_hint=cap_hint)
        else:
            # Full-sequence path (training / non-incremental)
            sa_out = self.self_attn(
                tgt_last_tok,
                tgt,
                tgt,
                attn_mask=None,  # None, because we only care about the last tag
                key_padding_mask=tgt_key_padding_mask,
                need_weights=False,  # Optimization: Don't compute attention weights
            )[0]
            self_kv_new = None

        tgt_last_tok = self.norm1(tgt_last_tok + self.dropout1(sa_out))

        # cross-attn
        if memory is not None:
            if memory_kv is not None:
                # -- Flash-compatible cross-attn using precomputed K_mem/V_mem --
                mha = self.multihead_attn
                E = mha.embed_dim
                H = mha.num_heads
                Dh = E // H

                K_mem, V_mem = memory_kv  # [B, H, S, Dh]
                B, H, S, Dh = K_mem.shape  # Extract dimensions

                # Q projection (query comes from decoder side)
                W_q = mha.in_proj_weight[:E, :]
                b_q = mha.in_proj_bias[:E] if mha.in_proj_bias is not None else None
                q = F.linear(tgt_last_tok, W_q, b_q)  # [1,B,E]
                # Keep 4D for Flash Attention
                q = q.permute(1, 0, 2).contiguous().view(B, 1, H, Dh).transpose(1, 2)  # [B,H,1,Dh]

                # K_mem,V_mem already [B,H,S,Dh] - perfect for Flash
                k = K_mem  # [B,H,S,Dh]
                v = V_mem  # [B,H,S,Dh]

                # Ensure contiguous for Flash
                q = q.contiguous()
                k = k.contiguous()
                v = v.contiguous()

                # Debug: Check Flash eligibility
                self._flash_debug_guard(q, k, v, False, "cross-attn")

                # Flash-compatible: No mask for better performance (assumes no padding)
                # If you have fixed CNN grid with no padding, this is optimal
                ctx = F.scaled_dot_product_attention(
                    q, k, v, 
                    attn_mask=None,  # No mask for Flash eligibility
                    dropout_p=0.0, 
                    is_causal=False
                )  # [B,H,1,Dh]

                # back to [1,B,E]
                ctx = ctx.permute(0, 2, 1, 3).contiguous().view(B, 1, E).transpose(0, 1)

                # ✅ IMPORTANT: apply the MHA output projection, same as the stock module
                ctx = mha.out_proj(ctx)  # preserves shape [1,B,E]

                tmp_tgt = ctx
            else:
                # fallback: regular MHA does its own projections
                tmp_tgt = self.multihead_attn(
                    tgt_last_tok, memory, memory,
                    attn_mask=memory_mask,
                    key_padding_mask=memory_key_padding_mask,
                    need_weights=False,
                )[0]

            tgt_last_tok = tgt_last_tok + self.dropout2(tmp_tgt)
            tgt_last_tok = self.norm2(tgt_last_tok)

        tmp_tgt = self.linear2(
            self.dropout(self.activation(self.linear1(tgt_last_tok)))
        )
        tgt_last_tok = tgt_last_tok + self.dropout3(tmp_tgt)
        tgt_last_tok = self.norm3(tgt_last_tok)
        return tgt_last_tok, self_kv_new  # Return tuple now


class Tag_Transformer(nn.Module):
    """
    "Attention Is All You Need" - https://arxiv.org/abs/1706.03762
    """

    def __init__(
            self,
            device,
            vocab_size,
            td_encode,
            embed_dim,
            encoder_layers,
            decoder_layers,
            enc_image_size,
            dropout=0.1,
            n_heads=4,
            dim_ff=1024,
    ):
        super(Tag_Transformer, self).__init__()

        self._device = device
        self._n_heads = n_heads
        self._embedding = nn.Embedding(vocab_size, embed_dim)
        self._positional_encoding = PositionalEncoding(embed_dim)
        self._td_encode = td_encode

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=dim_ff
        )
        self._encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=encoder_layers, enable_nested_tensor=False
        )

        self._decoder = TMTransformerDecoder(
            TMTransformerDecoderLayer(
                d_model=embed_dim,
                nhead=n_heads,
                dim_feedforward=dim_ff,
            ),
            num_layers=decoder_layers,
        )

        self._decoder_dim = embed_dim
        self._enc_image_size = enc_image_size
        self._input_filter = u.resnet_block(stride=1)
        self._fc = nn.Linear(embed_dim, vocab_size)

    def step_fullprefix(
            self,
            t: int,
            tgt_emb_buf: torch.Tensor,  # [Tmax+1, B, D]
            memory: torch.Tensor,  # [S, B, D]
            cache=None,
            memory_kv=None,
            sa_kv_cache=None,  # NEW: self-attention KV cache
            max_pred_len: Optional[int] = None,  # For capacity hint
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[List]]:
        # Always use incremental path (Checkpoint C)
        # Pass only the current token
        tgt_current = tgt_emb_buf[t:t + 1, :, :]  # [1, B, D] - current token with PE
        last_h_1BD, sa_kv_out = self._decoder(
            tgt_current, memory=memory, cache=None, memory_key_padding_mask=None,
            memory_kv=memory_kv, sa_kv_cache=sa_kv_cache, max_pred_len=max_pred_len
        )
        # Convert [1, B, D] -> [B, D] for compatibility
        last_H = last_h_1BD.squeeze(0)  # [B, D]
        return last_H, None, sa_kv_out  # [B, D], None (no tag cache), kv_cache

    def precompute_mem_kv(self, mem_enc: torch.Tensor):
        """
        mem_enc: [S, B, D] encoder output (your current shape)
        returns: List[(K_mem, V_mem)] per decoder layer; shapes [B, H, S, Dh]
        """
        mem = mem_enc.transpose(0, 1).contiguous()  # [B, S, D]
        B, S, D = mem.shape

        mem_kv = []
        for layer in self._decoder.layers:  # TMTransformerDecoderLayer
            mha = layer.multihead_attn
            E = mha.embed_dim
            H = mha.num_heads
            Dh = E // H

            W = mha.in_proj_weight  # [3E, E]
            b = mha.in_proj_bias  # [3E] or None
            # slice K and V projections
            W_k = W[E:2 * E, :]
            W_v = W[2 * E:, :]
            b_k = b[E:2 * E] if b is not None else None
            b_v = b[2 * E:] if b is not None else None

            K = F.linear(mem, W_k, b_k)  # [B, S, E]
            V = F.linear(mem, W_v, b_v)  # [B, S, E]
            # -> [B, H, S, Dh]
            K = K.view(B, S, H, Dh).transpose(1, 2).contiguous()
            V = V.view(B, S, H, Dh).transpose(1, 2).contiguous()
            mem_kv.append((K, V))
        return mem_kv

    def inference(self, enc_inputs, tags, tag_lens, num_cells):
        # CNN backbone image encoding
        enc_inputs = self._input_filter(enc_inputs.permute(0, 3, 1, 2)).permute(
            0, 2, 3, 1
        )

        batch_size = enc_inputs.size(0)
        encoder_dim = enc_inputs.size(-1)

        enc_inputs = enc_inputs.view(batch_size, -1, encoder_dim).to(self._device)

        enc_inputs = enc_inputs.permute(1, 0, 2)
        positions = enc_inputs.shape[0]
        # Transformer Encoder Encoded Image mask need to check if its useful
        encoder_mask = torch.zeros(
            (batch_size * self._n_heads, positions, positions), device=self._device
        ) == torch.ones(
            (batch_size * self._n_heads, positions, positions), device=self._device
        )

        # Transformer Encoder
        encoder_out = self._encoder(enc_inputs, mask=encoder_mask)

        decode_lengths = (tag_lens - 1).tolist()

        tgt = self._positional_encoding(self._embedding(tags).permute(1, 0, 2))

        decoded = self._decoder(tgt, memory=encoder_out)
        decoded = decoded.permute(1, 0, 2)
        predictions = self._fc(decoded)
        return predictions, decode_lengths
