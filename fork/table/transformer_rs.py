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
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TMTransformerDecoder(nn.TransformerDecoder):
    def forward(  # type: ignore
        self,
        tgt: Tensor,
        memory: Optional[Tensor] = None,
        cache: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_kv=None,   # NEW
        sa_kv_cache=None,  # NEW: self-attention KV cache list
    ):
        """
        Args:
            tgt (Tensor): encoded tags. (tags_len,bsz,hidden_dim)
            memory (Tensor): encoded image (enc_image_size,bsz,hidden_dim)
            cache (Optional[Tensor]): None during training, only used during inference.
        Returns:
            output (Tensor): (tags_len,bsz,hidden_dim)
        """

        output = tgt

        # cache
        tag_cache = []
        new_sa_kv_cache = [] if sa_kv_cache is not None else None
        
        for i, mod in enumerate(self.layers):
            # Use KV cache only for layer 0
            layer_self_kv = None
            if sa_kv_cache is not None and i == 0:  # only layer 0 uses KV
                layer_self_kv = sa_kv_cache[0]
            
            result = mod(
                output, memory,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_kv=None if memory_kv is None else memory_kv[i],
                self_kv=layer_self_kv,
            )
            
            # Handle the new tuple return format
            if isinstance(result, tuple):
                output_last, layer_kv_new = result
            else:
                # Fallback for compatibility
                output_last, layer_kv_new = result, None
            
            tag_cache.append(output_last)
            
            # Rebuild full prefix to feed next layer (unchanged)
            if cache is not None:
                output = torch.cat([cache[i], output_last], dim=0)
            else:
                # first step: no history; output_last is the only row
                output = output_last
                
            # Build new KV cache
            if new_sa_kv_cache is not None:
                new_sa_kv_cache.append(layer_kv_new if i == 0 else None)

        if cache is not None:
            out_cache = torch.cat([cache, torch.stack(tag_cache, dim=0)], dim=1)
        else:
            out_cache = torch.stack(tag_cache, dim=0)

        return output, out_cache, new_sa_kv_cache  # Return 3 items now


class TMTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def _sa_kv_step(
        self,
        last_in: torch.Tensor,                  # [1, B, D]  (layer input for the *last* token)
        kv_prev: Optional[Tuple[torch.Tensor, torch.Tensor]],  # (K_prev, V_prev) each [B,H,Tprev,Dh]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Incremental self-attention with KV cache.
        Return:
          sa_out: [1, B, D]   (self-attention output, before residual)
          kv_new: (K_cat, V_cat) with new step appended
        """
        mha = self.self_attn
        E  = mha.embed_dim
        H  = mha.num_heads
        Dh = E // H

        # Slice projections (same math as nn.MultiheadAttention)
        W = mha.in_proj_weight    # [3E, E]
        b = mha.in_proj_bias      # [3E] or None

        W_q = W[:E,     :]
        W_k = W[E:2*E,  :]
        W_v = W[2*E:,   :]
        b_q = b[:E]     if b is not None else None
        b_k = b[E:2*E]  if b is not None else None
        b_v = b[2*E:]   if b is not None else None

        # Project only the last token
        # last_in: [1,B,D] -> q,k,v: [1,B,E]
        q = F.linear(last_in, W_q, b_q)
        k = F.linear(last_in, W_k, b_k)
        v = F.linear(last_in, W_v, b_v)

        # Split heads
        # -> [B,H,1,Dh]
        B = q.size(1)
        q = q.view(1, B, H, Dh).permute(1, 2, 0, 3)  # [B,H,1,Dh]
        k = k.view(1, B, H, Dh).permute(1, 2, 0, 3)  # [B,H,1,Dh]
        v = v.view(1, B, H, Dh).permute(1, 2, 0, 3)  # [B,H,1,Dh]

        if kv_prev is not None:
            K_prev, V_prev = kv_prev                      # [B,H,Tprev,Dh]
            K_cat = torch.cat([K_prev, k], dim=2).contiguous()         # [B,H,Tprev+1,Dh]
            V_cat = torch.cat([V_prev, v], dim=2).contiguous()
        else:
            K_cat, V_cat = k, v

        # Attention: single-step query over cached keys/values
        # q: [B,H,1,Dh], K_cat: [B,H,T,Dh] -> scores: [B,H,1,T]
        scores = torch.matmul(q, K_cat.transpose(-2, -1)) * (1.0 / (Dh ** 0.5))
        attn   = torch.softmax(scores, dim=-1)            # [B,H,1,T]
        ctx    = torch.matmul(attn, V_cat)                # [B,H,1,Dh]

        # Merge heads -> [1,B,E]
        ctx = ctx.permute(2, 0, 1, 3).contiguous().view(1, B, E)

        # Out projection (must match stock MHA)
        sa_out = mha.out_proj(ctx)                        # [1,B,E]

        return sa_out, (K_cat, V_cat)

    def forward(  # type: ignore
        self,
        tgt: Tensor,
        memory: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_kv=None,   # NEW: (K_mem, V_mem) or None
        self_kv: Optional[Tuple[Tensor,Tensor]] = None,  # NEW: self-attn KV cache (UNUSED for now)
    ) -> Tuple[Tensor, Optional[Tuple[Tensor,Tensor]]]:
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

        # Use KV cache if provided, otherwise use stock MHA
        use_sa_kv = (self_kv is not None) and (tgt.size(0) >= 1)  # guard

        if use_sa_kv:
            sa_out, self_kv_new = self._sa_kv_step(tgt_last_tok, self_kv)
        else:
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
                # -- custom cross-attn using precomputed K_mem/V_mem --
                mha = self.multihead_attn
                E = mha.embed_dim
                H = mha.num_heads
                Dh = E // H

                K_mem, V_mem = memory_kv                      # [B, H, S, Dh]
                B, H, S, Dh = K_mem.shape                     # Extract dimensions

                # Q projection (query comes from decoder side)
                W_q = mha.in_proj_weight[:E, :]
                b_q = mha.in_proj_bias[:E] if mha.in_proj_bias is not None else None
                q = F.linear(tgt_last_tok, W_q, b_q)                  # [1,B,E]
                q = q.permute(1,0,2).contiguous().view(B, 1, H, Dh).transpose(1,2).reshape(B*H, 1, Dh)

                # K_mem,V_mem: [B,H,S,Dh] -> [B*H, S, Dh]
                k = K_mem.reshape(B*H, S, Dh)
                v = V_mem.reshape(B*H, S, Dh)

                # Handle padding mask for SDPA
                attn_mask = None
                if memory_key_padding_mask is not None:  # [B,S] True=pad
                    # SDPA needs an additive mask: True -> -inf, False -> 0.0
                    mask = memory_key_padding_mask.float().masked_fill(
                        memory_key_padding_mask, float('-inf')
                    ).masked_fill(~memory_key_padding_mask, 0.0)  # [B,S]
                    attn_mask = mask.unsqueeze(1).expand(B, H, S).reshape(B*H, 1, S)

                # SDPA (no dropout in eval)
                ctx = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
                )  # [B*H,1,Dh]

                # back to [1,B,E]
                ctx = ctx.reshape(B, H, 1, Dh).transpose(1,2).contiguous().view(1, B, E)

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
        tgt_emb_buf: torch.Tensor,   # [Tmax+1, B, D]
        memory: torch.Tensor,        # [S, B, D]
        cache=None,
        memory_kv=None,
        sa_kv_cache=None,           # NEW: self-attention KV cache
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[List]]:
        # feed the full prefix up to t (exactly your current behavior)
        tgt = tgt_emb_buf[: t + 1, :, :]             # [t+1, B, D]
        decoded, cache_out, sa_kv_out = self._decoder(
            tgt, memory=memory, cache=cache, memory_key_padding_mask=None,
            memory_kv=memory_kv, sa_kv_cache=sa_kv_cache
        )
        return decoded[-1, :, :], cache_out, sa_kv_out  # [B, D], cache, kv_cache

    def precompute_mem_kv(self, mem_enc: torch.Tensor):
        """
        mem_enc: [S, B, D] encoder output (your current shape)
        returns: List[(K_mem, V_mem)] per decoder layer; shapes [B, H, S, Dh]
        """
        mem = mem_enc.transpose(0, 1).contiguous()   # [B, S, D]
        B, S, D = mem.shape

        mem_kv = []
        for layer in self._decoder.layers:  # TMTransformerDecoderLayer
            mha = layer.multihead_attn
            E = mha.embed_dim
            H = mha.num_heads
            Dh = E // H

            W = mha.in_proj_weight    # [3E, E]
            b = mha.in_proj_bias      # [3E] or None
            # slice K and V projections
            W_k = W[E:2*E, :]
            W_v = W[2*E: , :]
            b_k = b[E:2*E] if b is not None else None
            b_v = b[2*E: ] if b is not None else None

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
