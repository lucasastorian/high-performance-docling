# fork/table/batched_decoder_kv.py
# Batched AR decoder with KV-cache + SDPA + precomputed cross-attn memory
# Minimal GPU-timer sampling and zero-allocation inner loop.

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

PROFILE_EVERY = int(os.getenv("AR_PROFILE_EVERY", "8"))
COMPACT_EVERY = int(os.getenv("AR_COMPACT_EVERY", "0"))  # 0 = disabled

# ---- SDPA guard (Flash kernels when eligible) ----
SDPA_CTX = torch.backends.cuda.sdp_kernel
SDPA_ARGS = dict(enable_flash=True, enable_math=True, enable_mem_efficient=True)


# -------------------------------
# Lean decoder blocks (step-wise)
# -------------------------------
class KVCache:
    """
    Per-layer KV cache for self-attn: preallocated [B,h,Tmax,d_h]
    We grow a 't' cursor each step; no reallocation.
    """
    def __init__(self, B, n_heads, Tmax, d_head, device, dtype):
        self.k = torch.empty(B, n_heads, Tmax, d_head, device=device, dtype=dtype)
        self.v = torch.empty(B, n_heads, Tmax, d_head, device=device, dtype=dtype)
        self.t = 0  # current length

    def append(self, k_t, v_t):
        # k_t/v_t: [B,h,1,d]
        self.k[:, :, self.t:self.t+1, :].copy_(k_t)
        self.v[:, :, self.t:self.t+1, :].copy_(v_t)
        self.t += 1

    def view(self):
        # Return current prefixes [B,h,t,d]
        return self.k[:, :, :self.t, :], self.v[:, :, :self.t, :]


class SDPAProjection(nn.Module):
    """Simple QKV projections (or Q only)."""
    def __init__(self, d_model, n_heads, bias=True):
        super().__init__()
        self.nh = n_heads
        self.dk = d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=bias)
        self.k = nn.Linear(d_model, d_model, bias=bias)
        self.v = nn.Linear(d_model, d_model, bias=bias)
        self.o = nn.Linear(d_model, d_model, bias=bias)

    def split(self, x):
        # x: [B, D] -> [B, h, 1, d]
        B, D = x.shape
        x = x.view(B, self.nh, self.dk)
        return x[:, :, None, :]

    def split_mem(self, xSB):
        # xSB: [S,B,D] -> [B,h,S,d]
        S, B, D = xSB.shape
        x = xSB.permute(1, 0, 2).contiguous().view(B, self.nh, S, self.dk)
        return x


class DecoderLayerStep(nn.Module):
    """
    One decoder layer, step-wise:
      - self-attn on last token with KV-cache
      - cross-attn against fixed memory (K_mem/V_mem precomputed)
      - MLP
    """
    def __init__(self, d_model, n_heads, dim_ff=1024, dropout=0.0):
        super().__init__()
        self.nh = n_heads
        self.dk = d_model // n_heads

        self.self_proj = SDPAProjection(d_model, n_heads)
        self.cross_q = nn.Linear(d_model, d_model)
        self.cross_k = nn.Linear(d_model, d_model)
        self.cross_v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.ff1 = nn.Linear(d_model, dim_ff)
        self.ff2 = nn.Linear(dim_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def precompute_memory_kv(self, memory_SBD):
        """Called once before decoding: build [B,h,S,d] K_mem/V_mem"""
        S, B, D = memory_SBD.shape
        K = self.cross_k(memory_SBD)  # [S,B,D]
        V = self.cross_v(memory_SBD)  # [S,B,D]
        # reshape to [B,h,S,d]
        nh = self.nh; dk = D // nh
        K = K.permute(1,0,2).contiguous().view(B, nh, S, dk)
        V = V.permute(1,0,2).contiguous().view(B, nh, S, dk)
        return K, V

    def step(self, x_last_BD, kv_cache: KVCache, K_mem_BhSd, V_mem_BhSd):
        """
        x_last_BD: [B, D] last token hidden state
        kv_cache:  per-layer KV cache for self-attn
        K_mem/V_mem: [B,h,S,d]
        returns: next_BD
        """
        B, D = x_last_BD.shape
        nh, dk = self.nh, self.dk

        # --- Self-attn (single query) ---
        y = self.norm1(x_last_BD)
        q = self.self_proj.q(y).view(B, nh, dk)[:, :, None, :]    # [B,h,1,d]
        k_t = self.self_proj.k(y).view(B, nh, dk)[:, :, None, :]
        v_t = self.self_proj.v(y).view(B, nh, dk)[:, :, None, :]
        kv_cache.append(k_t, v_t)
        K_self, V_self = kv_cache.view()                           # [B,h,t,d]

        with SDPA_CTX(**SDPA_ARGS):
            self_out = F.scaled_dot_product_attention(q, K_self, V_self, is_causal=False)  # [B,h,1,d]
        self_out = self_out.reshape(B, nh*dk)
        x = x_last_BD + self.dropout(self.self_proj.o(self_out))

        # --- Cross-attn ---
        y2 = self.norm2(x)
        q2 = self.cross_q(y2).view(B, nh, dk)[:, :, None, :]      # [B,h,1,d]
        with SDPA_CTX(**SDPA_ARGS):
            cross = F.scaled_dot_product_attention(q2, K_mem_BhSd, V_mem_BhSd, is_causal=False)
        cross = cross.reshape(B, nh*dk)
        x = x + self.dropout(self.out(cross))

        # --- MLP ---
        y3 = self.norm3(x)
        x = x + self.dropout(self.ff2(self.act(self.ff1(y3))))
        return x


class DecoderStack(nn.Module):
    """N layers of step-wise DecoderLayerStep + per-layer caches."""
    def __init__(self, d_model, n_layers, n_heads, dim_ff=1024, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayerStep(d_model, n_heads, dim_ff, dropout) for _ in range(n_layers)
        ])
        self.d_model = d_model
        self.n_heads = n_heads

    def build_state(self, B, Tmax, device, dtype):
        d_head = self.d_model // self.n_heads
        return [KVCache(B, self.n_heads, Tmax, d_head, device, dtype) for _ in self.layers]

    def precompute_memory(self, memory_SBD):
        # returns lists of K_mem/V_mem per layer
        Ks, Vs = [], []
        for L in self.layers:
            K_mem, V_mem = L.precompute_memory_kv(memory_SBD)
            Ks.append(K_mem)
            Vs.append(V_mem)
        return Ks, Vs

    def step(self, x_last_BD, kv_state: List[KVCache], K_mems, V_mems):
        x = x_last_BD
        for i, L in enumerate(self.layers):
            x = L.step(x, kv_state[i], K_mems[i], V_mems[i])
        return x


# -------------------------------
# Tag_Transformer (decoder part)
# -------------------------------
class Tag_Transformer_Fast(nn.Module):
    """
    Keep your encoder as-is; swap decoder to step-wise stack above.
    Public bits used by the AR decoder: _embedding, _positional_encoding, _decoder, _fc.
    """
    def __init__(self, device, vocab_size, embed_dim, decoder_layers, n_heads=4, dim_ff=1024, dropout=0.0):
        super().__init__()
        self._device = device
        self._embedding = nn.Embedding(vocab_size, embed_dim)
        self._positional_encoding = PositionalEncoding(embed_dim=embed_dim)  # reuse your PE class
        self._decoder = DecoderStack(embed_dim, decoder_layers, n_heads, dim_ff, dropout)
        self._fc = nn.Linear(embed_dim, vocab_size)


# -------------------------------
# Batched AR Decoder (KV/SDPA)
# -------------------------------
@dataclass
class BatchRuntime:
    decoded_tags: torch.Tensor  # [Tmax+1, B] long
    lengths: torch.Tensor       # [B] int
    finished: torch.Tensor      # [B] bool
    first_lcel: torch.Tensor    # [B] bool
    skip_next: torch.Tensor     # [B] bool
    prev_ucel: torch.Tensor     # [B] bool
    line_num: torch.Tensor      # [B] long
    bbox_ind: torch.Tensor      # [B] long
    # Tag-H buffer (preallocated): [B,Kmax,D]
    tag_H_buf: torch.Tensor
    k_counters: torch.Tensor    # [B] int
    # Span bookkeeping
    span_starts: torch.Tensor   # [B,Kspan] long (init -1)
    span_ends: torch.Tensor     # [B,Kspan] long (init -1)
    span_cnt: torch.Tensor      # [B] long
    # Embedding buffer (prefix growth): [Tmax+1,B,D]
    tgt_emb_buf: torch.Tensor


class BatchedTableDecoderKV:
    def __init__(self, model, device: str):
        self.model = model
        self.device = device
        self._prof = model._prof

        # Pointers
        self.tt = model._tag_transformer
        wm = model._init_data["word_map"]["word_map_tag"]

        # IDs
        self.start_id = torch.tensor(wm["<start>"], device=device, dtype=torch.long)
        self.end_id   = torch.tensor(wm["<end>"],   device=device, dtype=torch.long)
        self.lcel_id  = torch.tensor(wm["lcel"], device=device, dtype=torch.long) if "lcel" in wm else None
        self.fcel_id  = torch.tensor(wm["fcel"], device=device, dtype=torch.long) if "fcel" in wm else None
        self.ucel_id  = torch.tensor(wm["ucel"], device=device, dtype=torch.long) if "ucel" in wm else None
        self.xcel_id  = torch.tensor(wm["xcel"], device=device, dtype=torch.long) if "xcel" in wm else None
        self.nl_id    = torch.tensor(wm["nl"],   device=device, dtype=torch.long) if "nl"   in wm else None

        # LUTs
        emit_names = ["fcel","ecel","ched","rhed","srow","nl","ucel"]
        skip_names = ["nl","ucel","xcel"]
        V = self.tt._fc.out_features
        self.emit_lut = torch.zeros(V, dtype=torch.bool, device=device)
        self.skip_lut = torch.zeros(V, dtype=torch.bool, device=device)
        for n in emit_names:
            if n in wm: self.emit_lut[wm[n]] = True
        for n in skip_names:
            if n in wm: self.skip_lut[wm[n]] = True

    # ----- helpers -----
    @staticmethod
    def _trim_to_end(seq_1TB: torch.Tensor, end_id: int):
        # seq_1TB: [T,B], single column used
        seq_list = seq_1TB.tolist()
        try:
            j = seq_list.index(end_id)
            return seq_list[:j+1]
        except ValueError:
            return seq_list

    # ----- main -----
    @torch.inference_mode()
    def predict_batched(self, enc_out_batch_NCHW, mem_enc_SBD, max_steps: int, timer=None):
        device = self.device
        tt = self.tt
        FC = tt._fc
        E = tt._embedding
        PE = tt._positional_encoding
        DEC = tt._decoder  # DecoderStack

        B = enc_out_batch_NCHW.size(0)
        D = E.embedding_dim
        Tmax = min(max_steps, self.model._max_pred_len)

        # Precompute positional rows once [Tmax+1, D]
        pe_rows = PE.pe[:Tmax+1, 0].to(device=device)

        # Precompute memory K/V for all decoder layers
        if timer: timer.start_section("precompute_mem_kv")
        K_mems, V_mems = DEC.precompute_memory(mem_enc_SBD)  # lists len=L, each [B,h,S,d]
        if timer: timer.end_section("precompute_mem_kv")

        # Build KV-cache per layer
        kv_state = DEC.build_state(B, Tmax, device, dtype=mem_enc_SBD.dtype)

        # --- Allocate runtime buffers (no per-step allocs) ---
        decoded_tags = torch.full((Tmax+1, B), self.end_id.item(), dtype=torch.long, device=device)
        decoded_tags[0] = self.start_id

        lengths   = torch.zeros(B, dtype=torch.int32, device=device)
        finished  = torch.zeros(B, dtype=torch.bool,  device=device)
        first_lcel= torch.ones(B,  dtype=torch.bool,  device=device)
        skip_next = torch.ones(B,  dtype=torch.bool,  device=device)
        prev_ucel = torch.zeros(B, dtype=torch.bool,  device=device)
        line_num  = torch.zeros(B, dtype=torch.long,  device=device)
        bbox_ind  = torch.zeros(B, dtype=torch.long,  device=device)

        Kmax   = max(1, Tmax // 2)
        Kspan  = max(1, Tmax // 2)
        tag_H_buf   = torch.empty(B, Kmax, D, device=device, dtype=mem_enc_SBD.dtype)
        k_counters  = torch.zeros(B, dtype=torch.int32, device=device)
        span_starts = torch.full((B, Kspan), -1, device=device, dtype=torch.long)
        span_ends   = torch.full((B, Kspan), -1, device=device, dtype=torch.long)
        span_cnt    = torch.zeros(B, device=device, dtype=torch.long)

        tgt_emb_buf = torch.empty(Tmax+1, B, D, device=device, dtype=mem_enc_SBD.dtype)
        tgt_emb_buf[0] = E(decoded_tags[0]) + pe_rows[0]

        rt = BatchRuntime(decoded_tags, lengths, finished, first_lcel, skip_next, prev_ucel,
                          line_num, bbox_ind, tag_H_buf, k_counters, span_starts, span_ends,
                          span_cnt, tgt_emb_buf)

        # ---- AR loop ----
        if timer: timer.start_section("ar_loop")
        t = 0

        for step in range(Tmax):
            # Sampled timing inside the loop
            do_profile = (PROFILE_EVERY > 0) and (step % PROFILE_EVERY == 0) and timer is not None
            if do_profile: timer.start_section("ar_step_self")
            # forward one step through all layers
            x_last = rt.tgt_emb_buf[t]  # [B,D]
            x_last = DEC.step(x_last, kv_state, K_mems, V_mems)
            if do_profile: timer.end_section("ar_step_self")

            if do_profile: timer.start_section("ar_step_fc")
            logits = FC(x_last)             # [B,V]
            new_tags = logits.argmax(dim=1) # [B]
            if do_profile: timer.end_section("ar_step_fc")

            # masks
            is_end  = (new_tags == self.end_id)
            is_lcel = (new_tags == self.lcel_id) if self.lcel_id is not None else torch.zeros_like(is_end)
            emit    = self.emit_lut[new_tags]
            skipnow = self.skip_lut[new_tags]

            # rule fixes
            if (self.xcel_id is not None) and (self.lcel_id is not None):
                new_tags = torch.where((rt.line_num == 0) & (new_tags == self.xcel_id), self.lcel_id, new_tags)
            if (self.ucel_id is not None) and (self.lcel_id is not None) and (self.fcel_id is not None):
                new_tags = torch.where(rt.prev_ucel & (new_tags == self.lcel_id), self.fcel_id, new_tags)

            # finalize finished
            rt.finished |= is_end
            end_vec = self.end_id.expand_as(new_tags)
            new_tags = torch.where(rt.finished, end_vec, new_tags)

            # write token
            t += 1
            rt.decoded_tags[t] = new_tags

            # grow embedding prefix for next step
            if t < Tmax:
                rt.tgt_emb_buf[t] = E(new_tags) + pe_rows[t]

            # book-keeping
            rt.lengths = torch.where(~rt.finished, rt.lengths + 1, rt.lengths)

            # collect bbox features (append once per step)
            m_emit = (~rt.skip_next) & emit & (~rt.finished)
            m_first = rt.first_lcel & is_lcel & (~rt.finished)
            append_mask = m_emit | m_first
            append_idx = append_mask.nonzero(as_tuple=False).squeeze(1)

            if append_idx.numel() > 0:
                rows = x_last.index_select(0, append_idx)   # [K,D]
                slots = rt.k_counters[append_idx].long()
                base = append_idx * rt.tag_H_buf.size(1) + slots
                rt.tag_H_buf.view(-1, D).index_copy_(0, base, rows)
                rt.k_counters[append_idx] += 1

                # span starts for first_lcel
                first_idx = (m_first & append_mask).nonzero(as_tuple=False).squeeze(1)
                if first_idx.numel() > 0:
                    slot = rt.span_cnt[first_idx]
                    rt.span_starts[first_idx, slot] = rt.bbox_ind[first_idx]
                    rt.span_cnt[first_idx] += 1

                # span ends for emits that are not first_lcel
                end_idx = (m_emit & (~m_first) & (~rt.finished)).nonzero(as_tuple=False).squeeze(1)
                if end_idx.numel() > 0:
                    slot = (rt.span_cnt[end_idx] - 1).clamp_min(0)
                    rt.span_ends[end_idx, slot] = rt.bbox_ind[end_idx]

                rt.bbox_ind[append_idx] += 1

            # flag updates
            rt.first_lcel = ~is_lcel | rt.finished
            rt.skip_next = skipnow
            rt.prev_ucel = (new_tags == self.ucel_id) if self.ucel_id is not None else torch.zeros_like(rt.finished)
            if self.nl_id is not None:
                rt.line_num += (new_tags == self.nl_id).to(rt.line_num.dtype)

            # optional alive compaction (disabled by default)
            if COMPACT_EVERY and (step % COMPACT_EVERY == COMPACT_EVERY - 1):
                alive = (~rt.finished).nonzero(as_tuple=False).squeeze(1)
                if 0 < alive.numel() < B:
                    # NOTE: to keep this lightweight, we skip compaction by default.
                    # If enabled, you'd slice all per-batch buffers & caches by 'alive'.
                    pass

            # early exit if all finished
            if int((~rt.finished).sum().item()) == 0:
                break

        if timer: timer.end_section("ar_loop")

        # ---- Materialize outputs ----
        end_id_int = int(self.end_id.item())
        seqs = []
        for b in range(B):
            seq_len = min(t + 1, int(rt.lengths[b].item()) + 1)
            seqs.append(self._trim_to_end(rt.decoded_tags[:seq_len, b], end_id_int))

        # ---- BBox head (per table) ----
        outputs = []
        if timer: timer.start_section("bbox_decode")

        k_list = rt.k_counters.detach().cpu().tolist()
        span_cnt_host = rt.span_cnt.detach().cpu().tolist()

        for b, k_b in enumerate(k_list):
            if self.model._bbox and k_b > 0:
                enc_nchw = enc_out_batch_NCHW[b:b+1]
                tag_H_tensor = rt.tag_H_buf[b, :k_b]
                cls_logits, coords = self.model._bbox_decoder.inference(enc_nchw, tag_H_tensor)
            else:
                cls_logits = torch.empty(0, device=device)
                coords = torch.empty(0, device=device)

            # merge spans
            outputs.append(self._merge_spans_tensor(
                cls_logits, coords,
                rt.span_starts[b:b+1], rt.span_ends[b:b+1], rt.span_cnt[b:b+1]
            ))

        if timer: timer.end_section("bbox_decode")

        # format final tuples: (seq, cls, coord)
        final = []
        for b, (cls, coord) in enumerate(outputs):
            final.append((seqs[b], cls, coord))
        return final

    def _merge_spans_tensor(self, cls_logits, coords, starts, ends, cnt):
        device = coords.device if coords.numel() > 0 else self.device
        if coords.numel() == 0:
            return torch.empty(0, device=device), torch.empty(0, device=device)

        N = coords.size(0)
        keep = torch.ones(N, device=device, dtype=torch.bool)
        end_map = torch.full((N,), -1, device=device, dtype=torch.long)
        nspans = int(cnt[0].item()) if cnt.numel() > 0 else 0

        if nspans > 0:
            valid = (starts[0, :nspans] >= 0) & (ends[0, :nspans] >= 0)
            if valid.any():
                st = starts[0, :nspans][valid]
                ed = ends[0, :nspans][valid]
                ok = (st < N) & (ed < N)
                if ok.any():
                    end_map[st[ok]] = ed[ok]

        merged_cls, merged_coord = [], []
        for i in range(N):
            if not keep[i]:
                continue
            j = int(end_map[i].item())
            if 0 <= j < N:
                keep[j] = False
                merged_cls.append(cls_logits[i])
                merged_coord.append(self.model.mergebboxes(coords[i], coords[j]))
            else:
                merged_cls.append(cls_logits[i])
                merged_coord.append(coords[i])

        return (torch.stack(merged_cls) if merged_cls else torch.empty(0, device=device),
                torch.stack(merged_coord) if merged_coord else torch.empty(0, device=device))
