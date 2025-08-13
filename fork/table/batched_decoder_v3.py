# fork/table/batched_decoder_v3.py
# Optimized Batched AR decoder with GPT-5 Phase 1 & 2 optimizations implemented
# PHASE 1: Reduced GPU→CPU sync frequency - check "all finished" every 8 steps
# PHASE 2: Reduced nonzero() calls from 3 to 2 per step - CRITICAL: keep ~first_lcel in end condition
#
# TIMING SECTIONS (when timer is provided):
# - ar_loop: Total AR loop time
#   - ar_setup: Initial setup before AR loop
#   - ar_transformer: Transformer forward pass (first step only)
#   - ar_masking: Mask computations (first step only)
#   - ar_bbox_emit: BBox emission logic (first step only)
# - output_materialization: Converting tensors to output format
# - bbox_decode: Total bbox decoding time
#   - bbox_prep: Preparing counters and lists
#   - bbox_inference: Actual bbox inference (first table only)
#   - span_merging: Merging spans (measured on last table)

import torch
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


@dataclass
class BatchState:
    """State for batched AR decoding with preallocated buffers"""
    decoded_tags: torch.Tensor  # [Tmax+1, B] Long - preallocated
    lengths: torch.Tensor  # [B] Int - actual length per seq (excluding start)
    finished: torch.Tensor  # [B] Bool
    cache: Optional[torch.Tensor]  # transformer cache

    # Per-seq ragged buffers for bbox head (keep as flat list + indices for less sync)
    tag_H_flat: List[torch.Tensor]  # flat list of all hidden states
    tag_H_sample_ids: List[int]  # which sample each belongs to

    # Decoding flags (device tensors)
    first_lcel: torch.Tensor  # [B] Bool
    skip_next: torch.Tensor  # [B] Bool
    prev_ucel: torch.Tensor  # [B] Bool
    line_num: torch.Tensor  # [B] Long

    # Bbox indexing (device counters)
    bbox_ind: torch.Tensor  # [B] Long (running index)

    # Span bookkeeping - track one open span per sample
    open_span_start: List[int]  # [-1 if no open span, else start index] per sample


class BatchedTableDecoderV3:
    def __init__(self, model, device: str):
        self.model = model
        self.device = device
        self._prof = model._prof

        # Prebind references
        self.tt = model._tag_transformer
        wm = model._init_data["word_map"]["word_map_tag"]

        # Pre-create scalar tensors to avoid repeated allocations
        self.start_id = torch.tensor(wm["<start>"], device=device, dtype=torch.long)
        self.end_id = torch.tensor(wm["<end>"], device=device, dtype=torch.long)

        # Optional ids as tensors (or None)
        self.xcel_id = torch.tensor(wm["xcel"], device=device, dtype=torch.long) if "xcel" in wm else None
        self.lcel_id = torch.tensor(wm["lcel"], device=device, dtype=torch.long) if "lcel" in wm else None
        self.fcel_id = torch.tensor(wm["fcel"], device=device, dtype=torch.long) if "fcel" in wm else None
        self.ucel_id = torch.tensor(wm["ucel"], device=device, dtype=torch.long) if "ucel" in wm else None
        self.nl_id = torch.tensor(wm["nl"], device=device, dtype=torch.long) if "nl" in wm else None

        # Emission/skip sets (tensor on device)
        emit_names = ["fcel", "ecel", "ched", "rhed", "srow", "nl", "ucel"]
        self.emit_ids = torch.tensor(
            [wm[t] for t in emit_names if t in wm],
            device=device, dtype=torch.long
        )
        skip_names = ["nl", "ucel", "xcel"]
        self.skip_ids = torch.tensor(
            [wm[t] for t in skip_names if t in wm],
            device=device, dtype=torch.long
        )

        # OPTIMIZATION 1: Replace torch.isin with LUTs for O(1) lookup
        V = model._tag_transformer._fc.out_features  # Vocabulary size
        self.emit_lut = torch.zeros(V, dtype=torch.bool, device=device)
        if self.emit_ids.numel() > 0:
            self.emit_lut[self.emit_ids] = True

        self.skip_lut = torch.zeros(V, dtype=torch.bool, device=device)
        if self.skip_ids.numel() > 0:
            self.skip_lut[self.skip_ids] = True

    def _trim_sequence(self, seq_tensor: torch.Tensor, end_id: int) -> List[int]:
        """Trim sequence to first <end> token"""
        seq_list = seq_tensor.tolist()
        try:
            idx = seq_list.index(end_id)
            return seq_list[:idx + 1]
        except ValueError:
            return seq_list

    def _maybe_grow_buffer(self, buf: torch.Tensor, counters: torch.Tensor, needed_idx: torch.Tensor):
        """FIX 2: Grow buffer on demand if any sample is about to overflow"""
        if needed_idx.numel() == 0:
            return buf, counters

        # Check if any sample needs more space
        need = (counters[needed_idx] >= buf.size(1)).any()
        if not bool(need):
            return buf, counters

        # Grow by 50% + 8
        new_K = int(buf.size(1) * 1.5 + 8)
        B, Kold, D = buf.shape
        grown = torch.empty(B, new_K, D, device=buf.device, dtype=buf.dtype)
        grown[:, :Kold].copy_(buf)
        return grown, counters

    def _maybe_grow_spans(self, starts: torch.Tensor, ends: torch.Tensor, cnt: torch.Tensor, idx: torch.Tensor):
        """Grow span buffers on demand if any sample is about to overflow"""
        if idx.numel() == 0:
            return starts, ends, cnt

        need = (cnt[idx] >= starts.size(1)).any()
        if not bool(need):
            return starts, ends, cnt

        newK = int(starts.size(1) * 1.5 + 8)
        B = starts.size(0)
        s2 = torch.full((B, newK), -1, device=starts.device, dtype=starts.dtype)
        s2[:, :starts.size(1)] = starts
        e2 = torch.full((B, newK), -1, device=ends.device, dtype=ends.dtype)
        e2[:, :ends.size(1)] = ends
        return s2, e2, cnt

    @torch.inference_mode()
    def predict_batched(
            self,
            enc_out_batch: torch.Tensor,  # [B,C,H,W] NCHW - bbox decoder optimized for NCHW
            mem_enc: torch.Tensor,  # [S,B,D] encoder memory (precomputed, no duplicate processing)
            max_steps: int,
            timer=None  # Optional timer for detailed profiling
    ) -> List[Tuple[List[int], torch.Tensor, torch.Tensor]]:
        device = self.device
        tt = self.tt
        B = enc_out_batch.size(0)

        # OPTIMIZATION 3: Cache module references to avoid attribute lookups
        E = tt._embedding  # Embedding module
        FC = tt._fc  # Final classification layer
        decoder = tt._decoder  # Decoder module
        pos_enc = tt._positional_encoding  # Positional encoding

        # Clamp to model's max
        Tmax = min(max_steps, self.model._max_pred_len)

        # ---- Preallocate all buffers (Fix B: avoid reallocation) ----
        decoded_tags = torch.full((Tmax + 1, B), self.end_id.item(), dtype=torch.long, device=device)
        decoded_tags[0, :] = self.start_id

        # Device-only state tracking
        lengths = torch.zeros(B, dtype=torch.int32, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        first_lcel = torch.ones(B, dtype=torch.bool, device=device)
        skip_next = torch.ones(B, dtype=torch.bool, device=device)
        prev_ucel = torch.zeros(B, dtype=torch.bool, device=device)
        line_num = torch.zeros(B, dtype=torch.long, device=device)
        bbox_ind = torch.zeros(B, dtype=torch.long, device=device)

        # OPTIMIZATION 2: Preallocate tag hidden states buffer [B, Kmax, D]
        D_embed = E.embedding_dim
        Kmax = max(1, Tmax // 2)  # Safe upper bound for tag hidden states
        tag_H_buf = torch.empty(B, Kmax, D_embed, device=device)
        k_counters = torch.zeros(B, dtype=torch.int32, device=device)  # Track count per sample

        # OPTIMIZATION: Tensorized span tracking - no more Python dicts/lists
        Kspan = max(1, Tmax // 2)  # rough upper bound for spans
        span_starts = torch.full((B, Kspan), -1, device=device, dtype=torch.long)
        span_ends = torch.full((B, Kspan), -1, device=device, dtype=torch.long)
        span_cnt = torch.zeros(B, device=device, dtype=torch.long)

        # ---- Incremental embedding buffer optimization (Fix B: avoid O(T²)) ----
        D = E.embedding_dim
        tgt_emb_buf = torch.empty(Tmax + 1, B, D, device=device)

        # FIX 1: Clean PE shape handling - extract [Tmax+1, D] directly
        pe = pos_enc.pe  # [max_len, 1, D] positional encoding buffer
        pos_rows = pe[:Tmax + 1, 0]  # [Tmax+1, D] - clean 2D tensor, no awkward broadcasting

        # FIX 4: Prebuild end vector once
        end_vec = torch.full((B,), self.end_id.item(), device=device, dtype=torch.long)

        # OPTIMIZATION 2: Prebuild mask constants to avoid re-allocations
        onesB = torch.ones(B, dtype=torch.bool, device=device)
        zerosB = torch.zeros(B, dtype=torch.bool, device=device)

        # Initialize first step with start tokens
        start_row = decoded_tags[0, :]  # [B] start tokens
        tgt_emb_buf[0] = E(start_row) + pos_rows[0]  # [B,D] + [D] broadcasts cleanly

        # Track current step
        t = 0
        cache = None

        # SIMPLE TIMING
        if timer:
            timer.start_section('ar_loop')

        # AR LOOP
        for step in range(Tmax):
            if timer:
                timer.start_section('ar_transformer')
            
            # Use incremental embedding buffer - only pass what we've computed so far
            tgt = tgt_emb_buf[:t + 1, :, :]  # [t+1,B,D] - grows each step

            # Transformer decoder with cache
            decoded, cache = decoder(
                tgt, memory=mem_enc, cache=cache, memory_key_padding_mask=None
            )

            # FIX: Ensure contiguous for better memory layout
            last_H = decoded[-1].contiguous()  # [B,D]
            logits = FC(last_H)  # [B,V] - use cached reference
            new_tags = logits.argmax(dim=1)  # [B] Long

            if timer:
                timer.end_section('ar_transformer')
                timer.start_section('ar_other')

            # OPTIMIZATION 2: Compute masks once and reuse
            is_end = (new_tags == self.end_id)
            is_lcel = (new_tags == self.lcel_id) if self.lcel_id is not None else zerosB
            emit = self.emit_lut[new_tags]
            skip_now = self.skip_lut[new_tags]

            # ---- Structure corrections (all on GPU) ----
            if self.xcel_id is not None and self.lcel_id is not None:
                new_tags = torch.where((line_num == 0) & (new_tags == self.xcel_id), self.lcel_id, new_tags)

            # For ucel->lcel correction, check prev_ucel from PREVIOUS step
            if self.ucel_id is not None and self.lcel_id is not None and self.fcel_id is not None:
                new_tags = torch.where(prev_ucel & (new_tags == self.lcel_id), self.fcel_id, new_tags)

            # Update finished and force <end> for already finished sequences
            finished |= is_end
            new_tags = torch.where(finished, end_vec, new_tags)

            # Write to preallocated buffer
            t += 1
            decoded_tags[t, :] = new_tags

            # Update incremental embedding buffer for next step
            if t < Tmax:  # Only if we'll do another step
                # FIX 1: Use clean pos_rows without unsqueeze
                tgt_emb_buf[t] = E(new_tags) + pos_rows[t]  # [B,D] + [D] - use cached reference

            # Update lengths for non-finished sequences
            lengths = torch.where(~finished, lengths + 1, lengths)

            # PHASE 1 OPTIMIZATION: reduce GPU→CPU syncs: check "all finished" every 8 steps
            if (step & 7) == 7:
                if not (~finished).any().item():
                    if timer:
                        timer.end_section('ar_other')
                    break

            # ---- BBox emission decisions (minimize syncs) ----
            # OPTIMIZATION 2: Use precomputed masks
            m_emit_bbox = (~skip_next) & emit & (~finished)
            m_first_lcel = first_lcel & is_lcel & (~finished)

            # Collect indices that need bbox features (minimize CPU syncs)
            append_mask = m_emit_bbox | m_first_lcel
            append_idx = append_mask.nonzero(as_tuple=False).squeeze(1)
            if append_idx.numel() > 0:
                # FIX 2: Check and grow buffer if needed before writing
                tag_H_buf, k_counters = self._maybe_grow_buffer(tag_H_buf, k_counters, append_idx)

                # OPTIMIZATION 2: Write to preallocated buffer instead of list
                rows = last_H.index_select(0, append_idx)  # [K,D]
                slots = k_counters[append_idx].long()  # [K] - current slot for each sample
                flat_idx = append_idx * tag_H_buf.size(1) + slots  # [K] - use current buffer size
                tag_H_buf.view(-1, D_embed).index_copy_(0, flat_idx, rows)  # Vectorized write
                k_counters[append_idx] += 1  # Increment counters

                # PHASE 2 CORRECTED: Use global masks to preserve exact semantics
                # --- SPANS (use global masks to preserve semantics) ---
                # starts: m_first_lcel
                if m_first_lcel.any():
                    first_lcel_idx = m_first_lcel.nonzero(as_tuple=False).squeeze(1)
                    span_starts, span_ends, span_cnt = self._maybe_grow_spans(
                        span_starts, span_ends, span_cnt, first_lcel_idx
                    )
                    slot = span_cnt[first_lcel_idx]
                    span_starts[first_lcel_idx, slot] = bbox_ind[first_lcel_idx]
                    span_cnt[first_lcel_idx] += 1

                # ends: emit only & already inside an open span (i.e., not first_lcel)
                # first_lcel==False means we've seen the first lcel and the span is open
                m_end_global = m_emit_bbox & (~first_lcel) & (~finished)
                if m_end_global.any():
                    end_idx = m_end_global.nonzero(as_tuple=False).squeeze(1)
                    slot = (span_cnt[end_idx] - 1).clamp_min(0)
                    span_ends[end_idx, slot] = bbox_ind[end_idx]

                # Increment bbox indices
                bbox_ind[append_idx] += 1

            # ---- Update flags (all on GPU) ----
            # OPTIMIZATION 2: Simplified flag updates with precomputed masks
            # first_lcel: true unless current is lcel (one kernel instead of where)
            first_lcel = ~is_lcel | finished

            # Update skip_next
            skip_next = skip_now

            # Update prev_ucel for NEXT iteration
            prev_ucel = (new_tags == self.ucel_id) if self.ucel_id is not None else zerosB

            # Update line number
            if self.nl_id is not None:
                line_num += (new_tags == self.nl_id).to(line_num.dtype)

            if timer:
                timer.end_section('ar_other')

        if timer:
            timer.end_section('ar_loop')

        # ---- Materialize outputs (minimal sync points) ----
        # Trim sequences to actual length
        end_id_int = self.end_id.item()
        seqs = []
        for b in range(B):
            seq_len = min(t + 1, lengths[b].item() + 1)  # +1 for start token
            seq = self._trim_sequence(decoded_tags[:seq_len, b], end_id_int)
            seqs.append(seq)

        # ---- Per-table bbox head (already vectorized) ----
        outputs = []

        if timer:
            timer.start_section('bbox_decode')

        # ONE sync here instead of B syncs
        k_list = k_counters.detach().cpu().tolist()  # length B
        cnt_list = span_cnt.detach().cpu().tolist()  # length B

        for b, k_b in enumerate(k_list):
            if self.model._bbox and k_b > 0:
                enc_nchw = enc_out_batch[b:b + 1]  # [1, C, H, W] on GPU
                tag_H_tensor = tag_H_buf[b, :k_b]  # [k_b, D] on GPU
                cls_logits, coords = self.model._bbox_decoder.inference(enc_nchw, tag_H_tensor)
            else:
                cls_logits = torch.empty(0, device=device)
                coords = torch.empty(0, device=device)

            # Use the already-host materialized span count for indexing safety
            nspans_b = cnt_list[b]
            merged_cls, merged_coord = self._merge_spans_tensor(
                cls_logits, coords,
                span_starts[b:b + 1], span_ends[b:b + 1],
                span_cnt[b:b + 1]  # ok to pass the GPU tensors; we just avoided per-step .item()
            )
            outputs.append((seqs[b], merged_cls, merged_coord))

        if timer:
            timer.end_section('bbox_decode')
            
            # Print timing breakdown
            timer.finalize()
            print("\n  === Detailed Decoder Timing ===")
            print(f"    ar_loop:             {timer.get_time('ar_loop'):7.1f} ms")
            print(f"      ar_transformer:    {timer.get_time('ar_transformer'):7.1f} ms")
            print(f"      ar_other:          {timer.get_time('ar_other'):7.1f} ms")
            print(f"    bbox_decode:         {timer.get_time('bbox_decode'):7.1f} ms")

        return outputs

    def _merge_spans_tensor(self, cls_logits: torch.Tensor, coords: torch.Tensor,
                            starts: torch.Tensor, ends: torch.Tensor, cnt: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """Tensorized span merging - no Python dicts"""
        device = coords.device if coords is not None and coords.numel() > 0 else self.device

        if coords.numel() == 0:
            return torch.empty(0, device=device), torch.empty(0, device=device)

        N = coords.size(0)
        keep = torch.ones(N, device=device, dtype=torch.bool)
        merged_cls = []
        merged_coord = []

        # Build a dense map for start->end lookup
        end_map = torch.full((N,), -1, device=device, dtype=torch.long)
        nspans = int(cnt[0].item()) if cnt.numel() > 0 else 0

        if nspans > 0:
            # Extract valid spans from the first sample (b=0)
            valid = (starts[0, :nspans] >= 0) & (ends[0, :nspans] >= 0)
            if valid.any():
                st = starts[0, :nspans][valid]
                ed = ends[0, :nspans][valid]
                # Ensure indices are within bounds
                valid_idx = (st < N) & (ed < N)
                if valid_idx.any():
                    st = st[valid_idx]
                    ed = ed[valid_idx]
                    end_map[st] = ed

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

    def _merge_spans(self, cls_logits: torch.Tensor, coords: torch.Tensor, merge_map: Dict[int, int]) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """Merge bbox spans according to merge_map"""
        device = coords.device if coords is not None and coords.numel() > 0 else self.device
        N = len(coords) if coords is not None else 0

        if N == 0:
            return torch.empty(0, device=device), torch.empty(0, device=device)

        out_cls, out_coord, skip = [], [], set()
        for i in range(N):
            if i in skip:
                continue
            j = merge_map.get(i, -1)
            if 0 <= j < N:
                skip.add(j)
                out_cls.append(cls_logits[i])
                out_coord.append(self.model.mergebboxes(coords[i], coords[j]))
            else:
                out_cls.append(cls_logits[i])
                out_coord.append(coords[i])

        out_cls = torch.stack(out_cls) if out_cls else torch.empty(0, device=device)
        out_coord = torch.stack(out_coord) if out_coord else torch.empty(0, device=device)
        return out_cls, out_coord
