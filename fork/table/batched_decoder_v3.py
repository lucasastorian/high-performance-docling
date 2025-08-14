# fork/table/batched_decoder_v3.py
# Optimized Batched AR decoder with GPT-5 Phase 1 & 2 optimizations implemented
# PHASE 1: Reduced GPU→CPU sync frequency - check "all finished" every 8 steps
# PHASE 2: Reduced nonzero() calls from 3 to 2 per step - CRITICAL: keep ~first_lcel in end condition
#

import torch
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


def _make_alive_pos(B: int, alive_idx: torch.Tensor, device) -> torch.Tensor:
    """Map original batch index -> position in alive batch (or -1)"""
    alive_pos = torch.full((B,), -1, device=device, dtype=torch.long)
    # 0..B_alive-1 assigned into positions 'alive_idx'
    alive_pos.index_copy_(0, alive_idx, torch.arange(alive_idx.numel(), device=device))
    return alive_pos


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
            enc_out_batch: torch.Tensor,
            mem_enc: torch.Tensor,
            max_steps: int,
            timer=None
    ) -> List[Tuple[List[int], torch.Tensor, torch.Tensor]]:
        device = self.device
        tt = self.tt
        B = enc_out_batch.size(0)

        # Cache module references
        E = tt._embedding
        FC = tt._fc
        decoder = tt._decoder
        pos_enc = tt._positional_encoding

        Tmax = min(max_steps, self.model._max_pred_len)

        decoded_tags = torch.full((Tmax + 1, B), self.end_id.item(), dtype=torch.long, device=device)
        decoded_tags[0, :] = self.start_id

        lengths = torch.zeros(B, dtype=torch.int32, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        first_lcel = torch.ones(B, dtype=torch.bool, device=device)
        skip_next = torch.ones(B, dtype=torch.bool, device=device)
        prev_ucel = torch.zeros(B, dtype=torch.bool, device=device)
        line_num = torch.zeros(B, dtype=torch.long, device=device)
        bbox_ind = torch.zeros(B, dtype=torch.long, device=device)

        D_embed = E.embedding_dim
        Kmax = max(1, Tmax // 2)
        tag_H_buf = torch.empty(B, Kmax, D_embed, device=device)
        k_counters = torch.zeros(B, dtype=torch.int32, device=device)

        Kspan = max(1, Tmax // 2)
        span_starts = torch.full((B, Kspan), -1, device=device, dtype=torch.long)
        span_ends = torch.full((B, Kspan), -1, device=device, dtype=torch.long)
        span_cnt = torch.zeros(B, device=device, dtype=torch.long)

        tgt_emb_buf = torch.empty(Tmax + 1, B, D_embed, device=device)
        pos_rows = pos_enc.pe[:Tmax + 1, 0]

        end_vec = torch.full((B,), self.end_id.item(), device=device, dtype=torch.long)
        zerosB = torch.zeros(B, dtype=torch.bool, device=device)

        # init first embedding row
        tgt_emb_buf[0] = E(decoded_tags[0, :]) + pos_rows[0]

        t = 0
        cache = None

        for step in range(Tmax):
            # --- NEW: Compact batch if some sequences finished ---
            alive_idx = (~finished).nonzero(as_tuple=False).squeeze(1)
            if alive_idx.numel() == 0:
                break
            if alive_idx.numel() != B:
                # shrink all per-seq buffers to only alive rows
                decoded_tags = decoded_tags[:, alive_idx]
                lengths = lengths.index_select(0, alive_idx)
                finished = finished.index_select(0, alive_idx)
                first_lcel = first_lcel.index_select(0, alive_idx)
                skip_next = skip_next.index_select(0, alive_idx)
                prev_ucel = prev_ucel.index_select(0, alive_idx)
                line_num = line_num.index_select(0, alive_idx)
                bbox_ind = bbox_ind.index_select(0, alive_idx)
                tag_H_buf = tag_H_buf.index_select(0, alive_idx)
                k_counters = k_counters.index_select(0, alive_idx)
                span_starts = span_starts.index_select(0, alive_idx)
                span_ends = span_ends.index_select(0, alive_idx)
                span_cnt = span_cnt.index_select(0, alive_idx)
                tgt_emb_buf = tgt_emb_buf[:, alive_idx]
                enc_out_batch = enc_out_batch.index_select(0, alive_idx)
                mem_enc = mem_enc.index_select(1, alive_idx)
                if cache is not None:
                    cache = cache.index_select(2, alive_idx)
                B = alive_idx.size(0)

            # decode alive batch
            tgt_alive = tgt_emb_buf[:t + 1]
            if timer:
                timer.start_section('ar_transformer')
            decoded_alive, cache_alive_new = decoder(tgt_alive, memory=mem_enc, cache=cache,
                                                     memory_key_padding_mask=None)
            last_H_alive = decoded_alive[-1].contiguous()
            logits_alive = FC(last_H_alive)
            new_tags_alive = logits_alive.argmax(dim=1)
            if timer:
                timer.end_section('ar_transformer')
                timer.start_section('ar_other')

            # structure corrections
            if self.xcel_id is not None and self.lcel_id is not None:
                new_tags_alive = torch.where((line_num == 0) & (new_tags_alive == self.xcel_id), self.lcel_id,
                                             new_tags_alive)
            if self.ucel_id is not None and self.lcel_id is not None and self.fcel_id is not None:
                new_tags_alive = torch.where(prev_ucel & (new_tags_alive == self.lcel_id), self.fcel_id, new_tags_alive)

            # finished mask
            is_end = (new_tags_alive == self.end_id)
            finished |= is_end
            new_tags_alive = torch.where(finished, end_vec[:B], new_tags_alive)

            # write new tags
            t += 1
            decoded_tags[t, :] = new_tags_alive

            # update embedding buf
            if t < Tmax:
                tgt_emb_buf[t] = E(new_tags_alive) + pos_rows[t]

            # update lengths
            lengths = torch.where(~finished, lengths + 1, lengths)
            if (step & 7) == 7 and not (~finished).any().item():
                if timer:
                    timer.end_section('ar_other')
                break

            # bbox emission
            emit = self.emit_lut[new_tags_alive]
            skip_now = self.skip_lut[new_tags_alive]
            is_lcel = (new_tags_alive == self.lcel_id) if self.lcel_id is not None else zerosB[:B]

            m_emit_bbox = (~skip_next) & emit & (~finished)
            m_first_lcel = first_lcel & is_lcel & (~finished)
            append_mask = m_emit_bbox | m_first_lcel
            if append_mask.any():
                append_idx = append_mask.nonzero(as_tuple=False).squeeze(1)
                rows = last_H_alive.index_select(0, append_idx)
                tag_H_buf, k_counters = self._maybe_grow_buffer(tag_H_buf, k_counters, append_idx)
                slots = k_counters[append_idx].long()
                flat = append_idx * tag_H_buf.size(1) + slots
                tag_H_buf.view(-1, D_embed).index_copy_(0, flat, rows)
                k_counters[append_idx] += 1
                if m_first_lcel.any():
                    first_lcel_idx = m_first_lcel.nonzero(as_tuple=False).squeeze(1)
                    span_starts, span_ends, span_cnt = self._maybe_grow_spans(span_starts, span_ends, span_cnt,
                                                                              first_lcel_idx)
                    slot = span_cnt[first_lcel_idx]
                    span_starts[first_lcel_idx, slot] = bbox_ind[first_lcel_idx]
                    span_cnt[first_lcel_idx] += 1
                m_end_global = m_emit_bbox & (~first_lcel) & (~finished)
                if m_end_global.any():
                    end_idx = m_end_global.nonzero(as_tuple=False).squeeze(1)
                    slot = (span_cnt[end_idx] - 1).clamp_min(0)
                    span_ends[end_idx, slot] = bbox_ind[end_idx]
                bbox_ind[append_idx] += 1

            # update flags
            first_lcel = ~is_lcel | finished
            skip_next = skip_now
            prev_ucel = (new_tags_alive == self.ucel_id) if self.ucel_id is not None else zerosB[:B]
            if self.nl_id is not None:
                line_num += (new_tags_alive == self.nl_id).to(line_num.dtype)
            if timer:
                timer.end_section('ar_other')

            cache = cache_alive_new

        # build outputs
        seqs = []
        for b in range(B):
            seq_len = min(t + 1, lengths[b].item() + 1)
            seqs.append(self._trim_sequence(decoded_tags[:seq_len, b], self.end_id.item()))

        outputs = []
        k_list = k_counters.detach().cpu().tolist()
        cnt_list = span_cnt.detach().cpu().tolist()
        for b, k_b in enumerate(k_list):
            if self.model._bbox and k_b > 0:
                enc_nchw = enc_out_batch[b:b + 1]
                tag_H_tensor = tag_H_buf[b, :k_b]
                cls_logits, coords = self.model._bbox_decoder.inference(enc_nchw, tag_H_tensor)
            else:
                cls_logits = torch.empty(0, device=device)
                coords = torch.empty(0, device=device)
            merged_cls, merged_coord = self._merge_spans_tensor(
                cls_logits, coords,
                span_starts[b:b + 1], span_ends[b:b + 1],
                span_cnt[b:b + 1]
            )
            outputs.append((seqs[b], merged_cls, merged_coord))
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

    def _compact_batch(self, state: BatchState, alive_idx: torch.Tensor):
        """
        In-place compaction of all per-sequence state tensors after some sequences are finished.
        Keeps only sequences indexed by alive_idx.
        """
        # Basic batch vars
        state.decoded_tags = state.decoded_tags.index_select(1, alive_idx)
        state.lengths = state.lengths.index_select(0, alive_idx)
        state.finished = state.finished.index_select(0, alive_idx)
        state.first_lcel = state.first_lcel.index_select(0, alive_idx)
        state.skip_next = state.skip_next.index_select(0, alive_idx)

        # Cache: [num_layers, B, ...]
        if state.cache is not None:
            state.cache = state.cache.index_select(1, alive_idx)

        # BBox head ragged buffers
        # Keep only entries where sample_id is still alive
        old_to_new = {int(old.item()): i for i, old in enumerate(alive_idx)}
        new_H_flat = []
        new_sample_ids = []
        for hid, sid in zip(state.tag_H_flat, state.tag_H_sample_ids):
            if sid in old_to_new:
                new_H_flat.append(hid)
                new_sample_ids.append(old_to_new[sid])
        state.tag_H_flat = new_H_flat
        state.tag_H_sample_ids = new_sample_ids

        return state
