# optimized/table/batched_decoder.py
# Batched AR decoder for TableModel04_rs (no new learnable params)

import torch
from dataclasses import dataclass
from typing import List, Dict, Optional
from docling_ibm_models.tableformer.utils.app_profiler import AggProfiler


@dataclass
class BatchState:
    decoded_tags: torch.Tensor          # [T, B] Long
    finished: torch.Tensor              # [B] Bool
    cache: Optional[torch.Tensor]       # transformer cache (layers x T x B x D)

    # Per-seq ragged buffers for bbox head (GPU tensors stored in Python lists)
    tag_H_buf: List[List[torch.Tensor]]

    # Decoding flags (device tensors)
    first_lcel: torch.Tensor            # [B] Bool
    skip_next: torch.Tensor             # [B] Bool
    prev_ucel: torch.Tensor             # [B] Bool
    line_num: torch.Tensor              # [B] Long

    # Bbox indexing (device counters) + span bookkeeping (Python lists)
    bbox_ind: torch.Tensor              # [B] Long (running index)
    span_starts: List[List[int]]        # per-seq start indices
    span_ends:   List[List[int]]        # per-seq end indices (may hold -1)


class BatchedTableDecoder:
    def __init__(self, model, device: str):
        self.model = model
        self.device = device
        self._prof = model._prof

        # Prebind a few references
        self.tt = model._tag_transformer
        wm = model._init_data["word_map"]["word_map_tag"]
        self.start_id = wm["<start>"]
        self.end_id   = wm["<end>"]

        # Optional ids
        self.xcel_id = wm.get("xcel", -1)
        self.lcel_id = wm.get("lcel", -1)
        self.fcel_id = wm.get("fcel", -1)
        self.ucel_id = wm.get("ucel", -1)
        self.nl_id   = wm.get("nl",   -1)

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

    @torch.inference_mode()
    def predict_batched(
        self,
        enc_out_batch: torch.Tensor,   # [B,H,W,C] for bbox head
        mem_enc: torch.Tensor,         # [S,B,D] encoder memory for tag transformer
        max_steps: int
    ):
        device = self.device
        tt = self.tt
        B = enc_out_batch.size(0)

        # ---- Init state
        decoded_tags = torch.full((1, B), self.start_id, dtype=torch.long, device=device)
        state = BatchState(
            decoded_tags=decoded_tags,
            finished=torch.zeros(B, dtype=torch.bool, device=device),
            cache=None,
            tag_H_buf=[[] for _ in range(B)],
            first_lcel=torch.ones(B, dtype=torch.bool, device=device),
            skip_next=torch.ones(B, dtype=torch.bool, device=device),
            prev_ucel=torch.zeros(B, dtype=torch.bool, device=device),
            line_num=torch.zeros(B, dtype=torch.long, device=device),
            bbox_ind=torch.zeros(B, dtype=torch.long, device=device),
            span_starts=[[] for _ in range(B)],
            span_ends=[[] for _ in range(B)],
        )

        # Helper (2.1+ has torch.isin; keep fallbacks)
        def isin(vals: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
            if ids.numel() == 0:
                return torch.zeros_like(vals, dtype=torch.bool)
            return torch.isin(vals, ids) if hasattr(torch, "isin") else (vals[..., None] == ids).any(-1)

        prof = AggProfiler()
        prof.begin("batched_ar_loop", self._prof)
        for _ in range(max_steps):
            # [T,B,D]
            tgt = tt._positional_encoding(tt._embedding(state.decoded_tags))

            # Transformer decoder (last-token optimized layer, with cache)
            decoded, state.cache = tt._decoder(
                tgt, memory=mem_enc, cache=state.cache, memory_key_padding_mask=None
            )

            last_H = decoded[-1, :, :]       # [B,D]
            logits = tt._fc(last_H)           # [B,V]
            new_tags = logits.argmax(dim=1)   # [B] Long

            # Structure corrections on device
            if self.xcel_id != -1 and self.lcel_id != -1:
                mask_first_line = (state.line_num == 0) & (new_tags == self.xcel_id)
                new_tags = torch.where(mask_first_line, new_tags.new_full((B,), self.lcel_id), new_tags)

            if self.ucel_id != -1 and self.lcel_id != -1 and self.fcel_id != -1:
                mask_ucel_lcel = state.prev_ucel & (new_tags == self.lcel_id)
                new_tags = torch.where(mask_ucel_lcel, new_tags.new_full((B,), self.fcel_id), new_tags)

            # Force <end> for finished seqs
            new_tags = torch.where(state.finished, new_tags.new_full((B,), self.end_id), new_tags)

            # Append token
            state.decoded_tags = torch.cat([state.decoded_tags, new_tags.view(1, B)], dim=0)

            # Termination check
            state.finished |= (new_tags == self.end_id)
            if state.finished.all():
                break

            # BBox emission masks
            m_emit_bbox  = (~state.skip_next) & isin(new_tags, self.emit_ids)
            m_is_lcel    = (new_tags == self.lcel_id) if self.lcel_id != -1 else torch.zeros_like(new_tags, dtype=torch.bool)
            m_first_lcel = state.first_lcel & m_is_lcel
            m_not_lcel   = ~m_is_lcel
            m_end_span   = m_emit_bbox & (~state.first_lcel)
            m_nl         = (new_tags == self.nl_id) if self.nl_id != -1 else torch.zeros_like(new_tags, dtype=torch.bool)

            # Append features for bbox head (union of emit | first_lcel)
            append_idx = torch.nonzero(m_emit_bbox | m_first_lcel, as_tuple=False).squeeze(1)
            if append_idx.numel():
                rows = last_H.index_select(0, append_idx)  # [K,D]
                for i, row in zip(append_idx.tolist(), rows):  # tiny, unavoidable Python loop over K≪B
                    state.tag_H_buf[i].append(row.unsqueeze(0))  # keep [1,D]

            # Span starts: record start index = current bbox_ind BEFORE increment
            if m_first_lcel.any():
                idx = torch.nonzero(m_first_lcel, as_tuple=False).squeeze(1)
                starts = state.bbox_ind.index_select(0, idx).tolist()
                for i, s in zip(idx.tolist(), starts):
                    state.span_starts[i].append(s)
                    state.span_ends[i].append(-1)

            # Span ends: when we emit and are inside a span, set end = current bbox_ind BEFORE increment
            if m_end_span.any():
                idx = torch.nonzero(m_end_span, as_tuple=False).squeeze(1)
                ends = state.bbox_ind.index_select(0, idx).tolist()
                for i, e in zip(idx.tolist(), ends):
                    if state.span_ends[i] and state.span_ends[i][-1] == -1:
                        state.span_ends[i][-1] = e

            # Advance bbox_ind for any step that emitted or started a span
            if append_idx.numel():
                state.bbox_ind.index_add_(0, append_idx, torch.ones_like(append_idx, dtype=state.bbox_ind.dtype))

            # Update flags
            state.first_lcel = torch.where(
                m_not_lcel, torch.ones_like(state.first_lcel),
                torch.where(m_first_lcel, torch.zeros_like(state.first_lcel), state.first_lcel)
            )
            state.skip_next = isin(new_tags, self.skip_ids)
            state.prev_ucel = (new_tags == self.ucel_id) if self.ucel_id != -1 else torch.zeros_like(new_tags, dtype=torch.bool)
            state.line_num  = state.line_num + m_nl.to(state.line_num.dtype)

        prof.end("batched_ar_loop", self._prof)

        # Materialize outputs
        seqs = [state.decoded_tags[:, b].tolist() for b in range(B)]

        # Build span merge dicts
        merges: List[Dict[int, int]] = []
        for b in range(B):
            d = {s: e for s, e in zip(state.span_starts[b], state.span_ends[b])}
            merges.append(d)

        # Per-table bbox head (already vectorized inside one table)
        outputs = []
        prof.begin("batched_bbox_decode", self._prof)
        for b in range(B):
            tag_H_buf_b = state.tag_H_buf[b]
            if self.model._bbox and len(tag_H_buf_b):
                cls_logits, coords = self.model._bbox_decoder.inference(
                    enc_out_batch[b:b+1], tag_H_buf_b
                )
            else:
                cls_logits = torch.empty(0, device=device)
                coords     = torch.empty(0, device=device)

            # Merge spans (treat -1 as open span -> keep first box)
            merged_cls, merged_coord = self._merge_spans(cls_logits, coords, merges[b])
            outputs.append((seqs[b], merged_cls, merged_coord))
        prof.end("batched_bbox_decode", self._prof)

        return outputs

    def _merge_spans(self, cls_logits, coords, merge_map: Dict[int, int]):
        device = coords.device if coords is not None else self.device
        N = len(coords) if coords is not None else 0
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
