# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
# Batched decoder implementation for TableModel04_rs
# Processes multiple tables in parallel instead of sequentially

import torch
from dataclasses import dataclass
from typing import List, Optional, Dict

from docling_ibm_models.tableformer.utils.app_profiler import AggProfiler


@dataclass
class BatchState:
    """State for batched decoding - tensors on GPU; Python lists only for ragged buffers."""
    decoded_tags: torch.Tensor          # [T, B] Long - generated tokens
    finished: torch.Tensor              # [B] Bool - sequences done
    cache: Optional[torch.Tensor]       # decoder cache
    # B lists of step features for bbox head (GPU tensors), kept as per-seq ragged lists
    tag_H_buf: List[List[torch.Tensor]]

    # Decoding/book-keeping flags (all on device)
    first_lcel: torch.Tensor            # [B] Bool - tracking horizontal spans
    skip_next_tag: torch.Tensor         # [B] Bool - skip bbox for next token
    prev_tag_ucel: torch.Tensor         # [B] Bool - was previous tag ucel
    line_num: torch.Tensor              # [B] Long - current line number

    # Bbox indexing (device)
    bbox_ind: torch.Tensor              # [B] Long - running bbox index
    cur_bbox_ind: torch.Tensor          # [B] Long - current span start index or -1

    # Span bookkeeping (Python ragged lists of ints; filled from batched tensor snapshots)
    span_starts: List[List[int]]        # per sequence: start idx list
    span_ends: List[List[int]]          # per sequence: end idx list (may contain -1 placeholders)


class BatchedTableDecoder:
    """Batched decoder for table structure prediction"""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self._prof = model._prof

    @torch.inference_mode()
    def predict_batched(self, enc_out_batch: torch.Tensor, max_steps: int) -> List:
        """
        Batched prediction for multiple tables at once

        Args:
            enc_out_batch: [B, H, W, C] encoded features for B tables
            max_steps: maximum decoding steps

        Returns:
            List of (seq, outputs_class, outputs_coord) per table
        """
        device = self.device
        B = enc_out_batch.size(0)

        # If batch size is 1, fall back to original implementation for safety
        if B == 1:
            return self._predict_single(enc_out_batch, max_steps)

        tt = self.model._tag_transformer
        word_map = self.model._init_data["word_map"]["word_map_tag"]

        # ---- Precompute frequently used ids (ints) ----
        start_id = word_map["<start>"]
        end_id   = word_map["<end>"]
        xcel_id  = word_map.get("xcel", -999)
        lcel_id  = word_map.get("lcel", -999)
        fcel_id  = word_map.get("fcel", -999)
        nl_id    = word_map.get("nl",   -999)
        ucel_id  = word_map.get("ucel", -999)

        bbox_emit_list = [word_map.get(k, -999) for k in ["fcel", "ecel", "ched", "rhed", "srow", "nl", "ucel"]]
        bbox_emit_ids = torch.tensor([t for t in bbox_emit_list if t != -999], device=device, dtype=torch.long)

        # ---- Encoder preprocessing ----
        AggProfiler().begin("batched_input_filter", self._prof)
        # [B, H, W, C] -> [B, h, w, C]
        encoder_out = tt._input_filter(enc_out_batch.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        AggProfiler().end("batched_input_filter", self._prof)

        C = encoder_out.size(-1)
        S = encoder_out.size(1) * encoder_out.size(2)

        # [B, S, C] -> [S, B, C]
        memory = encoder_out.reshape(B, S, C).permute(1, 0, 2).contiguous()

        AggProfiler().begin("batched_encoder", self._prof)
        mem_enc = tt._encoder(memory, mask=None)  # [S, B, C]
        AggProfiler().end("batched_encoder", self._prof)

        # ---- Init state ----
        decoded_tags = torch.full((1, B), start_id, dtype=torch.long, device=device)

        state = BatchState(
            decoded_tags=decoded_tags,
            finished=torch.zeros(B, dtype=torch.bool, device=device),
            cache=None,
            tag_H_buf=[[] for _ in range(B)],
            first_lcel=torch.ones(B, dtype=torch.bool, device=device),
            skip_next_tag=torch.ones(B, dtype=torch.bool, device=device),  # starts True
            prev_tag_ucel=torch.zeros(B, dtype=torch.bool, device=device),
            line_num=torch.zeros(B, dtype=torch.long, device=device),
            bbox_ind=torch.zeros(B, dtype=torch.long, device=device),
            cur_bbox_ind=torch.full((B,), -1, dtype=torch.long, device=device),
            span_starts=[[] for _ in range(B)],
            span_ends=[[] for _ in range(B)],
        )

        # Helper for membership (Torch 2.1+ has torch.isin)
        def isin_tags(tags: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
            return torch.isin(tags, ids) if hasattr(torch, "isin") else (tags.unsqueeze(-1) == ids).any(-1)

        # ---- Autoregressive loop ----
        AggProfiler().begin("batched_ar_loop", self._prof)
        for _ in range(max_steps):
            # [T, B, D]
            tgt_emb = tt._positional_encoding(tt._embedding(state.decoded_tags))

            AggProfiler().begin("batched_decoder_step", self._prof)
            decoded, state.cache = tt._decoder(
                tgt_emb,
                memory=mem_enc,
                cache=state.cache,
                memory_key_padding_mask=None,
            )
            AggProfiler().end("batched_decoder_step", self._prof)

            # logits and next tokens
            logits = tt._fc(decoded[-1, :, :])   # [B, vocab]
            new_tags = logits.argmax(dim=1)      # [B] Long on device
            last_H = decoded[-1, :, :]           # [B, D]

            # --- Structure corrections (pure GPU) ---
            if xcel_id != -999 and lcel_id != -999:
                mask_first_line = (state.line_num == 0) & (new_tags == xcel_id)
                new_tags = torch.where(mask_first_line, new_tags.new_full((B,), lcel_id), new_tags)

            if lcel_id != -999 and fcel_id != -999 and ucel_id != -999:
                mask_ucel_lcel = state.prev_tag_ucel & (new_tags == lcel_id)
                new_tags = torch.where(mask_ucel_lcel, new_tags.new_full((B,), fcel_id), new_tags)

            # Force <end> for already finished sequences
            new_tags = torch.where(state.finished, new_tags.new_full((B,), end_id), new_tags)

            # --- Masks we reuse ---
            m_emit_bbox  = (~state.skip_next_tag) & isin_tags(new_tags, bbox_emit_ids)
            m_is_lcel    = (new_tags == lcel_id) if lcel_id != -999 else torch.zeros_like(new_tags, dtype=torch.bool)
            m_first_lcel = state.first_lcel & m_is_lcel
            m_not_lcel   = ~m_is_lcel
            m_end_span   = m_emit_bbox & (~state.first_lcel)
            m_nl         = (new_tags == nl_id) if nl_id != -999 else torch.zeros_like(new_tags, dtype=torch.bool)

            # --- Append features for bbox head for the union of emit|first_lcel ---
            append_idx = torch.nonzero(m_emit_bbox | m_first_lcel, as_tuple=False).squeeze(1)
            if append_idx.numel() > 0:
                # slice last_H rows and then scatter into ragged lists
                # We keep per-i small Python loops, but indices are batched above; no .item() on the hot path.
                rows = last_H.index_select(0, append_idx)  # [K, D]
                idx_list = append_idx.tolist()
                for i, row in zip(idx_list, rows):
                    state.tag_H_buf[i].append(row.unsqueeze(0))

            # --- Span starts: store start index = current bbox_ind BEFORE increment ---
            if m_first_lcel.any():
                start_idx = torch.nonzero(m_first_lcel, as_tuple=False).squeeze(1)
                start_vals = state.bbox_ind.index_select(0, start_idx).tolist()  # batched snapshot -> Python once
                for i, start_v in zip(start_idx.tolist(), start_vals):
                    state.cur_bbox_ind[i] = start_v  # stays on device
                    state.span_starts[i].append(start_v)
                    state.span_ends[i].append(-1)    # placeholder; filled on end

            # --- Span ends: when emitting bbox and we are inside a span, record end = current bbox_ind BEFORE increment ---
            if m_end_span.any():
                end_idx = torch.nonzero(m_end_span, as_tuple=False).squeeze(1)
                end_vals = state.bbox_ind.index_select(0, end_idx).tolist()
                for i, end_v in zip(end_idx.tolist(), end_vals):
                    # fill the last open span end if present
                    if state.span_ends[i] and state.span_ends[i][-1] == -1:
                        state.span_ends[i][-1] = end_v

            # --- Advance bbox_ind for any step that emitted or started a span ---
            if append_idx.numel() > 0:
                state.bbox_ind.index_add_(0, append_idx, torch.ones_like(append_idx, dtype=state.bbox_ind.dtype))

            # --- Update flags ---
            # first_lcel resets to True when not lcel; becomes False when first_lcel occurs this step
            state.first_lcel = torch.where(
                m_not_lcel, torch.ones_like(state.first_lcel),
                torch.where(m_first_lcel, torch.zeros_like(state.first_lcel), state.first_lcel)
            )

            # skip_next_tag becomes True for nl/ucel/xcel, else False
            m_skip_now = (new_tags == nl_id) | (new_tags == ucel_id) | (new_tags == xcel_id)
            state.skip_next_tag = m_skip_now

            # prev_tag_ucel for next step
            state.prev_tag_ucel = (new_tags == ucel_id)

            # increment line number on nl where not finished
            state.line_num = state.line_num + m_nl.to(state.line_num.dtype)

            # --- Append tokens & termination ---
            state.decoded_tags = torch.cat([state.decoded_tags, new_tags.view(1, B)], dim=0)
            state.finished |= (new_tags == end_id)
            if state.finished.all():
                break
        AggProfiler().end("batched_ar_loop", self._prof)

        # ---- Materialize outputs ----
        seqs = [state.decoded_tags[:, b].tolist() for b in range(B)]

        # Build bboxes_to_merge dicts *once* per sequence (no inner-loop Python writes)
        bboxes_to_merge_all: List[Dict[int, int]] = []
        for b in range(B):
            d: Dict[int, int] = {}
            starts = state.span_starts[b]
            ends   = state.span_ends[b]
            # they are parallel; length equal by construction
            for s, e in zip(starts, ends):
                d[s] = e  # e may be -1 (open span)
            bboxes_to_merge_all.append(d)

        outputs = []
        AggProfiler().begin("batched_bbox_decode", self._prof)
        for b in range(B):
            if self.model._bbox and len(state.tag_H_buf[b]) > 0:
                outputs_class, outputs_coord = self.model._bbox_decoder.inference(
                    enc_out_batch[b:b+1],
                    state.tag_H_buf[b]
                )
            else:
                outputs_class = torch.empty(0, device=device)
                outputs_coord = torch.empty(0, device=device)

            outputs_class, outputs_coord = self._merge_span_bboxes(
                outputs_class, outputs_coord, bboxes_to_merge_all[b]
            )
            outputs.append((seqs[b], outputs_class, outputs_coord))
        AggProfiler().end("batched_bbox_decode", self._prof)

        return outputs

    def _predict_single(self, enc_out_batch: torch.Tensor, max_steps: int) -> List:
        """Fallback to original single-item prediction"""
        img = torch.zeros(1, 3, 448, 448, device=self.device)  # dummy, not used
        seq, outputs_class, outputs_coord = self.model._predict(
            img, max_steps, 1, False,
            precomputed_enc=enc_out_batch
        )
        return [(seq, outputs_class, outputs_coord)]

    def _merge_span_bboxes(self, outputs_class, outputs_coord, bboxes_to_merge):
        """Merge first and last bbox for each span"""
        device = self.device

        outputs_class = outputs_class.to(device) if outputs_class is not None else torch.empty(0, device=device)
        outputs_coord = outputs_coord.to(device) if outputs_coord is not None else torch.empty(0, device=device)

        outputs_class1 = []
        outputs_coord1 = []
        boxes_to_skip = set()

        for box_ind in range(len(outputs_coord)):
            if box_ind in boxes_to_skip:
                continue

            box1 = outputs_coord[box_ind]
            cls1 = outputs_class[box_ind]

            if box_ind in bboxes_to_merge:
                box2_ind = bboxes_to_merge[box_ind]
                if 0 <= box2_ind < len(outputs_coord):
                    boxes_to_skip.add(box2_ind)
                    box2 = outputs_coord[box2_ind]
                    boxm = self.model.mergebboxes(box1, box2)
                    outputs_coord1.append(boxm)
                    outputs_class1.append(cls1)
                else:
                    # open span (end == -1) or out of range: keep box1
                    outputs_coord1.append(box1)
                    outputs_class1.append(cls1)
            else:
                outputs_coord1.append(box1)
                outputs_class1.append(cls1)

        outputs_coord1 = torch.stack(outputs_coord1) if len(outputs_coord1) else torch.empty(0, device=device)
        outputs_class1 = torch.stack(outputs_class1) if len(outputs_class1) else torch.empty(0, device=device)
        return outputs_class1, outputs_coord1
