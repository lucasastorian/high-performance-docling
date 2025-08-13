# fork/table/batched_decoder_v2.py
# Optimized Batched AR decoder with GPT-5 feedback implemented

import torch
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from docling_ibm_models.tableformer.utils.app_profiler import AggProfiler


@dataclass
class BatchState:
    """State for batched AR decoding with preallocated buffers"""
    decoded_tags: torch.Tensor          # [Tmax+1, B] Long - preallocated
    lengths: torch.Tensor               # [B] Int - actual length per seq (excluding start)
    finished: torch.Tensor              # [B] Bool
    cache: Optional[torch.Tensor]       # transformer cache
    
    # Per-seq ragged buffers for bbox head (keep as flat list + indices for less sync)
    tag_H_flat: List[torch.Tensor]      # flat list of all hidden states
    tag_H_sample_ids: List[int]         # which sample each belongs to
    
    # Decoding flags (device tensors)
    first_lcel: torch.Tensor            # [B] Bool
    skip_next: torch.Tensor             # [B] Bool
    prev_ucel: torch.Tensor             # [B] Bool
    line_num: torch.Tensor              # [B] Long
    
    # Bbox indexing (device counters)
    bbox_ind: torch.Tensor              # [B] Long (running index)
    
    # Span bookkeeping - track one open span per sample
    open_span_start: List[int]          # [-1 if no open span, else start index] per sample


class BatchedTableDecoderV2:
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
    
    def _trim_sequence(self, seq_tensor: torch.Tensor, end_id: int) -> List[int]:
        """Trim sequence to first <end> token"""
        seq_list = seq_tensor.tolist()
        try:
            idx = seq_list.index(end_id)
            return seq_list[:idx + 1]
        except ValueError:
            return seq_list
    
    @torch.inference_mode()
    def predict_batched(
        self,
        enc_out_batch: torch.Tensor,   # [B,H,W,C] for bbox head
        mem_enc: torch.Tensor,         # [S,B,D] encoder memory
        max_steps: int
    ) -> List[Tuple[List[int], torch.Tensor, torch.Tensor]]:
        device = self.device
        tt = self.tt
        B = enc_out_batch.size(0)
        
        # Clamp to model's max
        Tmax = min(max_steps, self.model._max_pred_len)
        
        # ---- Preallocate all buffers ----
        decoded_tags = torch.full((Tmax + 1, B), self.end_id.item(), dtype=torch.long, device=device)
        decoded_tags[0, :] = self.start_id
        
        # Initialize span maps correctly - one dict per sample
        span_maps: List[Dict[int, int]] = [dict() for _ in range(B)]
        
        state = BatchState(
            decoded_tags=decoded_tags,
            lengths=torch.zeros(B, dtype=torch.int32, device=device),
            finished=torch.zeros(B, dtype=torch.bool, device=device),
            cache=None,
            tag_H_flat=[],
            tag_H_sample_ids=[],
            first_lcel=torch.ones(B, dtype=torch.bool, device=device),
            skip_next=torch.ones(B, dtype=torch.bool, device=device),
            prev_ucel=torch.zeros(B, dtype=torch.bool, device=device),
            line_num=torch.zeros(B, dtype=torch.long, device=device),
            bbox_ind=torch.zeros(B, dtype=torch.long, device=device),
            open_span_start=[-1] * B,  # Track one open span per sample
        )
        
        # Helper for isin with guard
        def safe_isin(vals: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
            if ids.numel() == 0:
                return torch.zeros_like(vals, dtype=torch.bool)
            return torch.isin(vals, ids)
        
        prof = AggProfiler()
        prof.begin("batched_ar_loop_v2", self._prof)
        
        # ---- Incremental embedding buffer optimization ----
        D = tt._embedding.embedding_dim
        tgt_emb_buf = torch.empty(Tmax + 1, B, D, device=device)
        pe = tt._positional_encoding.pe  # [max_len, D] positional encoding buffer
        
        # Initialize first step with start tokens
        start_row = decoded_tags[0, :]  # [B] start tokens
        tgt_emb_buf[0] = tt._embedding(start_row) + pe[0].unsqueeze(0)  # [B,D]
        
        # Track current step
        t = 0
        
        for step in range(Tmax):
            # Use incremental embedding buffer - only pass what we've computed so far
            tgt = tgt_emb_buf[:t+1, :, :]  # [t+1,B,D] - grows each step
            
            # Transformer decoder with cache
            prof.begin("decoder_step", self._prof)
            decoded, state.cache = tt._decoder(
                tgt, memory=mem_enc, cache=state.cache, memory_key_padding_mask=None
            )
            prof.end("decoder_step", self._prof)
            
            last_H = decoded[-1, :, :]       # [B,D]
            logits = tt._fc(last_H)          # [B,V]
            new_tags = logits.argmax(dim=1)  # [B] Long
            
            # ---- Structure corrections (all on GPU) ----
            if self.xcel_id is not None and self.lcel_id is not None:
                mask_first_line = (state.line_num == 0) & (new_tags == self.xcel_id)
                new_tags = torch.where(mask_first_line, self.lcel_id.expand(B), new_tags)
            
            # For ucel->lcel correction, check prev_ucel from PREVIOUS step
            if self.ucel_id is not None and self.lcel_id is not None and self.fcel_id is not None:
                mask_ucel_lcel = state.prev_ucel & (new_tags == self.lcel_id)
                new_tags = torch.where(mask_ucel_lcel, self.fcel_id.expand(B), new_tags)
            
            # Force <end> for already finished sequences
            new_tags = torch.where(state.finished, self.end_id.expand(B), new_tags)
            
            # Write to preallocated buffer
            t += 1
            decoded_tags[t, :] = new_tags
            
            # Update incremental embedding buffer for next step
            if t < Tmax:  # Only if we'll do another step
                tgt_emb_buf[t] = tt._embedding(new_tags) + pe[t].unsqueeze(0)  # [B,D]
            
            # Update lengths for non-finished sequences
            state.lengths = torch.where(~state.finished, state.lengths + 1, state.lengths)
            
            # Update finished status
            newly_finished = (new_tags == self.end_id)
            state.finished |= newly_finished
            
            # Early exit if all finished
            if state.finished.all():
                break
            
            # ---- BBox emission decisions (minimize syncs) ----
            m_emit_bbox = (~state.skip_next) & safe_isin(new_tags, self.emit_ids) & (~state.finished)
            m_is_lcel = (new_tags == self.lcel_id) if self.lcel_id is not None else torch.zeros(B, dtype=torch.bool, device=device)
            m_first_lcel = state.first_lcel & m_is_lcel & (~state.finished)
            
            # Collect indices that need bbox features (single sync point)
            append_mask = m_emit_bbox | m_first_lcel
            if append_mask.any():
                append_idx = append_mask.nonzero(as_tuple=False).squeeze(1)
                if append_idx.numel() > 0:
                    # Batch extract hidden states
                    rows = last_H.index_select(0, append_idx)  # [K,D]
                    # Store flat with sample IDs (avoid per-sample lists)
                    for idx, row in zip(append_idx.tolist(), rows):
                        state.tag_H_flat.append(row)
                        state.tag_H_sample_ids.append(idx)
                    
                    # Handle span tracking inline (much simpler!)
                    first_lcel_idx = (m_first_lcel & append_mask).nonzero(as_tuple=False).squeeze(1)
                    if first_lcel_idx.numel() > 0:
                        # Record span starts - use bbox_ind BEFORE increment
                        start_indices = state.bbox_ind[first_lcel_idx].tolist()
                        for sid, start in zip(first_lcel_idx.tolist(), start_indices):
                            state.open_span_start[sid] = start
                    
                    # Handle span ends
                    end_span_mask = m_emit_bbox & (~state.first_lcel) & append_mask
                    if end_span_mask.any():
                        end_idx = end_span_mask.nonzero(as_tuple=False).squeeze(1)
                        end_indices = state.bbox_ind[end_idx].tolist()
                        for sid, end in zip(end_idx.tolist(), end_indices):
                            start = state.open_span_start[sid]
                            if start >= 0:
                                span_maps[sid][start] = end
                                state.open_span_start[sid] = -1
                    
                    # Increment bbox indices
                    state.bbox_ind[append_idx] += 1
            
            # ---- Update flags (all on GPU) ----
            # Reset first_lcel when not lcel
            state.first_lcel = torch.where(m_is_lcel, torch.zeros_like(state.first_lcel), torch.ones_like(state.first_lcel))
            
            # Update skip_next
            state.skip_next = safe_isin(new_tags, self.skip_ids)
            
            # Update prev_ucel for NEXT iteration
            state.prev_ucel = (new_tags == self.ucel_id) if self.ucel_id is not None else torch.zeros_like(new_tags, dtype=torch.bool)
            
            # Update line number
            if self.nl_id is not None:
                state.line_num += (new_tags == self.nl_id).to(state.line_num.dtype)
        
        prof.end("batched_ar_loop_v2", self._prof)
        
        # ---- Materialize outputs (single sync point) ----
        # Trim sequences to actual length
        end_id_int = self.end_id.item()
        seqs = []
        for b in range(B):
            seq_len = min(t + 1, state.lengths[b].item() + 1)  # +1 for start token
            seq = self._trim_sequence(decoded_tags[:seq_len, b], end_id_int)
            seqs.append(seq)
        
        # Span maps are already built inline during the loop - no reconstruction needed!
        
        # Reconstruct per-sample tag_H buffers
        tag_H_per_sample = [[] for _ in range(B)]
        for h, sid in zip(state.tag_H_flat, state.tag_H_sample_ids):
            if sid < B:
                tag_H_per_sample[sid].append(h.unsqueeze(0))  # Keep [1,D] shape
        
        # ---- Per-table bbox head (already vectorized) ----
        outputs = []
        prof.begin("batched_bbox_decode_v2", self._prof)
        for b in range(B):
            tag_H_buf_b = tag_H_per_sample[b]
            if self.model._bbox and len(tag_H_buf_b) > 0:
                cls_logits, coords = self.model._bbox_decoder.inference(
                    enc_out_batch[b:b+1], tag_H_buf_b
                )
            else:
                cls_logits = torch.empty(0, device=device)
                coords = torch.empty(0, device=device)
            
            # Merge spans
            merged_cls, merged_coord = self._merge_spans(cls_logits, coords, span_maps[b])
            outputs.append((seqs[b], merged_cls, merged_coord))
        prof.end("batched_bbox_decode_v2", self._prof)
        
        return outputs
    
    def _merge_spans(self, cls_logits: torch.Tensor, coords: torch.Tensor, merge_map: Dict[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
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