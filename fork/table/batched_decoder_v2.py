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
    
    @torch.inference_mode()
    def predict_batched(
        self,
        enc_out_batch: torch.Tensor,   # [B,C,H,W] NCHW - bbox decoder optimized for NCHW
        mem_enc: torch.Tensor,         # [S,B,D] encoder memory (precomputed, no duplicate processing)
        max_steps: int
    ) -> List[Tuple[List[int], torch.Tensor, torch.Tensor]]:
        device = self.device
        tt = self.tt
        B = enc_out_batch.size(0)
        
        # Clamp to model's max
        Tmax = min(max_steps, self.model._max_pred_len)
        
        # ---- Preallocate all buffers (Fix B: avoid reallocation) ----
        decoded_tags = torch.full((Tmax + 1, B), self.end_id.item(), dtype=torch.long, device=device)
        decoded_tags[0, :] = self.start_id
        
        # Initialize span maps correctly - one dict per sample
        span_maps: List[Dict[int, int]] = [dict() for _ in range(B)]
        
        # Device-only state tracking
        lengths = torch.zeros(B, dtype=torch.int32, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        first_lcel = torch.ones(B, dtype=torch.bool, device=device)
        skip_next = torch.ones(B, dtype=torch.bool, device=device)
        prev_ucel = torch.zeros(B, dtype=torch.bool, device=device)
        line_num = torch.zeros(B, dtype=torch.long, device=device)
        bbox_ind = torch.zeros(B, dtype=torch.long, device=device)
        
        # OPTIMIZATION 2: Preallocate tag hidden states buffer [B, Kmax, D]
        D = tt._embedding.embedding_dim
        Kmax = max(1, Tmax // 2)  # Safe upper bound for tag hidden states
        tag_H_buf = torch.empty(B, Kmax, D, device=device)
        k_counters = torch.zeros(B, dtype=torch.int32, device=device)  # Track count per sample
        
        # Per-sample span tracking
        open_span_start = [-1] * B  # Track one open span per sample
        
        prof = AggProfiler()
        prof.begin("batched_ar_loop_v2", self._prof)
        
        # ---- Incremental embedding buffer optimization (Fix B: avoid O(T²)) ----
        D = tt._embedding.embedding_dim
        tgt_emb_buf = torch.empty(Tmax + 1, B, D, device=device)
        
        # FIX 1: Clean PE shape handling - extract [Tmax+1, D] directly
        pe = tt._positional_encoding.pe  # [max_len, 1, D] positional encoding buffer
        pos_rows = pe[:Tmax + 1, 0]  # [Tmax+1, D] - clean 2D tensor, no awkward broadcasting
        
        # FIX 4: Prebuild end vector once
        end_vec = torch.full((B,), self.end_id.item(), device=device, dtype=torch.long)
        
        # Initialize first step with start tokens
        start_row = decoded_tags[0, :]  # [B] start tokens
        tgt_emb_buf[0] = tt._embedding(start_row) + pos_rows[0]  # [B,D] + [D] broadcasts cleanly
        
        # Track current step
        t = 0
        cache = None
        
        for step in range(Tmax):
            # Use incremental embedding buffer - only pass what we've computed so far
            tgt = tgt_emb_buf[:t+1, :, :]  # [t+1,B,D] - grows each step
            
            # Transformer decoder with cache
            prof.begin("decoder_step", self._prof)
            decoded, cache = tt._decoder(
                tgt, memory=mem_enc, cache=cache, memory_key_padding_mask=None
            )
            prof.end("decoder_step", self._prof)
            
            # FIX: Ensure contiguous for better memory layout
            last_H = decoded[-1].contiguous()  # [B,D]
            logits = tt._fc(last_H)            # [B,V]
            new_tags = logits.argmax(dim=1)    # [B] Long
            
            # ---- Structure corrections (all on GPU) ----
            if self.xcel_id is not None and self.lcel_id is not None:
                mask_first_line = (line_num == 0) & (new_tags == self.xcel_id)
                new_tags = torch.where(mask_first_line, self.lcel_id.expand(B), new_tags)
            
            # For ucel->lcel correction, check prev_ucel from PREVIOUS step
            if self.ucel_id is not None and self.lcel_id is not None and self.fcel_id is not None:
                mask_ucel_lcel = prev_ucel & (new_tags == self.lcel_id)
                new_tags = torch.where(mask_ucel_lcel, self.fcel_id.expand(B), new_tags)
            
            # Force <end> for already finished sequences - FIX 4: Use prebuilt end_vec
            new_tags = torch.where(finished, end_vec, new_tags)
            
            # Write to preallocated buffer
            t += 1
            decoded_tags[t, :] = new_tags
            
            # Update incremental embedding buffer for next step
            if t < Tmax:  # Only if we'll do another step
                # FIX 1: Use clean pos_rows without unsqueeze
                tgt_emb_buf[t] = tt._embedding(new_tags) + pos_rows[t]  # [B,D] + [D]
            
            # Update lengths for non-finished sequences
            lengths = torch.where(~finished, lengths + 1, lengths)
            
            # Update finished status
            newly_finished = (new_tags == self.end_id)
            finished |= newly_finished
            
            # Early exit if all finished
            if finished.all():
                break
            
            # ---- BBox emission decisions (minimize syncs) ----
            # OPTIMIZATION 1: Use LUT instead of torch.isin for O(1) lookup
            emit_mask = self.emit_lut[new_tags]  # [B] bool - O(1) lookup
            m_emit_bbox = (~skip_next) & emit_mask & (~finished)
            m_is_lcel = (new_tags == self.lcel_id) if self.lcel_id is not None else torch.zeros(B, dtype=torch.bool, device=device)
            m_first_lcel = first_lcel & m_is_lcel & (~finished)
            
            # Collect indices that need bbox features (minimize CPU syncs)
            append_mask = m_emit_bbox | m_first_lcel
            if append_mask.any():
                append_idx = append_mask.nonzero(as_tuple=False).squeeze(1)
                if append_idx.numel() > 0:
                    # FIX 2: Check and grow buffer if needed before writing
                    tag_H_buf, k_counters = self._maybe_grow_buffer(tag_H_buf, k_counters, append_idx)
                    
                    # OPTIMIZATION 2: Write to preallocated buffer instead of list
                    rows = last_H.index_select(0, append_idx)  # [K,D]
                    slots = k_counters[append_idx].long()      # [K] - current slot for each sample
                    flat_idx = append_idx * tag_H_buf.size(1) + slots  # [K] - use current buffer size
                    tag_H_buf.view(-1, D).index_copy_(0, flat_idx, rows)  # Vectorized write
                    k_counters[append_idx] += 1  # Increment counters
                    
                    # Handle span tracking (device operations where possible)
                    first_lcel_mask = m_first_lcel & append_mask
                    if first_lcel_mask.any():
                        first_lcel_idx = first_lcel_mask.nonzero(as_tuple=False).squeeze(1)
                        # Record span starts - use bbox_ind BEFORE increment
                        start_indices = bbox_ind[first_lcel_idx].tolist()  # Batch sync
                        for sid, start in zip(first_lcel_idx.tolist(), start_indices):
                            open_span_start[sid] = start  # Track open span start index
                    
                    # Handle span ends
                    end_span_mask = m_emit_bbox & (~first_lcel) & append_mask
                    if end_span_mask.any():
                        end_idx = end_span_mask.nonzero(as_tuple=False).squeeze(1)
                        end_indices = bbox_ind[end_idx].tolist()  # Batch sync
                        for sid, end in zip(end_idx.tolist(), end_indices):
                            start = open_span_start[sid]
                            if start >= 0:  # -1 means no open span
                                span_maps[sid][start] = end  # Record span for merging
                                open_span_start[sid] = -1  # Close span
                    
                    # Increment bbox indices
                    bbox_ind[append_idx] += 1
            
            # ---- Update flags (all on GPU) ----
            # Reset first_lcel=True on every non-lcel step (matches serial semantics)
            first_lcel = torch.where(m_is_lcel, torch.zeros_like(first_lcel), torch.ones_like(first_lcel))
            
            # Update skip_next - OPTIMIZATION 1: Use LUT instead of torch.isin
            skip_next = self.skip_lut[new_tags]  # [B] bool - O(1) lookup
            
            # Update prev_ucel for NEXT iteration
            prev_ucel = (new_tags == self.ucel_id) if self.ucel_id is not None else torch.zeros_like(new_tags, dtype=torch.bool)
            
            # Update line number
            if self.nl_id is not None:
                line_num += (new_tags == self.nl_id).to(line_num.dtype)
        
        prof.end("batched_ar_loop_v2", self._prof)
        
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
        prof.begin("batched_bbox_decode_v2", self._prof)
        for b in range(B):
            # OPTIMIZATION 2: Extract from preallocated buffer
            k_b = int(k_counters[b].item())
            if self.model._bbox and k_b > 0:
                # Pass NCHW directly (bbox decoder optimized for NCHW)
                enc_nchw = enc_out_batch[b:b+1]  # [1, 256, 28, 28] NCHW
                tag_H_tensor = tag_H_buf[b, :k_b]  # [k_b, D] - extract used portion
                cls_logits, coords = self.model._bbox_decoder.inference(
                    enc_nchw, tag_H_tensor
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