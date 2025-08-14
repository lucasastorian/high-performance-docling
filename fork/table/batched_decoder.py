import torch
from typing import List, Dict, Tuple


class BatchedTableDecoder:
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

        # Pre-build lookup tables for fast membership checks (no torch.isin!)
        vocab_size = len(wm)
        emit_lut = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        skip_lut = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        if self.emit_ids.numel() > 0:
            emit_lut[self.emit_ids] = True
        if self.skip_ids.numel() > 0:
            skip_lut[self.skip_ids] = True
        self.emit_lut = emit_lut  # Store as attribute
        self.skip_lut = skip_lut

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
            enc_out_batch: torch.Tensor,  # [B,C,H,W] NCHW - bbox decoder optimized for NCHW
            mem_enc: torch.Tensor,  # [S,B,D] encoder memory (precomputed, no duplicate processing)
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

        # Device-only state tracking
        lengths = torch.zeros(B, dtype=torch.int32, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        first_lcel = torch.ones(B, dtype=torch.bool, device=device)
        skip_next = torch.ones(B, dtype=torch.bool, device=device)
        prev_ucel = torch.zeros(B, dtype=torch.bool, device=device)
        line_num = torch.zeros(B, dtype=torch.long, device=device)
        # bbox_ind removed - using tag_H_counts instead

        # GPU-only bbox tracking with preallocated buffers
        # Worst case: every token emits bbox = Tmax emissions per sample
        hidden_dim = tt._decoder_dim  # This is an attribute of Tag_Transformer, not the decoder!
        # Defer buffer creation until we know the dtype from last_H
        tag_H_buffer = None
        tag_H_flat = None
        tag_H_counts = torch.zeros(B, dtype=torch.long, device=device)  # [B] count per sample

        # Span tracking on GPU - allocate more capacity (Tmax instead of Tmax//2)
        open_span_start = torch.full((B,), -1, dtype=torch.long, device=device)  # [B]
        # Allocate Tmax spans per sample (safe, won't overflow)
        span_starts = torch.full((B, Tmax), -1, dtype=torch.long, device=device)
        span_ends = torch.full((B, Tmax), -1, dtype=torch.long, device=device)
        span_counts = torch.zeros(B, dtype=torch.long, device=device)

        # Drop bbox_ind - just use tag_H_counts (they're identical)

        # Use pre-built LUTs from __init__
        emit_lut = self.emit_lut
        skip_lut = self.skip_lut

        # Preallocate fixed-shape tensors for graph capture
        B_idx = torch.arange(B, device=device)  # [B] for indexing
        ones_B = torch.ones(B, dtype=torch.long, device=device)  # [B] reusable ones
        zeros_B = torch.zeros(B, dtype=torch.bool, device=device)  # [B] reusable zeros
        ones_B_bool = torch.ones(B, dtype=torch.bool, device=device)  # [B] reusable bool ones
        neg_ones_B = torch.full((B,), -1, dtype=torch.long, device=device)  # [B] reusable -1s
        col_mask = torch.zeros(B, Tmax, dtype=torch.bool, device=device)  # [B, Tmax] reusable mask

        # ---- Incremental embedding buffer optimization (Fix B: avoid O(T²)) ----
        D = tt._embedding.embedding_dim
        tgt_emb_buf = torch.empty(Tmax + 1, B, D, device=device)
        pe = tt._positional_encoding.pe  # could be [L, D] OR [L, 1, D]

        def pe_gather_bD(pe: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
            """
            Gather PE rows by per-sample positions -> [B, D], regardless of pe's rank.
            idx must be Long on CUDA.
            """
            idx = idx.to(torch.long)
            out = pe.index_select(0, idx)  # [B, D] or [B,1,D]
            if out.dim() == 3:  # common case: [B,1,D]
                out = out.squeeze(1)  # -> [B, D]
            return out.contiguous()

        # Initialize first step with start tokens (per-sample position indexing)
        start_row = decoded_tags[0, :]  # [B] start tokens
        pos_idx = torch.zeros(B, dtype=torch.long, device=device)  # [B] all zeros
        pos_vec = pe_gather_bD(pe, pos_idx)  # [B, D]
        tgt_emb_buf[0] = tt._embedding(start_row) + pos_vec  # [B, D]

        # Track current step
        t = 0
        # Checkpoint B: Switch to incremental mode (no tag cache)
        sa_kv_cache = None  # KV cache for all layers

        # Step 4: Precompute cross-attention memory K/V once
        USE_MEM_KV = True  # set False to disable the custom path
        mem_kv = tt.precompute_mem_kv(mem_enc) if USE_MEM_KV else None

        for step in range(Tmax):
            # Use step_fullprefix wrapper - now always incremental (Checkpoint C)
            # Pass max_pred_len for optimal KV cache sizing
            last_H, _, sa_kv_cache = tt.step_fullprefix(
                t, tgt_emb_buf, memory=mem_enc, cache=None, memory_kv=mem_kv,
                sa_kv_cache=sa_kv_cache, max_pred_len=Tmax
            )

            # Create tag_H_buffer with correct dtype on first iteration
            if tag_H_buffer is None:
                tag_H_buffer = torch.empty(B, Tmax, hidden_dim, device=device, dtype=last_H.dtype)
                tag_H_flat = tag_H_buffer.view(B * Tmax, hidden_dim)

            logits = tt._fc(last_H)  # [B,V]
            new_tags = logits.argmax(dim=1)  # [B] Long

            # ---- Structure corrections (all on GPU) ----
            if self.xcel_id is not None and self.lcel_id is not None:
                mask_first_line = (line_num == 0) & (new_tags == self.xcel_id)
                new_tags = torch.where(mask_first_line, self.lcel_id.expand(B), new_tags)

            # For ucel->lcel correction, check prev_ucel from PREVIOUS step
            if self.ucel_id is not None and self.lcel_id is not None and self.fcel_id is not None:
                mask_ucel_lcel = prev_ucel & (new_tags == self.lcel_id)
                new_tags = torch.where(mask_ucel_lcel, self.fcel_id.expand(B), new_tags)

            # Force <end> for already finished sequences
            new_tags = torch.where(finished, self.end_id.expand(B), new_tags)

            # Write to preallocated buffer
            t += 1
            decoded_tags[t, :] = new_tags

            # 1) Decide which samples were active BEFORE this step's token
            #    (i.e., they were not already finished)
            active_prev = ~finished  # [B]  -- use the old 'finished' from BEFORE this step

            # 2) Assign the position for the token we just emitted:
            #    - Active sequences advance by +1 (first real token goes to PE[1], etc.)
            #    - Previously-finished sequences DO NOT advance (they keep their last pos)
            pos_idx = torch.where(active_prev, lengths + 1, lengths)  # int32
            pos_idx = pos_idx.clamp_max(pe.size(0) - 1).to(torch.long)  # bounds + dtype
            pos_vec = pe_gather_bD(pe, pos_idx)  # [B, D]

            # Update incremental embedding buffer for next step
            if t < Tmax:  # Only if we'll do another step
                tgt_emb_buf[t] = tt._embedding(new_tags) + pos_vec  # [B, D]

            # 3) Now actually advance lengths for sequences that were active
            lengths = torch.where(active_prev, lengths + 1, lengths)

            # 4) Only AFTER embedding & length update, mark newly-finished for next step
            newly_finished = (new_tags == self.end_id)
            finished = finished | newly_finished

            # No early exit - keep computing for graph capture compatibility

            # ---- BBox emission decisions (TRULY NO SYNCS!) ----
            # Use LUT for fast membership (no isin!)
            m_emit_bbox = (~skip_next) & emit_lut[new_tags] & (~finished)

            # Fix: Proper boolean handling for m_is_lcel
            if self.lcel_id is not None:
                m_is_lcel = (new_tags == self.lcel_id)
            else:
                m_is_lcel = zeros_B

            m_first_lcel = first_lcel & m_is_lcel & (~finished)

            # Collect bbox features with fixed-shape masked writes
            append_mask = m_emit_bbox | m_first_lcel
            positions = tag_H_counts  # [B]
            valid_store = append_mask & (positions < Tmax) & (~finished)  # Bounds check

            # Fixed-shape masked write - no variable K tensors
            row_pos = tag_H_counts.clamp_max(Tmax - 1)  # [B], current per-row cursor
            col_mask.zero_()  # Reset reusable mask
            col_mask[B_idx, row_pos] = valid_store  # Mark positions to update

            # Masked scatter into tag_H_buffer using broadcasting
            expanded_H = last_H.unsqueeze(1)  # [B, 1, D]
            tag_H_buffer = torch.where(col_mask.unsqueeze(-1), expanded_H, tag_H_buffer)

            # Span starts with fixed-shape masked operations
            start_mask = m_first_lcel & valid_store  # [B]
            open_span_start = torch.where(start_mask, tag_H_counts, open_span_start)

            # Span ends with fixed-shape masked operations
            end_mask = m_emit_bbox & (~m_first_lcel) & valid_store & (open_span_start >= 0)  # [B]
            span_capacity_mask = span_counts < Tmax  # [B]
            valid_end_mask = end_mask & span_capacity_mask  # [B]

            # Store spans using masked scatter - write at current span_counts position
            span_positions = span_counts.clamp_max(Tmax - 1)  # [B]
            # Only update where valid_end_mask is True
            span_starts[B_idx, span_positions] = torch.where(valid_end_mask, open_span_start, span_starts[B_idx, span_positions])
            span_ends[B_idx, span_positions] = torch.where(valid_end_mask, tag_H_counts, span_ends[B_idx, span_positions])
            
            # Increment span counts with masked addition
            span_counts = span_counts + valid_end_mask.to(span_counts.dtype)

            # Close spans (reset to -1) using masked assignment
            open_span_start = torch.where(end_mask, neg_ones_B, open_span_start)

            # Counters: increment with masked addition (fixed-shape)
            tag_H_counts = tag_H_counts + valid_store.to(tag_H_counts.dtype)

            # ---- Update flags (all on GPU) ----
            # Reset first_lcel=True on every non-lcel step (matches serial semantics)
            first_lcel = torch.where(m_is_lcel, zeros_B, ones_B_bool)

            # Update skip_next using LUT (no isin!)
            skip_next = skip_lut[new_tags]

            # Update prev_ucel for NEXT iteration
            prev_ucel = (new_tags == self.ucel_id) if self.ucel_id is not None else zeros_B

            # Update line number
            if self.nl_id is not None:
                line_num += (new_tags == self.nl_id).to(line_num.dtype)

        # ---- Check for overflow AFTER loop (single sync, not per-token!) ----
        if __debug__:
            if (tag_H_counts > Tmax).any():
                raise RuntimeError(f"tag_H_counts overflow: max={tag_H_counts.max().item()} > {Tmax}")
            if (span_counts > span_starts.size(1)).any():
                raise RuntimeError(f"span buffer overflow: max={span_counts.max().item()} > {span_starts.size(1)}")

        # ---- Materialize outputs (SINGLE sync point after loop!) ----
        # Trim sequences to actual length
        end_id_int = self.end_id.item()
        seqs = []
        for b in range(B):
            seq_len = min(t + 1, lengths[b].item() + 1)  # +1 for start token
            seq = self._trim_sequence(decoded_tags[:seq_len, b], end_id_int)
            seqs.append(seq)

        # Keep everything on GPU as long as possible
        # Only sync tag_H_counts to know how many hiddens per sample
        tag_H_counts_cpu = tag_H_counts.cpu()

        # ---- Per-table bbox head with tensor inputs ----
        outputs = []
        for b in range(B):
            count = tag_H_counts_cpu[b].item()

            if self.model._bbox and count > 0:
                # Pass tensor directly (no list construction!)
                tag_H_tensor = tag_H_buffer[b, :count].contiguous()  # [N, D] on device
                enc_nchw = enc_out_batch[b:b + 1]  # [1, 256, 28, 28] NCHW

                # Check type explicitly (no try/except overhead)
                if isinstance(tag_H_tensor, torch.Tensor):
                    cls_logits, coords = self.model._bbox_decoder.inference(
                        enc_nchw, tag_H_tensor  # Pass tensor directly!
                    )
                else:
                    # Should never hit this path with our GPU-only collection
                    tag_H_list = [tag_H_tensor[i:i + 1] for i in range(count)]
                    cls_logits, coords = self.model._bbox_decoder.inference(
                        enc_nchw, tag_H_list
                    )
            else:
                cls_logits = torch.empty(0, device=device)
                coords = torch.empty(0, device=device)

            # GPU-based span merging
            merged_cls, merged_coord = self._merge_spans_gpu(
                cls_logits, coords,
                span_starts[b], span_ends[b], span_counts[b]
            )
            outputs.append((seqs[b], merged_cls, merged_coord))

        return outputs

    def _merge_spans_gpu(
            self,
            cls_logits: torch.Tensor,
            coords: torch.Tensor,
            starts: torch.Tensor,  # [Tmax] with -1 for invalid
            ends: torch.Tensor,  # [Tmax] with -1 for invalid
            count: torch.Tensor  # scalar, number of valid spans
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """GPU-based span merging without CPU sync"""
        device = coords.device if coords is not None and coords.numel() > 0 else self.device
        N = len(coords) if coords is not None else 0

        if N == 0 or count == 0:
            return cls_logits, coords

        # Extract valid spans
        num_spans = count.item()  # Single sync point
        if num_spans == 0:
            return cls_logits, coords

        valid_starts = starts[:num_spans].clamp_min(0)  # [M]
        valid_ends = ends[:num_spans].clamp_min(0)  # [M]

        # Create drop mask (mark ends for removal)
        drop = torch.zeros(N, dtype=torch.bool, device=device)
        valid_mask = (valid_ends < N) & (valid_starts < N)
        drop[valid_ends[valid_mask]] = True
        keep = ~drop

        # Merge coordinates for start positions (BATCHED - no Python loop!)
        if valid_mask.any():
            merge_starts = valid_starts[valid_mask]
            merge_ends = valid_ends[valid_mask]

            # Clone coords to avoid in-place issues
            coords = coords.clone()

            # Use batched merge - single kernel, no loop!
            if hasattr(self.model, 'mergebboxes_batch'):
                coords[merge_starts] = self.model.mergebboxes_batch(
                    coords[merge_starts], coords[merge_ends]
                )
            else:
                # Fallback to loop if batched version not available
                for i in range(len(merge_starts)):
                    coords[merge_starts[i]] = self.model.mergebboxes(
                        coords[merge_starts[i]], coords[merge_ends[i]]
                    )

        # Select only kept rows
        return cls_logits[keep], coords[keep]

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
