# fork/table/batched_decoder_v2.py
# Optimized Batched AR decoder with all CPU syncs eliminated

import os
import torch
from typing import List, Dict, Optional, Tuple


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
        
        # CUDA Graph support
        self._graph = None
        self._graph_stream = torch.cuda.Stream() if "cuda" in str(device) else None
        self._static = {}  # holds static buffers used during capture/replay
        self._graph_bs = int(os.getenv("BS_GRAPH", "32"))
        self._graph_tmax = int(os.getenv("TMAX_GRAPH", "160"))

    def _prepare_graph_static(self, enc_out_batch, mem_enc, Tmax, B):
        """Prepare static buffers for CUDA Graph capture"""
        dev = self.device
        tt = self.tt
        D = tt._embedding.embedding_dim
        H = tt._decoder_dim
        S = mem_enc.size(0)
        
        st = self._static
        st.clear()
        
        # ---- Inputs/state that the loop will mutate in-place ----
        st["decoded_tags"] = torch.full((Tmax + 1, B), self.end_id.item(),
                                        dtype=torch.long, device=dev)
        st["decoded_tags"][0].fill_(self.start_id.item())  # seed
        
        st["lengths"] = torch.zeros(B, dtype=torch.int32, device=dev)
        st["finished"] = torch.zeros(B, dtype=torch.bool, device=dev)
        st["first_lcel"] = torch.ones(B, dtype=torch.bool, device=dev)
        st["skip_next"] = torch.ones(B, dtype=torch.bool, device=dev)
        st["prev_ucel"] = torch.zeros(B, dtype=torch.bool, device=dev)
        st["line_num"] = torch.zeros(B, dtype=torch.long, device=dev)
        
        st["tag_H_counts"] = torch.zeros(B, dtype=torch.long, device=dev)
        st["open_span_start"] = torch.full((B,), -1, dtype=torch.long, device=dev)
        st["span_starts"] = torch.full((B, Tmax), -1, dtype=torch.long, device=dev)
        st["span_ends"] = torch.full((B, Tmax), -1, dtype=torch.long, device=dev)
        st["span_counts"] = torch.zeros(B, dtype=torch.long, device=dev)
        
        st["tgt_emb_buf"] = torch.empty(Tmax + 1, B, D, device=dev)
        st["tag_H_buffer"] = torch.empty(B, Tmax, H, device=dev, dtype=torch.float32)
        st["tag_H_flat"] = st["tag_H_buffer"].view(B * Tmax, H)
        
        # Precompute indices
        st["arangeB"] = torch.arange(B, device=dev, dtype=torch.long)
        
        # Static copies of model-constant tensors
        st["pe"] = tt._positional_encoding.pe  # don't clone; it's static
        
        # Memory buffers (pre-allocated, copy into them per call)
        st["mem_enc_buf"] = torch.empty(S, B, D, device=dev)
        st["mem_enc_buf"].copy_(mem_enc)
        
        # Pre-allocate mem_kv buffers per decoder layer
        st["memK"] = []
        st["memV"] = []
        for layer in tt._decoder.layers:
            mha = layer.multihead_attn
            H_mha = mha.num_heads
            Dh = mha.embed_dim // H_mha
            st["memK"].append(torch.empty(B, H_mha, S, Dh, device=dev))
            st["memV"].append(torch.empty(B, H_mha, S, Dh, device=dev))
        
        # Compute mem_kv in-place
        self._rebuild_mem_kv_inplace()
        
        # Encode start row embedding once
        start_row = st["decoded_tags"][0]  # [B]
        pe0 = st["pe"][0] if st["pe"].dim() == 2 else st["pe"][0].squeeze(0)
        st["tgt_emb_buf"][0].copy_(tt._embedding(start_row) + pe0)
        
        # KV cache list (per layer) initialized to None
        st["sa_kv_cache"] = [None] * len(tt._decoder.layers)
        
        # Keep reference to enc_out_batch
        st["enc_out_batch"] = enc_out_batch
    
    def _rebuild_mem_kv_inplace(self):
        """Rebuild memory K/V in-place without allocating new tensors"""
        st = self._static
        tt = self.tt
        mem = st["mem_enc_buf"].transpose(0, 1).contiguous()  # [B,S,D]
        B, S, D = mem.shape
        
        for (Kbuf, Vbuf), layer in zip(zip(st["memK"], st["memV"]), tt._decoder.layers):
            mha = layer.multihead_attn
            E = mha.embed_dim
            H = mha.num_heads
            Dh = E // H
            W = mha.in_proj_weight
            b = mha.in_proj_bias
            Wk = W[E:2*E, :]
            Wv = W[2*E:, :]
            bk = b[E:2*E] if b is not None else None
            bv = b[2*E:] if b is not None else None
            K = torch.nn.functional.linear(mem, Wk, bk)  # [B,S,E]
            V = torch.nn.functional.linear(mem, Wv, bv)  # [B,S,E]
            Kbuf.copy_(K.view(B, S, H, Dh).transpose(1, 2).contiguous())
            Vbuf.copy_(V.view(B, S, H, Dh).transpose(1, 2).contiguous())
    
    def _ar_loop_graph_body(self, Tmax, B):
        """Graph-friendly AR loop with fixed shapes and no breaks"""
        st = self._static
        tt = self.tt
        dev = self.device
        
        # Unpack static tensors
        decoded_tags = st["decoded_tags"]
        lengths = st["lengths"]
        finished = st["finished"]
        first_lcel = st["first_lcel"]
        skip_next = st["skip_next"]
        prev_ucel = st["prev_ucel"]
        line_num = st["line_num"]
        tag_H_counts = st["tag_H_counts"]
        open_span_start = st["open_span_start"]
        span_starts = st["span_starts"]
        span_ends = st["span_ends"]
        span_counts = st["span_counts"]
        tgt_emb_buf = st["tgt_emb_buf"]
        tag_H_buffer = st["tag_H_buffer"]
        tag_H_flat = st["tag_H_flat"]
        pe = st["pe"]
        mem_enc_buf = st["mem_enc_buf"]
        memK = st["memK"]
        memV = st["memV"]
        sa_kv_cache = st["sa_kv_cache"]
        arangeB = st["arangeB"]
        
        # Use pre-built LUTs
        emit_lut = self.emit_lut
        skip_lut = self.skip_lut
        
        # Create mem_kv list from static buffers
        mem_kv = list(zip(memK, memV))
        
        # Run exactly Tmax steps (no early break)
        for t in range(Tmax):
            # Use Python int for indexing
            last_H, _, sa_kv_cache = tt.step_fullprefix(
                t, tgt_emb_buf,
                memory=mem_enc_buf, cache=None, memory_kv=mem_kv,
                sa_kv_cache=sa_kv_cache, max_pred_len=Tmax
            )
            
            # FC + argmax
            logits = tt._fc(last_H)  # [B,V]
            new_tags = logits.argmax(dim=1)  # [B]
            
            # Structure corrections (all on GPU)
            if self.xcel_id is not None and self.lcel_id is not None:
                mask_first_line = (line_num == 0) & (new_tags == self.xcel_id)
                new_tags = torch.where(mask_first_line, self.lcel_id, new_tags)
            
            if self.ucel_id is not None and self.lcel_id is not None and self.fcel_id is not None:
                mask_ucel_lcel = prev_ucel & (new_tags == self.lcel_id)
                new_tags = torch.where(mask_ucel_lcel, self.fcel_id, new_tags)
            
            # Force <end> for finished sequences
            new_tags = torch.where(finished, self.end_id, new_tags)
            
            # Write to preallocated buffer
            decoded_tags[t + 1, :] = new_tags
            
            # PE using broadcast (no per-sample gather)
            pe_row = pe[t + 1] if pe.dim() == 2 else pe[t + 1].squeeze(0)  # [D]
            if t + 1 < Tmax:
                tgt_emb_buf[t + 1] = tt._embedding(new_tags) + pe_row  # broadcast
            
            # Advance lengths for active sequences
            active_prev = ~finished
            lengths = torch.where(active_prev, lengths + 1, lengths)
            
            # Mark newly finished
            finished = finished | (new_tags == self.end_id)
            
            # ---- BBox emission (fixed-shape, no nonzero) ----
            m_emit_bbox = (~skip_next) & emit_lut[new_tags] & (~finished)
            
            if self.lcel_id is not None:
                m_is_lcel = (new_tags == self.lcel_id)
            else:
                m_is_lcel = torch.zeros(B, dtype=torch.bool, device=dev)
            
            m_first_lcel = first_lcel & m_is_lcel & (~finished)
            
            # Compute append mask
            append_mask = m_emit_bbox | m_first_lcel
            
            # Fixed-shape bbox storage using masked operations
            # Always do the operations, but mask controls what gets written
            positions = tag_H_counts
            valid_store = append_mask & (positions < Tmax)
            
            # Use 2D indexing with mask instead of nonzero
            # This writes to all B positions but only updates where valid_store is True
            flat_idx = arangeB * Tmax + positions
            mask_expanded = valid_store.unsqueeze(1).expand(-1, last_H.size(1))
            tag_H_flat[flat_idx] = torch.where(mask_expanded, last_H, tag_H_flat[flat_idx])
            
            # Update counts where valid
            tag_H_counts = tag_H_counts + valid_store.to(tag_H_counts.dtype)
            
            # Span tracking with fixed shapes
            # Start spans where m_first_lcel & append_mask
            span_start_mask = m_first_lcel & append_mask
            open_span_start = torch.where(span_start_mask, positions, open_span_start)
            
            # End spans where appropriate
            span_end_mask = m_emit_bbox & (~m_first_lcel) & append_mask & (open_span_start >= 0)
            if span_end_mask.any():  # Single check to avoid unnecessary work
                # Find where to write spans (use span_counts as index)
                can_write = span_counts < Tmax
                write_mask = span_end_mask & can_write
                
                # Write spans using advanced indexing
                write_idx = torch.where(write_mask)[0]
                if write_idx.numel() > 0:
                    span_idx = span_counts[write_idx]
                    span_starts[write_idx, span_idx] = open_span_start[write_idx]
                    span_ends[write_idx, span_idx] = positions[write_idx]
                    span_counts[write_idx] = span_counts[write_idx] + 1
                
                # Close spans that ended
                open_span_start = torch.where(span_end_mask, torch.tensor(-1, device=dev), open_span_start)
            
            # Update flags
            first_lcel = torch.where(m_is_lcel, torch.zeros_like(first_lcel), torch.ones_like(first_lcel))
            skip_next = skip_lut[new_tags]
            prev_ucel = (new_tags == self.ucel_id) if self.ucel_id is not None else torch.zeros_like(new_tags, dtype=torch.bool)
            
            if self.nl_id is not None:
                line_num = line_num + (new_tags == self.nl_id).to(line_num.dtype)
        
        # Store back mutated state
        st["lengths"] = lengths
        st["finished"] = finished
        st["sa_kv_cache"] = sa_kv_cache
        st["tag_H_counts"] = tag_H_counts
        st["span_counts"] = span_counts
        st["first_lcel"] = first_lcel
        st["skip_next"] = skip_next
        st["prev_ucel"] = prev_ucel
        st["line_num"] = line_num
        st["open_span_start"] = open_span_start
    
    def _capture_if_needed(self, enc_out_batch, mem_enc, Tmax, B):
        """Capture CUDA Graph if not already captured"""
        if self._graph is not None:
            return
        
        if self._graph_stream is None:
            return  # Not on CUDA
        
        torch.cuda.synchronize()
        
        # Warmup on a side stream
        with torch.cuda.stream(self._graph_stream):
            self._prepare_graph_static(enc_out_batch, mem_enc, Tmax, B)
            # One warmup run to populate kernels & cuBLAS handles
            self._ar_loop_graph_body(Tmax, B)
        
        torch.cuda.synchronize()
        
        # Now capture on the same stream
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=self._graph_stream):
            self._ar_loop_graph_body(Tmax, B)
        
        self._graph = g
    
    def _trim_sequence(self, seq_tensor: torch.Tensor, end_id: int) -> List[int]:
        """Trim sequence to first <end> token"""
        seq_list = seq_tensor.tolist()
        try:
            idx = seq_list.index(end_id)
            return seq_list[:idx + 1]
        except ValueError:
            return seq_list

    @torch.inference_mode()
    def predict_batched_graph(
            self,
            enc_out_batch: torch.Tensor,  # [B,C,H,W] 
            mem_enc: torch.Tensor,  # [S,B,D]
            max_steps: int
    ) -> List[Tuple[List[int], torch.Tensor, torch.Tensor]]:
        """CUDA Graph-based prediction"""
        device = self.device
        B_orig = enc_out_batch.size(0)
        S = mem_enc.size(0)
        
        # Use fixed graph dimensions
        B = self._graph_bs
        Tmax = self._graph_tmax
        
        # Pad inputs to graph size if needed
        if B_orig < B:
            # Pad enc_out_batch
            pad_b = B - B_orig
            enc_out_batch = torch.cat([
                enc_out_batch,
                torch.zeros(pad_b, *enc_out_batch.shape[1:], device=device, dtype=enc_out_batch.dtype)
            ], dim=0)
            
            # Pad mem_enc [S,B,D]
            mem_enc = torch.cat([
                mem_enc,
                torch.zeros(S, pad_b, mem_enc.size(2), device=device, dtype=mem_enc.dtype)
            ], dim=1)
        
        # Capture graph if needed
        self._capture_if_needed(enc_out_batch, mem_enc, Tmax, B)
        
        # Reset state for this batch
        st = self._static
        st["decoded_tags"].fill_(self.end_id.item())
        st["decoded_tags"][0].fill_(self.start_id.item())
        st["lengths"].zero_()
        st["finished"].zero_()
        
        # Mark padded samples as finished from the start
        if B_orig < B:
            st["finished"][B_orig:] = True
        
        st["first_lcel"].fill_(True)
        st["skip_next"].fill_(True)
        st["prev_ucel"].zero_()
        st["line_num"].zero_()
        st["tag_H_counts"].zero_()
        st["open_span_start"].fill_(-1)
        st["span_starts"].fill_(-1)
        st["span_ends"].fill_(-1)
        st["span_counts"].zero_()
        
        # Reset KV cache
        st["sa_kv_cache"] = [None] * len(self.tt._decoder.layers)
        
        # Copy memory to static buffers
        st["mem_enc_buf"].copy_(mem_enc)
        self._rebuild_mem_kv_inplace()
        
        # Re-encode start embeddings
        start_row = st["decoded_tags"][0]
        pe0 = st["pe"][0] if st["pe"].dim() == 2 else st["pe"][0].squeeze(0)
        st["tgt_emb_buf"][0].copy_(self.tt._embedding(start_row) + pe0)
        
        # Replay the graph
        self._graph.replay()
        torch.cuda.synchronize()
        
        # Read results from static buffers (only first B_orig samples)
        decoded_tags = st["decoded_tags"][:, :B_orig]
        lengths = st["lengths"][:B_orig]
        tag_H_counts = st["tag_H_counts"][:B_orig]
        tag_H_buffer = st["tag_H_buffer"][:B_orig]
        span_starts = st["span_starts"][:B_orig]
        span_ends = st["span_ends"][:B_orig]
        span_counts = st["span_counts"][:B_orig]
        enc_out_batch = enc_out_batch[:B_orig]
        
        # Process outputs (same as non-graph path)
        end_id_int = self.end_id.item()
        seqs = []
        for b in range(B_orig):
            # Find actual length
            actual_len = lengths[b].item() + 1  # +1 for start token
            seq_len = min(Tmax + 1, actual_len)
            seq = self._trim_sequence(decoded_tags[:seq_len, b], end_id_int)
            seqs.append(seq)
        
        # Process bboxes
        tag_H_counts_cpu = tag_H_counts.cpu()
        outputs = []
        
        for b in range(B_orig):
            count = tag_H_counts_cpu[b].item()
            
            if self.model._bbox and count > 0:
                tag_H_tensor = tag_H_buffer[b, :count].contiguous()
                enc_nchw = enc_out_batch[b:b + 1]
                
                cls_logits, coords = self.model._bbox_decoder.inference(
                    enc_nchw, tag_H_tensor
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
        
        # Check if we should use CUDA Graphs
        use_graph = (
            device == 'cuda' and 
            self._graph_stream is not None and
            B <= self._graph_bs and
            Tmax <= self._graph_tmax and
            os.getenv("USE_CUDA_GRAPHS", "1") == "1"
        )
        
        if use_graph:
            return self.predict_batched_graph(enc_out_batch, mem_enc, max_steps)
        
        # Otherwise use original non-graph path

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
        sa_kv_cache = None     # KV cache for all layers

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

            # Early exit if all finished
            if finished.all():
                break

            # ---- BBox emission decisions (TRULY NO SYNCS!) ----
            # Use LUT for fast membership (no isin!)
            m_emit_bbox = (~skip_next) & emit_lut[new_tags] & (~finished)

            # Fix: Proper boolean handling for m_is_lcel
            if self.lcel_id is not None:
                m_is_lcel = (new_tags == self.lcel_id)
            else:
                m_is_lcel = torch.zeros(B, dtype=torch.bool, device=device)

            m_first_lcel = first_lcel & m_is_lcel & (~finished)

            # Collect bbox features - guard with single check to save kernels
            append_mask = m_emit_bbox | m_first_lcel
            
            # GPT-5 suggestion: Guard the heavy block when nothing happens
            if append_mask.any():  # Single cheap sync that saves many kernels
                positions = tag_H_counts  # [B]
                valid_store = append_mask & (positions < Tmax)  # Bounds check

                # Get samples that need emission
                emit_samples = valid_store.nonzero(as_tuple=False).squeeze(-1)  # [K]
                store_pos = positions.index_select(0, emit_samples)  # [K]
                flat_idx = emit_samples * Tmax + store_pos  # [K]
                src = last_H.index_select(0, emit_samples)  # [K,D]
                tag_H_flat.index_copy_(0, flat_idx, src)

                # Span starts
                start_samples = (m_first_lcel & append_mask).nonzero(as_tuple=False).squeeze(-1)
                open_span_start.index_copy_(0, start_samples, tag_H_counts.index_select(0, start_samples))

                # Span ends
                end_samples = (m_emit_bbox & (~m_first_lcel) & append_mask & (open_span_start >= 0)).nonzero(as_tuple=False).squeeze(-1)

                # Filter for capacity and store
                span_idx = span_counts.index_select(0, end_samples)
                valid = span_idx < span_starts.size(1)
                valid_end_samples = end_samples[valid]
                valid_span_idx = span_idx[valid]

                # Store valid spans (read values BEFORE closing!)
                span_starts[valid_end_samples, valid_span_idx] = open_span_start.index_select(0, valid_end_samples)
                span_ends[valid_end_samples, valid_span_idx] = tag_H_counts.index_select(0, valid_end_samples)
                span_counts.index_add_(0, valid_end_samples, torch.ones_like(valid_span_idx))

                # ALWAYS close spans (even if capacity exceeded) to prevent stuck open spans
                open_span_start.index_fill_(0, end_samples, -1)

                # Counters: increment ONLY rows that actually stored
                ones = torch.ones(emit_samples.size(0), dtype=torch.long, device=device)
                tag_H_counts.index_add_(0, emit_samples, ones)

            # ---- Update flags (all on GPU) ----
            # Reset first_lcel=True on every non-lcel step (matches serial semantics)
            first_lcel = torch.where(m_is_lcel, torch.zeros_like(first_lcel), torch.ones_like(first_lcel))

            # Update skip_next using LUT (no isin!)
            skip_next = skip_lut[new_tags]

            # Update prev_ucel for NEXT iteration
            prev_ucel = (new_tags == self.ucel_id) if self.ucel_id is not None else torch.zeros_like(new_tags,
                                                                                                     dtype=torch.bool)

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
                    tag_H_list = [tag_H_tensor[i:i+1] for i in range(count)]
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
        ends: torch.Tensor,    # [Tmax] with -1 for invalid  
        count: torch.Tensor    # scalar, number of valid spans
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
        valid_ends = ends[:num_spans].clamp_min(0)      # [M]
        
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
