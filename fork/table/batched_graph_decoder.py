import torch
import torch.backends.cudnn as cudnn
import os
from typing import List, Dict, Tuple


class _GraphBlock:
    """Manages a single captured CUDA graph for U-step blocks"""
    def __init__(self, U: int):
        self.U = U
        self.graph = None
        self.pool = None


class BatchedTableDecoder:
    def __init__(self, model, device: str):
        self.model = model
        self.device = device
        self._prof = model._prof
        
        # Disable cuDNN benchmark for stable kernel selection during graph capture
        cudnn.benchmark = False

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
        
        # Block graph infrastructure for CUDA graph capture
        self._graphs = {}  # Dict[(B_bucket, U) -> _GraphBlock]
        self._static = None  # Static tensors for graph capture

    def _trim_sequence(self, seq_tensor: torch.Tensor, end_id: int) -> List[int]:
        """Trim sequence to first <end> token"""
        seq_list = seq_tensor.tolist()
        try:
            idx = seq_list.index(end_id)
            return seq_list[:idx + 1]
        except ValueError:
            return seq_list

    def _allocate_static(self, B_bucket: int):
        """Allocate static tensors for graph capture at given B_bucket size"""
        device = self.device
        tt = self.tt
        Tmax = self.model._max_pred_len
        hidden_dim = tt._decoder_dim
        D = tt._embedding.embedding_dim
        
        S = {}
        S['B_bucket'] = B_bucket
        S['B_active'] = B_bucket
        S['Tmax'] = Tmax
        S['hidden_dim'] = hidden_dim
        S['D'] = D
        
        # Core sequence state
        S['decoded_tags'] = torch.full((Tmax + 1, B_bucket), self.end_id.item(), dtype=torch.long, device=device)
        S['tgt_emb_buf'] = torch.empty(Tmax + 1, B_bucket, D, device=device)
        
        # Per-row state tracking
        S['lengths'] = torch.zeros(B_bucket, dtype=torch.int32, device=device)
        S['finished'] = torch.zeros(B_bucket, dtype=torch.bool, device=device)
        S['first_lcel'] = torch.ones(B_bucket, dtype=torch.bool, device=device)
        S['skip_next'] = torch.ones(B_bucket, dtype=torch.bool, device=device)
        S['prev_ucel'] = torch.zeros(B_bucket, dtype=torch.bool, device=device)
        S['line_num'] = torch.zeros(B_bucket, dtype=torch.long, device=device)
        
        # BBox tracking
        S['tag_H_buffer'] = torch.empty(B_bucket, Tmax, hidden_dim, device=device)
        S['tag_H_flat'] = S['tag_H_buffer'].view(B_bucket * Tmax, hidden_dim)
        S['tag_H_counts'] = torch.zeros(B_bucket, dtype=torch.long, device=device)
        
        # Span tracking
        S['open_span_start'] = torch.full((B_bucket,), -1, dtype=torch.long, device=device)
        S['span_starts'] = torch.full((B_bucket, Tmax), -1, dtype=torch.long, device=device)
        S['span_ends'] = torch.full((B_bucket, Tmax), -1, dtype=torch.long, device=device)
        S['span_counts'] = torch.zeros(B_bucket, dtype=torch.long, device=device)
        
        # Constants and indexing helpers
        S['B_idx'] = torch.arange(B_bucket, device=device)
        S['row_base'] = S['B_idx'] * Tmax
        S['ones_B'] = torch.ones(B_bucket, dtype=torch.long, device=device)
        S['zeros_B'] = torch.zeros(B_bucket, dtype=torch.bool, device=device)
        S['ones_B_bool'] = torch.ones(B_bucket, dtype=torch.bool, device=device)
        S['neg_ones_B'] = torch.full((B_bucket,), -1, dtype=torch.long, device=device)
        
        # Step tracking and control (device scalars)
        S['t'] = torch.zeros(1, dtype=torch.int32, device=device)
        S['finished_all_d'] = torch.zeros(1, dtype=torch.uint8, device=device)
        S['t0'] = 0  # Host-side time base for blocks (no device syncs)
        
        # Per-block buffers for t-agnostic capture (will be sized with U during capture)
        S['block_tags'] = None  # [U, B_bucket] - tokens from current block
        S['cur_tok'] = torch.empty(1, B_bucket, D, device=device)  # [1, B_bucket, D] current token
        
        # Memory placeholders (will be filled during input loading)
        S['mem_enc'] = None  # Will be allocated based on actual input size
        S['mem_kv'] = None   # Precomputed memory K/V
        S['enc_out_batch'] = None  # Will be allocated based on actual input
        
        # Preallocate KV cache with fixed capacity for all layers
        S['sa_kv'] = []
        if hasattr(tt, '_decoder') and hasattr(tt._decoder, 'layers'):
            # Get layer specs from the first layer
            first_layer = tt._decoder.layers[0]
            H = first_layer.self_attn.num_heads
            E = first_layer.self_attn.embed_dim
            Dh = E // H
            cap = Tmax + 1  # Fixed capacity
            
            for _ in tt._decoder.layers:
                K_buf = torch.empty(B_bucket, H, cap, Dh, device=device)
                V_buf = torch.empty(B_bucket, H, cap, Dh, device=device)
                t_dev = torch.zeros(1, dtype=torch.int32, device=device)  # Device tensor time pointer
                pos_cap = torch.arange(cap, device=device, dtype=torch.long)  # Prebuilt indices for masking
                S['sa_kv'].append((K_buf, V_buf, t_dev, cap, pos_cap))
        
        self._static = S

    def _static_load_dummy(self):
        """Load dummy data for warmup before graph capture"""
        S = self._static
        device = self.device
        tt = self.tt
        B_bucket = S['B_bucket']
        Tmax = S['Tmax']
        D = S['D']
        hidden_dim = S['hidden_dim']
        
        # Create minimal dummy memory if not allocated
        if S['mem_enc'] is None:
            S['mem_enc'] = torch.zeros(10, B_bucket, D, device=device)  # Dummy [S=10, B_bucket, D]
        
        # Create dummy encoder outputs if not allocated
        if S['enc_out_batch'] is None:
            S['enc_out_batch'] = torch.zeros(B_bucket, 256, 28, 28, device=device)  # Dummy NCHW
        
        # Initialize with start tokens for all positions
        S['decoded_tags'].fill_(self.end_id.item())
        S['decoded_tags'][0] = self.start_id
        
        # Reset all state - mark all as finished for warmup
        for k in ['lengths', 'tag_H_counts', 'span_counts', 'line_num']:
            S[k].zero_()
        
        S['finished'].fill_(True)  # All finished for warmup
        S['first_lcel'].fill_(True)
        S['skip_next'].fill_(True)
        S['prev_ucel'].zero_()
        
        S['open_span_start'].fill_(-1)
        S['span_starts'].fill_(-1)
        S['span_ends'].fill_(-1)
        
        S['t'].zero_()
        S['t0'] = 0
        S['finished_all_d'].zero_()
        
        # Initialize dummy embeddings
        pe = tt._positional_encoding.pe
        start_tokens = S['decoded_tags'][0]
        pos_idx = torch.zeros(B_bucket, dtype=torch.long, device=device)
        pos_vec = self._pe_gather_bD(pe, pos_idx)
        S['tgt_emb_buf'][0] = tt._embedding(start_tokens) + pos_vec
        
        # Initialize current token for t-agnostic decoder
        S['cur_tok'][0] = tt._embedding(start_tokens) + pos_vec
        
        # Reset KV cache time pointers for all layers (prevent warmup state corruption)
        for kv_tuple in S['sa_kv']:
            if len(kv_tuple) >= 3:  # (K_buf, V_buf, t_dev, cap, pos_cap)
                kv_tuple[2].zero_()  # Reset time pointer to 0
        
        # Precompute dummy memory K/V
        S['mem_kv'] = tt.precompute_mem_kv(S['mem_enc']) if hasattr(tt, 'precompute_mem_kv') else None

    def _static_load_inputs(self, enc_out_batch: torch.Tensor, mem_enc: torch.Tensor, Tmax: int, B_actual: int):
        """Load inputs into static tensors and reset state"""
        S = self._static
        device = self.device
        tt = self.tt
        
        # Allocate memory tensors if needed or resize if different
        S_mem, B_mem, D_mem = mem_enc.shape
        if S['mem_enc'] is None or S['mem_enc'].shape != (S_mem, S['B_bucket'], D_mem):
            S['mem_enc'] = torch.empty(S_mem, S['B_bucket'], D_mem, device=device, dtype=mem_enc.dtype)
        
        # Copy memory (front slice if B_actual < B_bucket)
        S['mem_enc'][:, :B_actual].copy_(mem_enc)
        
        # Allocate encoder output if needed
        if S['enc_out_batch'] is None or S['enc_out_batch'].shape[0] != S['B_bucket']:
            S['enc_out_batch'] = torch.empty(S['B_bucket'], *enc_out_batch.shape[1:], device=device, dtype=enc_out_batch.dtype)
        
        # Copy encoder outputs (front slice)
        S['enc_out_batch'][:B_actual].copy_(enc_out_batch)
        
        # Reset all state tensors
        S['decoded_tags'].fill_(self.end_id.item())
        S['decoded_tags'][0, :B_actual] = self.start_id
        
        # Reset counters and flags
        for k in ['lengths', 'tag_H_counts', 'span_counts', 'line_num']:
            S[k].zero_()
        
        S['finished'].zero_()
        S['first_lcel'].fill_(True)
        S['skip_next'].fill_(True)
        S['prev_ucel'].zero_()
        
        S['open_span_start'].fill_(-1)
        S['span_starts'].fill_(-1)
        S['span_ends'].fill_(-1)
        
        S['t'].zero_()
        S['finished_all_d'].zero_()
        S['t0'] = 0  # Reset host time base
        S['B_active'] = B_actual
        
        # Rebuild indexing for full bucket (padded rows are finished)
        S['B_idx'] = torch.arange(S['B_bucket'], device=device)
        S['row_base'] = S['B_idx'] * S['Tmax']
        
        # Reset KV cache time pointers for all layers (prevent state corruption between runs)
        for kv_tuple in S['sa_kv']:
            if len(kv_tuple) >= 3:  # (K_buf, V_buf, t_dev, cap, pos_cap)
                kv_tuple[2].zero_()  # Reset time pointer to 0
        
        # Precompute memory K/V outside graph
        S['mem_kv'] = tt.precompute_mem_kv(S['mem_enc']) if hasattr(tt, 'precompute_mem_kv') else None
        
        # Initialize first step embeddings for actual batch
        pe = tt._positional_encoding.pe
        start_row = S['decoded_tags'][0, :B_actual]
        pos_idx = torch.zeros(B_actual, dtype=torch.long, device=device)
        pos_vec = self._pe_gather_bD(pe, pos_idx)
        S['tgt_emb_buf'][0, :B_actual] = tt._embedding(start_row) + pos_vec
        
        # Initialize current token for t-agnostic decoder
        S['cur_tok'].zero_()  # Clear all positions
        S['cur_tok'][0, :B_actual] = tt._embedding(start_row) + pos_vec
        
        # Preset padded rows to finished state for graph capture
        if B_actual < S['B_bucket']:
            S['finished'][B_actual:] = True
            S['lengths'][B_actual:] = 0
            S['tag_H_counts'][B_actual:] = 0
            S['span_counts'][B_actual:] = 0

    def _pe_gather_bD(self, pe: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Gather PE rows by per-sample positions -> [B, D]"""
        idx = idx.to(torch.long)
        out = pe.index_select(0, idx)
        if out.dim() == 3:
            out = out.squeeze(1)
        return out.contiguous()

    def _ar_block_advance_U(self, U: int):
        """T-agnostic AR block advance - no time indexing, uses current token embedding"""
        S = self._static
        tt = self.tt
        
        # Alias for readability
        B_bucket = S['B_bucket']
        Tmax = S['Tmax']
        emit_lut = self.emit_lut
        skip_lut = self.skip_lut
        
        for i in range(U):
            # 1) T-agnostic forward pass using current token embedding
            last_H, S['sa_kv'] = tt.step_incremental(
                S['cur_tok'], memory=S['mem_enc'], memory_kv=S['mem_kv'],
                sa_kv_cache=S['sa_kv'], max_pred_len=Tmax
            )
            
            # 2) Get token predictions
            logits = tt._fc(last_H)  # [B_bucket, V]
            new_tags = logits.argmax(dim=1)  # [B_bucket]
            
            # 3) Structure corrections - full bucket
            if self.xcel_id is not None and self.lcel_id is not None:
                mask_first_line = (S['line_num'] == 0) & (new_tags == self.xcel_id)
                new_tags = torch.where(mask_first_line, self.lcel_id.expand(B_bucket), new_tags)
            
            if self.ucel_id is not None and self.lcel_id is not None and self.fcel_id is not None:
                mask_ucel_lcel = S['prev_ucel'] & (new_tags == self.lcel_id)
                new_tags = torch.where(mask_ucel_lcel, self.fcel_id.expand(B_bucket), new_tags)
            
            # Force <end> for finished sequences (includes padded rows)
            new_tags = torch.where(S['finished'], self.end_id.expand(B_bucket), new_tags)
            
            # 4) Store token in per-block buffer at fixed row i (constant during capture)
            S['block_tags'][i] = new_tags
            
            # 5) Update lengths/finished/flags & collect tag_H/spans
            active_prev = ~S['finished']  # [B_bucket]
            S['lengths'] = torch.where(active_prev, S['lengths'] + 1, S['lengths'])
            
            newly_finished = (new_tags == self.end_id)
            S['finished'] = S['finished'] | newly_finished
            
            # BBox emission with efficient indexing - full bucket
            m_emit_bbox = (~S['skip_next']) & emit_lut[new_tags] & (~S['finished'])
            
            if self.lcel_id is not None:
                m_is_lcel = (new_tags == self.lcel_id)
            else:
                m_is_lcel = S['zeros_B']
            
            m_first_lcel = S['first_lcel'] & m_is_lcel & (~S['finished'])
            
            # Efficient bbox feature collection - full bucket
            append_mask = m_emit_bbox | m_first_lcel
            valid_store = append_mask & (S['tag_H_counts'] < Tmax)
            
            # Per-row indexed write for tag_H_buffer - full bucket
            dest_col = S['tag_H_counts'].clamp_max(Tmax - 1)
            flat_idx = S['row_base'] + dest_col
            old_vals = S['tag_H_flat'].index_select(0, flat_idx)
            src = torch.where(valid_store.unsqueeze(1), last_H, old_vals)
            S['tag_H_flat'].index_copy_(0, flat_idx, src)
            
            # Span tracking - full bucket
            start_mask = m_first_lcel & valid_store
            S['open_span_start'] = torch.where(start_mask, S['tag_H_counts'], S['open_span_start'])
            
            end_mask = m_emit_bbox & (~m_first_lcel) & valid_store & (S['open_span_start'] >= 0)
            span_capacity_mask = S['span_counts'] < Tmax
            valid_end_mask = end_mask & span_capacity_mask
            
            # Efficient span storage - full bucket
            span_pos = S['span_counts'].clamp_max(Tmax - 1)
            span_flat_idx = S['row_base'] + span_pos
            
            span_starts_flat = S['span_starts'].view(-1)
            old_starts = span_starts_flat.index_select(0, span_flat_idx)
            new_starts = torch.where(valid_end_mask, S['open_span_start'], old_starts)
            span_starts_flat.index_copy_(0, span_flat_idx, new_starts)
            
            span_ends_flat = S['span_ends'].view(-1)
            old_ends = span_ends_flat.index_select(0, span_flat_idx)
            new_ends = torch.where(valid_end_mask, S['tag_H_counts'], old_ends)
            span_ends_flat.index_copy_(0, span_flat_idx, new_ends)
            
            # Update counters - full bucket
            S['span_counts'] = S['span_counts'] + valid_end_mask.to(S['span_counts'].dtype)
            S['open_span_start'] = torch.where(end_mask, S['neg_ones_B'], S['open_span_start'])
            S['tag_H_counts'] = S['tag_H_counts'] + valid_store.to(S['tag_H_counts'].dtype)
            
            # Update flags - full bucket
            S['first_lcel'] = torch.where(m_is_lcel, S['zeros_B'], S['ones_B_bool'])
            S['skip_next'] = skip_lut[new_tags]
            S['prev_ucel'] = (new_tags == self.ucel_id) if self.ucel_id is not None else S['zeros_B']
            
            if self.nl_id is not None:
                S['line_num'] += (new_tags == self.nl_id).to(S['line_num'].dtype)
            
            # 6) Build next current token embedding in-place (no t needed)
            pos_idx = torch.where(~S['finished'], S['lengths'] + 1, S['lengths'])
            pos_idx = pos_idx.clamp_max(tt._positional_encoding.pe.size(0) - 1).to(torch.long)
            pos_vec = self._pe_gather_bD(tt._positional_encoding.pe, pos_idx)  # [B_bucket, D]
            S['cur_tok'][0] = tt._embedding(new_tags) + pos_vec
            
            # 7) Bump device-side t for trimming purposes
            S['t'].add_(1)
        
        # At end of block, compute finished_all on device
        S['finished_all_d'].copy_(S['finished'].all().to(torch.uint8))

    def _ensure_block_graph(self, B_bucket: int, U: int) -> _GraphBlock:
        """Ensure we have a captured graph for the given (B_bucket, U) configuration"""
        key = (B_bucket, U)
        gb = self._graphs.get(key)
        if gb and gb.graph:
            return gb
        
        # Allocate static buffers for this B_bucket
        if self._static is None or self._static['B_bucket'] != B_bucket:
            self._allocate_static(B_bucket)
        
        # Allocate block_tags buffer for this specific U
        if self._static['block_tags'] is None or self._static['block_tags'].shape[0] != U:
            self._static['block_tags'] = torch.full((U, B_bucket), self.end_id.item(), 
                                                    dtype=torch.long, device=self.device)
        
        # Load dummy data for warmup and capture
        self._static_load_dummy()
        
        # Warmup with eager execution to settle kernels
        for _ in range(3):
            self._ar_block_advance_U(U)
        
        # Create and capture the graph
        gb = _GraphBlock(U)
        gb.pool = torch.cuda.graphs.graph_pool_handle()
        g = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        
        try:
            with torch.cuda.graph(g, pool=gb.pool):
                self._ar_block_advance_U(U)
        except Exception:
            # Make sure no more CUDA calls run here; let the exception unwind
            raise
        
        gb.graph = g
        self._graphs[key] = gb
        return gb

    def _should_compact(self) -> bool:
        """Check if we should compact active rows (simple heuristic)"""
        if self._static is None:
            return False
        S = self._static
        # Compact if active batch drops to half the bucket size
        active_count = (~S['finished'][:S['B_active']]).sum().item()
        return active_count <= S['B_bucket'] // 2 and S['B_bucket'] > 8

    def _compact_active_rows(self) -> int:
        """Compact active rows and return new B_active"""
        S = self._static
        active = (~S['finished'][:S['B_active']]).nonzero(as_tuple=False).squeeze(-1)
        B_active = active.numel()
        
        if B_active == S['B_active']:
            return B_active
        
        # Helper functions for row-wise selection
        def take1D(buf):
            return buf.index_select(0, active)
        
        def take2D(buf):
            return buf.index_select(0, active)
        
        # Compact all per-row state
        for name in ['lengths', 'finished', 'first_lcel', 'skip_next', 'prev_ucel', 'line_num',
                     'tag_H_counts', 'open_span_start', 'span_counts']:
            S[name] = take1D(S[name])
        
        for name in ['span_starts', 'span_ends', 'tag_H_buffer']:
            S[name] = take2D(S[name])
        
        # Update derived tensors
        S['tag_H_flat'] = S['tag_H_buffer'].view(B_active * S['Tmax'], S['hidden_dim'])
        S['B_active'] = B_active
        
        # Recompute indexing helpers
        S['B_idx'] = torch.arange(B_active, device=self.device)
        S['row_base'] = S['B_idx'] * S['Tmax']
        
        # Update reusable tensors to match new size
        S['ones_B'] = torch.ones(B_active, dtype=torch.long, device=self.device)
        S['zeros_B'] = torch.zeros(B_active, dtype=torch.bool, device=self.device)
        S['ones_B_bool'] = torch.ones(B_active, dtype=torch.bool, device=self.device)
        S['neg_ones_B'] = torch.full((B_active,), -1, dtype=torch.long, device=self.device)
        
        return B_active

    def _debug_validate_first_steps(self, max_steps: int = 4):
        """Quick correctness probe for first few steps"""
        if not os.getenv("DEBUG_GRAPH_CAPTURE"):
            return
        
        S = self._static
        print(f"🔍 Debug validation for first {max_steps} steps:")
        
        for step in range(min(max_steps, S['t'].item())):
            if step < S['block_tags'].shape[0]:
                # Top-3 token predictions for this step
                tokens = S['block_tags'][step, :3]  # First 3 batch items
                print(f"  Step {step}: top tokens = {tokens.tolist()}")
        
        # KV cache state validation
        if S['sa_kv']:
            kv_means = []
            t_values = []
            for i, kv_tuple in enumerate(S['sa_kv'][:2]):  # First 2 layers
                if len(kv_tuple) >= 3:
                    K_buf, V_buf, t_dev = kv_tuple[:3]
                    k_mean = K_buf.abs().mean().item()
                    v_mean = V_buf.abs().mean().item()
                    t_val = t_dev.item()
                    kv_means.append((k_mean, v_mean))
                    t_values.append(t_val)
            print(f"  t_dev values: {t_values}")
            print(f"  KV means (K,V): {kv_means}")

    def _materialize_outputs(self) -> List[Tuple[List[int], torch.Tensor, torch.Tensor]]:
        """Convert static tensors to final output format"""
        S = self._static
        device = self.device
        B_active = S['B_active']
        t_final = S['t'].item()
        end_id_int = self.end_id.item()
        
        # Trim sequences to actual length
        seqs = []
        for b in range(B_active):
            seq_len = min(t_final + 1, S['lengths'][b].item() + 1)
            seq = self._trim_sequence(S['decoded_tags'][:seq_len, b], end_id_int)
            seqs.append(seq)
        
        # Get tag_H_counts for bbox processing
        tag_H_counts_cpu = S['tag_H_counts'].cpu()
        
        # Process bbox outputs
        outputs = []
        for b in range(B_active):
            count = tag_H_counts_cpu[b].item()
            
            if self.model._bbox and count > 0:
                tag_H_tensor = S['tag_H_buffer'][b, :count].contiguous()
                enc_nchw = S['enc_out_batch'][b:b + 1]
                
                if isinstance(tag_H_tensor, torch.Tensor):
                    cls_logits, coords = self.model._bbox_decoder.inference(enc_nchw, tag_H_tensor)
                else:
                    tag_H_list = [tag_H_tensor[i:i + 1] for i in range(count)]
                    cls_logits, coords = self.model._bbox_decoder.inference(enc_nchw, tag_H_list)
            else:
                cls_logits = torch.empty(0, device=device)
                coords = torch.empty(0, device=device)
            
            # GPU-based span merging
            merged_cls, merged_coord = self._merge_spans_gpu(
                cls_logits, coords,
                S['span_starts'][b], S['span_ends'][b], S['span_counts'][b]
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
        """
        Block graph approach: capture U-step blocks, replay with early exit between blocks
        """
        B = enc_out_batch.size(0)
        Tmax = min(max_steps, self.model._max_pred_len)
        
        # Block size from environment or default
        U = int(os.getenv("AR_BLOCK_STEPS", "32"))
        num_blocks = (Tmax + U - 1) // U
        
        # Choose B bucket (powers of two for better graph reuse)
        B_bucket = 1 << (B - 1).bit_length()  # next power of 2
        
        # Ensure we have a captured graph for this configuration
        gb = self._ensure_block_graph(B_bucket, U)
        
        # Load inputs into static tensors and reset state
        self._static_load_inputs(enc_out_batch, mem_enc, Tmax, B)
        
        # Block-wise replay with early exit between blocks
        t0 = 0
        for blk in range(num_blocks):
            # Replay the captured graph for U steps
            gb.graph.replay()
            
            # Debug validation after first block
            if blk == 0:
                self._debug_validate_first_steps()
            
            # Splice block tokens into global buffer (host-side, between blocks)
            t1 = min(t0 + U, Tmax)
            u = t1 - t0
            if u > 0:
                self._static['decoded_tags'][t0+1:t1+1, :B] = self._static['block_tags'][:u, :B]
            t0 = t1
            
            # Check if all sequences finished (device->host sync)
            finished_all = bool(self._static['finished_all_d'].item())
            if finished_all:
                break
            
            # Skip compaction for now to avoid state loss (as recommended)
            # Optional: compact active rows to smaller buckets
            # if self._should_compact():
            #     new_B_active = self._compact_active_rows()
            #     new_B_bucket = 1 << (new_B_active - 1).bit_length()
            #     if new_B_bucket < B_bucket:
            #         B_bucket = new_B_bucket
            #         gb = self._ensure_block_graph(B_bucket, U)
        
        # Materialize final outputs (outside graph)
        return self._materialize_outputs()

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
