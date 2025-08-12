#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
# Batched decoder implementation for TableModel04_rs
# Processes multiple tables in parallel instead of sequentially

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Optional, Dict
from docling_ibm_models.tableformer.utils.app_profiler import AggProfiler


@dataclass
class BatchState:
    """State for batched decoding - all tensors stay on GPU"""
    decoded_tags: torch.Tensor          # [T, B] Long - the generated tokens
    finished: torch.Tensor              # [B] bool - which sequences are done
    cache: Optional[torch.Tensor]       # per-layer cache from transformer
    tag_H_buf: List[List[torch.Tensor]] # B lists of step features for bbox head (GPU tensors)
    first_lcel: torch.Tensor            # [B] bool - tracking horizontal spans
    skip_next_tag: torch.Tensor         # [B] bool - skip bbox for next token
    prev_tag_ucel: torch.Tensor         # [B] bool - was previous tag ucel
    line_num: torch.Tensor              # [B] int - current line number
    bboxes_to_merge: List[Dict]         # per item: {start_idx: end_idx}
    bbox_ind: torch.Tensor              # [B] int - current bbox index
    cur_bbox_ind: torch.Tensor          # [B] int - current span start index


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
        n_heads = tt._n_heads
        
        # Pre-filter + flatten encoder features once for all B
        # [B, H, W, C] -> [B, h, w, C]
        AggProfiler().begin("batched_input_filter", self._prof)
        encoder_out = tt._input_filter(enc_out_batch.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        AggProfiler().end("batched_input_filter", self._prof)
        
        batch_size = encoder_out.size(0)
        C = encoder_out.size(-1)
        S = encoder_out.size(1) * encoder_out.size(2)
        
        # Reshape for transformer: [B, S, C] -> [S, B, C]
        # Use reshape instead of view since encoder_out may be non-contiguous after permute
        memory = encoder_out.reshape(B, S, C).permute(1, 0, 2).contiguous()
        
        # Run encoder once for all B (no mask needed - it was all False anyway)
        AggProfiler().begin("batched_encoder", self._prof)
        mem_enc = tt._encoder(memory, mask=None)  # [S, B, C]
        AggProfiler().end("batched_encoder", self._prof)
        
        # Initialize tokens with <start> for all sequences
        start_id = word_map["<start>"]
        end_id = word_map["<end>"]
        decoded_tags = torch.full((1, B), start_id, dtype=torch.long, device=device)
        
        # Initialize per-sequence state
        state = BatchState(
            decoded_tags=decoded_tags,
            finished=torch.zeros(B, dtype=torch.bool, device=device),
            cache=None,
            tag_H_buf=[[] for _ in range(B)],
            first_lcel=torch.ones(B, dtype=torch.bool, device=device),
            skip_next_tag=torch.ones(B, dtype=torch.bool, device=device),  # starts True
            prev_tag_ucel=torch.zeros(B, dtype=torch.bool, device=device),
            line_num=torch.zeros(B, dtype=torch.long, device=device),
            bboxes_to_merge=[{} for _ in range(B)],
            bbox_ind=torch.zeros(B, dtype=torch.long, device=device),
            cur_bbox_ind=torch.full((B,), -1, dtype=torch.long, device=device),
        )
        
        # Main autoregressive loop
        AggProfiler().begin("batched_ar_loop", self._prof)
        for step in range(max_steps):
            # Embed and add positional encoding: [T, B, D]
            tgt_emb = tt._positional_encoding(tt._embedding(state.decoded_tags))
            
            # Decode: returns ([T, B, D], cache)
            AggProfiler().begin("batched_decoder_step", self._prof)
            decoded, state.cache = tt._decoder(
                tgt_emb, 
                memory=mem_enc, 
                cache=state.cache,
                memory_key_padding_mask=None
            )
            AggProfiler().end("batched_decoder_step", self._prof)
            
            # Get logits for last token for all B sequences
            logits = tt._fc(decoded[-1, :, :])  # [B, vocab]
            new_tags = logits.argmax(dim=1)     # [B]
            
            # Apply structure corrections vectorized on device (no CPU sync!)
            xcel_id = word_map.get("xcel", -999)
            lcel_id = word_map.get("lcel", -999)
            fcel_id = word_map.get("fcel", -999)
            
            # First line: xcel -> lcel
            if xcel_id != -999 and lcel_id != -999:
                mask_first_line = (state.line_num == 0) & (new_tags == xcel_id)
                new_tags = torch.where(mask_first_line, torch.full_like(new_tags, lcel_id), new_tags)
            
            # ucel then lcel -> fcel
            if lcel_id != -999 and fcel_id != -999:
                mask_ucel_lcel = state.prev_tag_ucel & (new_tags == lcel_id)
                new_tags = torch.where(mask_ucel_lcel, torch.full_like(new_tags, fcel_id), new_tags)
            
            # Force <end> for finished sequences
            new_tags = torch.where(state.finished, torch.full_like(new_tags, end_id), new_tags)
            
            # Store hidden states for bbox prediction
            last_H = decoded[-1, :, :]  # [B, D]
            
            # Cache tag IDs as tensors once
            if not hasattr(self, '_cached_tag_tensors'):
                device = new_tags.device
                self._cached_tag_tensors = {
                    'fcel': torch.tensor(word_map.get("fcel", -999), device=device),
                    'ecel': torch.tensor(word_map.get("ecel", -999), device=device),
                    'ched': torch.tensor(word_map.get("ched", -999), device=device),
                    'rhed': torch.tensor(word_map.get("rhed", -999), device=device),
                    'srow': torch.tensor(word_map.get("srow", -999), device=device),
                    'nl': torch.tensor(word_map.get("nl", -999), device=device),
                    'ucel': torch.tensor(word_map.get("ucel", -999), device=device),
                    'xcel': torch.tensor(word_map.get("xcel", -999), device=device),
                    'lcel': torch.tensor(word_map.get("lcel", -999), device=device),
                }
            
            # Create masks for vectorized operations (but still need per-batch processing for complex state)
            active_mask = ~torch.tensor(state.finished, dtype=torch.bool, device=new_tags.device)
            
            # Vectorized tag matching
            bbox_tag_ids = torch.stack([
                self._cached_tag_tensors['fcel'], self._cached_tag_tensors['ecel'],
                self._cached_tag_tensors['ched'], self._cached_tag_tensors['rhed'],
                self._cached_tag_tensors['srow'], self._cached_tag_tensors['nl'],
                self._cached_tag_tensors['ucel']
            ])
            emit_bbox_mask = torch.isin(new_tags, bbox_tag_ids)
            
            skip_tag_ids = torch.stack([
                self._cached_tag_tensors['nl'], self._cached_tag_tensors['ucel'], 
                self._cached_tag_tensors['xcel']
            ])
            skip_mask = torch.isin(new_tags, skip_tag_ids)
            
            is_lcel = new_tags.eq(self._cached_tag_tensors['lcel'])
            is_ucel = new_tags.eq(self._cached_tag_tensors['ucel'])
            
            # Process each batch element (reduced scalar extractions)
            for b in range(B):
                if state.finished[b]:
                    continue
                
                # Extract decisions for this batch element
                should_emit = bool(emit_bbox_mask[b].item()) and not state.skip_next_tag[b]
                is_lcel_b = bool(is_lcel[b].item())
                should_skip = bool(skip_mask[b].item())
                is_ucel_b = bool(is_ucel[b].item())
                
                # BBOX PREDICTION logic
                if should_emit:
                    state.tag_H_buf[b].append(last_H[b:b+1, :])  # Keep on GPU
                    if not state.first_lcel[b]:
                        # Mark end index for horizontal cell bbox merge (minimize .item() calls)
                        cur_idx = int(state.cur_bbox_ind[b])
                        bbox_idx = int(state.bbox_ind[b])
                        state.bboxes_to_merge[b][cur_idx] = bbox_idx
                    state.bbox_ind[b] += 1
                
                # Handle horizontal span bboxes
                if not is_lcel_b:
                    state.first_lcel[b] = True
                else:
                    if state.first_lcel[b]:
                        # Beginning of horizontal span
                        state.tag_H_buf[b].append(last_H[b:b+1, :])
                        state.first_lcel[b] = False
                        state.cur_bbox_ind[b] = state.bbox_ind[b]
                        cur_idx = int(state.cur_bbox_ind[b])
                        state.bboxes_to_merge[b][cur_idx] = -1
                        state.bbox_ind[b] += 1
                
                # Update flags
                state.skip_next_tag[b] = should_skip
                state.prev_tag_ucel[b] = is_ucel_b
            
            # Append new tokens
            state.decoded_tags = torch.cat([state.decoded_tags, new_tags.view(1, B)], dim=0)
            state.finished |= (new_tags == end_id)
            
            # Early exit if all sequences are done
            if state.finished.all():
                break
                
        AggProfiler().end("batched_ar_loop", self._prof)
        
        # Convert to per-sequence results
        seqs = [state.decoded_tags[:, b].tolist() for b in range(B)]
        
        # BBox prediction - still per-item for now (could be batched later)
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
            
            # Merge spanning bboxes
            outputs_class, outputs_coord = self._merge_span_bboxes(
                outputs_class, outputs_coord, state.bboxes_to_merge[b]
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
        boxes_to_skip = []
        
        for box_ind in range(len(outputs_coord)):
            box1 = outputs_coord[box_ind].to(device)
            cls1 = outputs_class[box_ind].to(device)
            
            if box_ind in bboxes_to_merge:
                box2_ind = bboxes_to_merge[box_ind]
                if box2_ind >= 0 and box2_ind < len(outputs_coord):
                    boxes_to_skip.append(box2_ind)
                    box2 = outputs_coord[box2_ind].to(device)
                    boxm = self.model.mergebboxes(box1, box2).to(device)
                    outputs_coord1.append(boxm)
                    outputs_class1.append(cls1)
                else:
                    outputs_coord1.append(box1)
                    outputs_class1.append(cls1)
            else:
                if box_ind not in boxes_to_skip:
                    outputs_coord1.append(box1)
                    outputs_class1.append(cls1)
        
        if len(outputs_coord1) > 0:
            outputs_coord1 = torch.stack(outputs_coord1)
        else:
            outputs_coord1 = torch.empty(0, device=device)
            
        if len(outputs_class1) > 0:
            outputs_class1 = torch.stack(outputs_class1)
        else:
            outputs_class1 = torch.empty(0, device=device)
        
        return outputs_class1, outputs_coord1