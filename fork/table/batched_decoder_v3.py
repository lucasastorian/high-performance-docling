# fork/table/batched_decoder_v2.py
# Same semantics. Faster by stopping per-step concat & avoiding syncs.

import os
import torch
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import torch.nn as nn

_USE_BF16 = os.getenv("DECODER_BF16", "1") == "1"

# enable SDPA/Flash for nn.MultiheadAttention (PyTorch 2.x)
try:
    torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True)
except Exception:
    pass


@dataclass
class BatchState:
    decoded_tags: torch.Tensor
    lengths: torch.Tensor
    finished: torch.Tensor
    cache: Optional[torch.Tensor]  # kept for API compatibility; we won't use it

    # ragged buffers -> replaced by prealloc tensor + counters in this version
    tag_H_flat: List[torch.Tensor]
    tag_H_sample_ids: List[int]

    first_lcel: torch.Tensor
    skip_next: torch.Tensor
    prev_ucel: torch.Tensor
    line_num: torch.Tensor

    bbox_ind: torch.Tensor
    open_span_start: List[int]


class BatchedTableDecoderV2:
    def __init__(self, model, device: str):
        self.model = model
        self.device = device
        self._prof = model._prof

        self.tt = model._tag_transformer
        wm = model._init_data["word_map"]["word_map_tag"]

        self.start_id = torch.tensor(wm["<start>"], device=device, dtype=torch.long)
        self.end_id   = torch.tensor(wm["<end>"],   device=device, dtype=torch.long)

        self.xcel_id = torch.tensor(wm["xcel"], device=device, dtype=torch.long) if "xcel" in wm else None
        self.lcel_id = torch.tensor(wm["lcel"], device=device, dtype=torch.long) if "lcel" in wm else None
        self.fcel_id = torch.tensor(wm["fcel"], device=device, dtype=torch.long) if "fcel" in wm else None
        self.ucel_id = torch.tensor(wm["ucel"], device=device, dtype=torch.long) if "ucel" in wm else None
        self.nl_id   = torch.tensor(wm["nl"],   device=device, dtype=torch.long) if "nl"   in wm else None

        emit_names = ["fcel","ecel","ched","rhed","srow","nl","ucel"]
        skip_names = ["nl","ucel","xcel"]
        V = self.tt._fc.out_features

        self.emit_lut = torch.zeros(V, dtype=torch.bool, device=device)
        for n in emit_names:
            if n in wm: self.emit_lut[wm[n]] = True

        self.skip_lut = torch.zeros(V, dtype=torch.bool, device=device)
        for n in skip_names:
            if n in wm: self.skip_lut[wm[n]] = True

    @staticmethod
    def _trim_sequence(seq_tensor: torch.Tensor, end_id: int) -> List[int]:
        lst = seq_tensor.tolist()
        try:
            j = lst.index(end_id)
            return lst[:j+1]
        except ValueError:
            return lst

    def _maybe_grow_buffer(self, buf: torch.Tensor, counters: torch.Tensor, needy_idx: torch.Tensor):
        if needy_idx.numel() == 0: return buf, counters
        if not bool((counters[needy_idx] >= buf.size(1)).any()): return buf, counters
        new_K = int(buf.size(1) * 1.5 + 8)
        B, Kold, D = buf.shape
        grown = torch.empty(B, new_K, D, device=buf.device, dtype=buf.dtype)
        grown[:, :Kold].copy_(buf)
        return grown, counters

    def _maybe_grow_spans(self, starts, ends, cnt, idx):
        if idx.numel() == 0: return starts, ends, cnt
        if not bool((cnt[idx] >= starts.size(1)).any()): return starts, ends, cnt
        newK = int(starts.size(1) * 1.5 + 8)
        B = starts.size(0)
        s2 = torch.full((B, newK), -1, device=starts.device, dtype=starts.dtype); s2[:, :starts.size(1)] = starts
        e2 = torch.full((B, newK), -1, device=ends.device,   dtype=ends.dtype);   e2[:, :ends.size(1)]   = ends
        return s2, e2, cnt

    @torch.inference_mode()
    def predict_batched(
        self,
        enc_out_batch: torch.Tensor,   # [B,C,H,W] (NCHW)
        mem_enc: torch.Tensor,         # [S,B,D]
        max_steps: int,
        timer=None
    ) -> List[Tuple[List[int], torch.Tensor, torch.Tensor]]:

        device = self.device
        tt = self.tt
        E  = tt._embedding
        FC = tt._fc
        decoder = tt._decoder
        pos_enc = tt._positional_encoding

        B = enc_out_batch.size(0)
        Tmax = min(max_steps, self.model._max_pred_len)
        D = E.embedding_dim

        # buffers
        decoded_tags = torch.full((Tmax+1, B), self.end_id.item(), dtype=torch.long, device=device)
        decoded_tags[0] = self.start_id
        lengths  = torch.zeros(B, dtype=torch.int32, device=device)
        finished = torch.zeros(B, dtype=torch.bool,  device=device)
        first_lcel = torch.ones(B,  dtype=torch.bool, device=device)
        skip_next  = torch.ones(B,  dtype=torch.bool, device=device)
        prev_ucel  = torch.zeros(B, dtype=torch.bool, device=device)
        line_num   = torch.zeros(B, dtype=torch.long, device=device)
        bbox_ind   = torch.zeros(B, dtype=torch.long, device=device)

        Kmax  = max(1, Tmax // 2)
        Kspan = max(1, Tmax // 2)
        tag_H_buf   = torch.empty(B, Kmax, D, device=device)
        k_counters  = torch.zeros(B, dtype=torch.int32, device=device)
        span_starts = torch.full((B, Kspan), -1, device=device, dtype=torch.long)
        span_ends   = torch.full((B, Kspan), -1, device=device, dtype=torch.long)
        span_cnt    = torch.zeros(B, device=device, dtype=torch.long)

        tgt_emb_buf = torch.empty(Tmax+1, B, D, device=device)
        pe_rows = pos_enc.pe[:Tmax+1, 0]                       # [Tmax+1, D]
        tgt_emb_buf[0] = E(decoded_tags[0]) + pe_rows[0]

        end_vec = self.end_id.expand(B)
        zerosB = torch.zeros(B, dtype=torch.bool, device=device)

        # bf16 autocast for decoder math (keeps logits in fp32)
        use_amp = _USE_BF16 and (device.startswith("cuda"))

        # ===== AR LOOP =====
        if timer: timer.start_section('ar_loop')
        t = 0

        for step in range(Tmax):
            # full prefix slice; we DO NOT pass cache to avoid per-step concat inside decoder
            tgt = tgt_emb_buf[:t+1]  # [t+1,B,D]

            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    decoded, _ = decoder(tgt, memory=mem_enc, cache=None, memory_key_padding_mask=None)
            else:
                decoded, _ = decoder(tgt, memory=mem_enc, cache=None, memory_key_padding_mask=None)

            last_H = decoded[-1]                 # [B,D]
            logits = FC(last_H.float())          # keep logits in fp32 for argmax stability
            new_tags = logits.argmax(dim=1)      # [B]

            is_end  = (new_tags == self.end_id)
            is_lcel = (new_tags == self.lcel_id) if self.lcel_id is not None else zerosB
            emit    = self.emit_lut[new_tags]
            skipnow = self.skip_lut[new_tags]

            # fixes (identical logic)
            if (self.xcel_id is not None) and (self.lcel_id is not None):
                new_tags = torch.where((line_num == 0) & (new_tags == self.xcel_id), self.lcel_id, new_tags)
            if (self.ucel_id is not None) and (self.lcel_id is not None) and (self.fcel_id is not None):
                new_tags = torch.where(prev_ucel & (new_tags == self.lcel_id), self.fcel_id, new_tags)

            finished |= is_end
            new_tags = torch.where(finished, end_vec, new_tags)

            t += 1
            decoded_tags[t] = new_tags
            if t < Tmax:
                tgt_emb_buf[t] = E(new_tags) + pe_rows[t]

            # lengths
            lengths = torch.where(~finished, lengths + 1, lengths)

            # bbox features (vectorized)
            m_emit  = (~skip_next) & emit & (~finished)
            m_first = first_lcel & is_lcel & (~finished)
            append_mask = m_emit | m_first
            append_idx = append_mask.nonzero(as_tuple=False).flatten()
            if append_idx.numel():
                # grow tag_H_buf if needed
                tag_H_buf, k_counters = self._maybe_grow_buffer(tag_H_buf, k_counters, append_idx)
                rows = last_H.index_select(0, append_idx)                # [K,D]
                slots = k_counters[append_idx].long()
                flat = append_idx * tag_H_buf.size(1) + slots
                tag_H_buf.view(-1, D).index_copy_(0, flat, rows)
                k_counters[append_idx] += 1

                # span starts
                first_idx = (m_first & append_mask).nonzero(as_tuple=False).flatten()
                if first_idx.numel():
                    span_starts, span_ends, span_cnt = self._maybe_grow_spans(span_starts, span_ends, span_cnt, first_idx)
                    slot = span_cnt[first_idx]
                    span_starts[first_idx, slot] = bbox_ind[first_idx]
                    span_cnt[first_idx] += 1

                # span ends
                end_idx = (m_emit & (~m_first) & (~finished)).nonzero(as_tuple=False).flatten()
                if end_idx.numel():
                    slot = (span_cnt[end_idx] - 1).clamp_min(0)
                    span_ends[end_idx, slot] = bbox_ind[end_idx]

                bbox_ind[append_idx] += 1

            # flags
            first_lcel = ~is_lcel | finished
            skip_next  = skipnow
            prev_ucel  = (new_tags == self.ucel_id) if self.ucel_id is not None else zerosB
            if self.nl_id is not None:
                line_num += (new_tags == self.nl_id).to(line_num.dtype)

            # early exit check without host roundtrip every step
            if t >= 2:
                # do a cheap check every 4 steps to reduce syncs
                if (step & 3) == 3:
                    if not (~finished).any():
                        break
        if timer: timer.end_section('ar_loop')

        # ==== materialize ====
        end_id_int = int(self.end_id.item())
        seqs = []
        # one host copy for lengths
        lengths_cpu = lengths.cpu()
        for b in range(B):
            seq_len = min(t + 1, int(lengths_cpu[b].item()) + 1)
            seqs.append(self._trim_sequence(decoded_tags[:seq_len, b], end_id_int))

        outputs = []
        if timer: timer.start_section('bbox_decode')
        k_list = k_counters.cpu().tolist()
        for b, k_b in enumerate(k_list):
            if self.model._bbox and k_b > 0:
                enc_nchw = enc_out_batch[b:b+1]
                tag_H_tensor = tag_H_buf[b, :k_b]
                cls_logits, coords = self.model._bbox_decoder.inference(enc_nchw, tag_H_tensor)
            else:
                cls_logits = torch.empty(0, device=device)
                coords = torch.empty(0, device=device)

            # merge spans for b
            out_cls, out_xywh = self._merge_spans_tensor(
                cls_logits, coords,
                span_starts[b:b+1], span_ends[b:b+1], span_cnt[b:b+1]
            )
            outputs.append((seqs[b], out_cls, out_xywh))
        if timer: timer.end_section('bbox_decode')

        return outputs

    def _merge_spans_tensor(self, cls_logits: torch.Tensor, coords: torch.Tensor,
                            starts: torch.Tensor, ends: torch.Tensor, cnt: torch.Tensor):
        device = coords.device if coords.numel() else self.device
        if coords.numel() == 0:
            return torch.empty(0, device=device), torch.empty(0, device=device)

        N = coords.size(0)
        end_map = torch.full((N,), -1, device=device, dtype=torch.long)
        nspans = int(cnt[0].item()) if cnt.numel() > 0 else 0

        if nspans > 0:
            valid = (starts[0, :nspans] >= 0) & (ends[0, :nspans] >= 0)
            if valid.any():
                st = starts[0, :nspans][valid]; ed = ends[0, :nspans][valid]
                ok = (st < N) & (ed < N)
                if ok.any():
                    end_map[st[ok]] = ed[ok]

        merged_cls, merged_coord = [], []
        for i in range(N):
            j = int(end_map[i].item())
            if 0 <= j < N:
                merged_cls.append(cls_logits[i])
                merged_coord.append(self.model.mergebboxes(coords[i], coords[j]))
            else:
                merged_cls.append(cls_logits[i])
                merged_coord.append(coords[i])

        return (torch.stack(merged_cls) if merged_cls else torch.empty(0, device=device),
                torch.stack(merged_coord) if merged_coord else torch.empty(0, device=device))
