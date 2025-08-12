# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT

import logging
import os
from typing import List, Tuple, Optional

import torch
import torch.nn as nn

import docling_ibm_models.tableformer.settings as s
from docling_ibm_models.tableformer.models.common.base_model import BaseModel
from docling_ibm_models.tableformer.utils.app_profiler import AggProfiler

from optimized.table.bbox_decoder import BBoxDecoder
from optimized.table.encoder04_rs import Encoder04
from optimized.table.transformer import Tag_Transformer
from optimized.table.batched_decoder import BatchedTableDecoder

LOG_LEVEL = logging.WARN


class TableModel04_rs(BaseModel, nn.Module):
    r"""
    Encoder + dual-decoder (tags + bbox) model with OTSL+ support.
    Optimized to avoid GPU->CPU sync in hot paths, and to minimize tensor churn.
    """

    def __init__(self, config, init_data, device: str):
        super(TableModel04_rs, self).__init__(config, init_data, device)

        self._prof = False
        self._device = device
        self._config = config
        self._init_data = init_data

        # Encoder
        self._enc_image_size = config["model"]["enc_image_size"]
        self._encoder_dim = config["model"]["hidden_dim"]
        self._encoder = Encoder04(self._enc_image_size, self._encoder_dim).to(device)

        # Word map
        word_map = init_data["word_map"]["word_map_tag"]
        tag_vocab_size = len(word_map)

        # td_encode indices (present tags only)
        td_encode = [word_map[t] for t in ["ecel", "fcel", "ched", "rhed", "srow"] if t in word_map]

        # Transformer dims
        self._tag_attention_dim = config["model"]["tag_attention_dim"]
        self._tag_embed_dim = config["model"]["tag_embed_dim"]
        self._tag_decoder_dim = config["model"]["tag_decoder_dim"]
        self._decoder_dim = config["model"]["hidden_dim"]
        self._dropout = config["model"]["dropout"]

        # BBox settings
        self._bbox = config["train"]["bbox"]
        self._bbox_attention_dim = config["model"]["bbox_attention_dim"]
        self._bbox_embed_dim = config["model"]["bbox_embed_dim"]
        self._bbox_decoder_dim = config["model"]["hidden_dim"]
        self._num_classes = config["model"]["bbox_classes"]

        # Transformer layers/heads
        self._enc_layers = config["model"]["enc_layers"]
        self._dec_layers = config["model"]["dec_layers"]
        self._n_heads = config["model"]["nheads"]

        self._max_pred_len = config["predict"]["max_steps"]

        # Tag transformer (SDPA toggle)
        self._tag_transformer = Tag_Transformer(
            device,
            tag_vocab_size,
            td_encode,
            self._decoder_dim,
            self._enc_layers,
            self._dec_layers,
            self._enc_image_size,
            n_heads=self._n_heads,
            use_sdpa=os.environ.get("USE_SDPA", "1") == "1",
        ).to(device)

        # BBox decoder
        self._bbox_decoder = BBoxDecoder(
            device,
            self._bbox_attention_dim,
            self._bbox_embed_dim,
            self._tag_decoder_dim,
            self._bbox_decoder_dim,
            self._num_classes,
            self._encoder_dim,
            self._dropout,
        ).to(device)

        # Batched decoder wrapper (use for ALL batch sizes to avoid per-step syncs)
        self._use_batched_decoder = os.environ.get("USE_BATCHED_DECODER", "1") == "1"
        self._batched_decoder = BatchedTableDecoder(self, device)

        # Precompute tag id tensors once (used in fallback paths/tests)
        self._tag_ids = {}
        for k in ["<start>", "<end>", "xcel", "lcel", "fcel", "ucel", "nl", "ecel", "ched", "rhed", "srow"]:
            if k in word_map:
                self._tag_ids[k] = torch.tensor(word_map[k], device=device, dtype=torch.long)

        # Convenience packed tensors
        def pack(keys):
            return torch.stack([self._tag_ids[k] for k in keys if k in self._tag_ids], dim=0) \
                   if all(k in self._tag_ids for k in keys) else None

        self._bbox_emit_ids = pack(["fcel", "xcel", "ecel", "ched", "rhed", "srow", "nl", "ucel"])
        self._skip_ids = pack(["nl", "ucel", "xcel"])

    def _log(self):
        return s.get_custom_logger(self.__class__.__name__, LOG_LEVEL)

    @staticmethod
    def _flatten_hw_to_sbc(hwc: torch.Tensor) -> torch.Tensor:
        """
        [B,H,W,C] -> [S,B,C] without extra copies where possible.
        """
        B, H, W, C = hwc.shape
        return hwc.flatten(1, 2).permute(1, 0, 2).contiguous()

    def mergebboxes(self, bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
        """
        Merge two [cx,cy,w,h] boxes into a span box. Assumes same device.
        """
        new_w = (bbox2[0] + bbox2[2] / 2) - (bbox1[0] - bbox1[2] / 2)
        new_h = (bbox2[1] + bbox2[3] / 2) - (bbox1[1] - bbox1[3] / 2)
        new_left = bbox1[0] - bbox1[2] / 2
        new_top = torch.minimum(bbox2[1] - bbox2[3] / 2, bbox1[1] - bbox1[3] / 2)
        new_cx = new_left + new_w / 2
        new_cy = new_top + new_h / 2
        return torch.stack([new_cx, new_cy, new_w, new_h])

    @torch.inference_mode()
    def predict(self, imgs: torch.Tensor, max_steps: int, k: int, return_attention: bool = False):
        """
        imgs: [B, 3, 448, 448]
        Returns: list of tuples (seq, outputs_class, outputs_coord) per table.
        """
        prof = AggProfiler()
        prof.begin("predict_total", self._prof)

        self._encoder.eval()
        self._tag_transformer.eval()

        # Optional: AMP (bfloat16) – measure before keeping enabled permanently.
        use_amp = os.environ.get("USE_AMP", "1") == "1"
        cm = torch.autocast(device_type=self._device, dtype=torch.bfloat16) if use_amp else torch.cpu.amp.autocast(enabled=False)

        with cm:
            enc_out_batch = self._encoder(imgs)  # [B, H, W, C]

        # Always use batched decoder when enabled (even for B=1) to avoid scalar syncs
        if self._use_batched_decoder:
            results = self._batched_decoder.predict_batched(enc_out_batch, max_steps)
        else:
            # Fallback: sequential
            results = []
            B = imgs.size(0)
            for i in range(B):
                seq, cls_t, coord_t = self._predict_compat(
                    imgs[i:i+1], max_steps, k, precomputed_enc=enc_out_batch[i:i+1].contiguous()
                )
                results.append((seq, cls_t, coord_t))

        prof.end("predict_total", self._prof)

        # Optional profiler dump
        if self._prof:
            print("\n📊 Model Profiling Results:")
            for key in ["predict_total"]:
                if hasattr(prof, "_timings") and key in prof._timings and prof._timings[key]:
                    avg_ms = (sum(prof._timings[key]) / len(prof._timings[key])) * 1000
                    print(f"  {key:30s}: {avg_ms:.2f} ms")

        return results

    @torch.inference_mode()
    def _predict_compat(
        self,
        imgs: torch.Tensor,
        max_steps: int,
        k: int,
        return_attention: bool = False,
        precomputed_enc: Optional[torch.Tensor] = None,
    ):
        """
        Compatibility single-item path kept for tests/debug.
        Not used in production when batched decoder is enabled.
        """
        device = self._device
        prof = AggProfiler()

        # --- Encoder (or use precomputed) ---
        prof.begin("model_encoder", self._prof)
        enc_out = precomputed_enc if precomputed_enc is not None else self._encoder(imgs)  # [1,H,W,C]
        prof.end("model_encoder", self._prof)

        # --- Tag transformer encoder ---
        # [1,H,W,C] -> [1,h,w,C]
        x = self._tag_transformer._input_filter(enc_out.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        mem = self._flatten_hw_to_sbc(x)  # [S,1,C]

        prof.begin("model_tag_transformer_encoder", self._prof)
        mem_enc = self._tag_transformer._encoder(mem, mask=None)  # [S,1,C]
        prof.end("model_tag_transformer_encoder", self._prof)

        # --- Autoregressive decode (single) – kept minimal, but not hot path ---
        start_id = self._tag_ids["<start>"]
        end_id = self._tag_ids["<end>"]
        decoded_tags = start_id.view(1, 1)  # [T=1, 1]

        cache = None
        output_tags: List[int] = []
        tag_H_buf: List[torch.Tensor] = []

        # Python state (ok here since this is not production path)
        skip_next_tag = True
        prev_tag_ucel = False
        line_num = 0
        first_lcel = True
        bboxes_to_merge = {}
        cur_bbox_ind = -1
        bbox_ind = 0

        for _ in range(self._max_pred_len):
            dec_emb = self._tag_transformer._positional_encoding(self._tag_transformer._embedding(decoded_tags))
            prof.begin("model_tag_transformer_decoder", self._prof)
            decoded, cache = self._tag_transformer._decoder(
                dec_emb, memory=mem_enc, cache=cache, memory_key_padding_mask=None
            )
            prof.end("model_tag_transformer_decoder", self._prof)

            prof.begin("model_tag_transformer_fc", self._prof)
            logits = self._tag_transformer._fc(decoded[-1, :, :])  # [1, vocab]
            prof.end("model_tag_transformer_fc", self._prof)

            new_tag = logits.argmax(1)  # [1]

            # Structure corrections
            if "xcel" in self._tag_ids and "lcel" in self._tag_ids and line_num == 0:
                if new_tag.eq(self._tag_ids["xcel"]).item():
                    new_tag = self._tag_ids["lcel"].view_as(new_tag)
            if "ucel" in self._tag_ids and "lcel" in self._tag_ids and "fcel" in self._tag_ids:
                if prev_tag_ucel and new_tag.eq(self._tag_ids["lcel"]).item():
                    new_tag = self._tag_ids["fcel"].view_as(new_tag)

            if new_tag.eq(end_id).item():
                output_tags.append(int(end_id.item()))
                decoded_tags = torch.cat([decoded_tags, new_tag.view(1, 1)], dim=0)
                break

            # BBox emission and span handling (compat path)
            if self._bbox_emit_ids is not None:
                emit_bbox = torch.isin(new_tag, self._bbox_emit_ids).item()
            else:
                emit_bbox = False

            if not skip_next_tag and emit_bbox:
                tag_H_buf.append(decoded[-1, :, :])
                if not first_lcel:
                    bboxes_to_merge[cur_bbox_ind] = bbox_ind
                bbox_ind += 1

            is_lcel = "lcel" in self._tag_ids and new_tag.eq(self._tag_ids["lcel"]).item()
            if not is_lcel:
                first_lcel = True
            else:
                if first_lcel:
                    tag_H_buf.append(decoded[-1, :, :])
                    first_lcel = False
                    cur_bbox_ind = bbox_ind
                    bboxes_to_merge[cur_bbox_ind] = -1
                    bbox_ind += 1

            # Update flags
            if self._skip_ids is not None:
                skip_next_tag = torch.isin(new_tag, self._skip_ids).item()
            prev_tag_ucel = ("ucel" in self._tag_ids) and new_tag.eq(self._tag_ids["ucel"]).item()
            if "nl" in self._tag_ids and new_tag.eq(self._tag_ids["nl"]).item():
                line_num += 1

            decoded_tags = torch.cat([decoded_tags, new_tag.view(1, 1)], dim=0)
            output_tags.append(int(new_tag.item()))

        seq = decoded_tags.squeeze(1).tolist()  # list of ints

        # BBox head
        if self._bbox:
            prof.begin("model_bbox_decoder", self._prof)
            outputs_class, outputs_coord = self._bbox_decoder.inference(enc_out, tag_H_buf)
            prof.end("model_bbox_decoder", self._prof)
        else:
            outputs_class, outputs_coord = torch.empty(0, device=device), torch.empty(0, device=device)

        # Merge span bboxes safely
        outputs_class, outputs_coord = self._merge_span_bboxes_safe(outputs_class, outputs_coord, bboxes_to_merge)

        return seq, outputs_class, outputs_coord

    def _merge_span_bboxes_safe(self, outputs_class, outputs_coord, bboxes_to_merge):
        """
        Safe bbox merge with O(1) skip tracking and bounds checks.
        """
        device = self._device
        outputs_class = outputs_class if outputs_class is not None else torch.empty(0, device=device)
        outputs_coord = outputs_coord if outputs_coord is not None else torch.empty(0, device=device)

        out_cls, out_coord = [], []
        skip = set()
        N = len(outputs_coord)

        for i in range(N):
            if i in skip:
                continue
            box1 = outputs_coord[i]
            cls1 = outputs_class[i] if len(outputs_class) > i else outputs_class.new_empty(())

            if i in bboxes_to_merge:
                j = bboxes_to_merge[i]
                if 0 <= j < N:
                    skip.add(j)
                    boxm = self.mergebboxes(box1, outputs_coord[j])
                    out_coord.append(boxm)
                    out_cls.append(cls1)
                else:
                    # open span (-1) or invalid -> keep box1
                    out_coord.append(box1)
                    out_cls.append(cls1)
            else:
                out_coord.append(box1)
                out_cls.append(cls1)

        out_coord = torch.stack(out_coord) if len(out_coord) else torch.empty(0, device=device)
        out_cls = torch.stack(out_cls) if len(out_cls) else torch.empty(0, device=device)
        return out_cls, out_coord
