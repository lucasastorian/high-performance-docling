#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import logging
import os

import torch
import torch.nn as nn

import docling_ibm_models.tableformer.settings as s
from docling_ibm_models.tableformer.models.common.base_model import BaseModel
from docling_ibm_models.tableformer.utils.app_profiler import AggProfiler
from fork.timers import _CPUTimer, _CudaTimer

from fork.table.encoder04_rs import Encoder04
from fork.table.bbox_decoder_rs import BBoxDecoder
from fork.table.transformer_rs import Tag_Transformer
from fork.table.batched_decoder_v2 import BatchedTableDecoderV2

LOG_LEVEL = logging.WARN


# LOG_LEVEL = logging.INFO
# LOG_LEVEL = logging.DEBUG


class TableModel04_rs(BaseModel, nn.Module):
    r"""
    TableNet04Model encoder, dual-decoder model with OTSL+ support
    """

    def __init__(self, config, init_data, device):
        super(TableModel04_rs, self).__init__(config, init_data, device)

        self._prof = True
        self._device = device
        # Extract the word_map from the init_data
        word_map = init_data["word_map"]

        # Encoder
        self._enc_image_size = config["model"]["enc_image_size"]
        self._encoder = Encoder04(self._enc_image_size).to(device)
        # CRITICAL: use config hidden_dim (512) to match checkpoint bbox decoder shapes
        self._encoder_dim = config["model"]["hidden_dim"]  # 512, not actual encoder output (256)

        tag_vocab_size = len(word_map["word_map_tag"])

        td_encode = []
        for t in ["ecel", "fcel", "ched", "rhed", "srow"]:
            if t in word_map["word_map_tag"]:
                td_encode.append(word_map["word_map_tag"][t])
        self._log().debug("td_encode length: {}".format(len(td_encode)))
        self._log().debug("td_encode: {}".format(td_encode))

        self._tag_attention_dim = config["model"]["tag_attention_dim"]
        self._tag_embed_dim = config["model"]["tag_embed_dim"]
        self._tag_decoder_dim = config["model"]["tag_decoder_dim"]
        self._decoder_dim = config["model"]["hidden_dim"]
        self._dropout = config["model"]["dropout"]

        self._bbox = config["train"]["bbox"]
        self._bbox_attention_dim = config["model"]["bbox_attention_dim"]
        self._bbox_embed_dim = config["model"]["bbox_embed_dim"]
        self._bbox_decoder_dim = config["model"]["hidden_dim"]

        self._enc_layers = config["model"]["enc_layers"]
        self._dec_layers = config["model"]["dec_layers"]
        self._n_heads = config["model"]["nheads"]

        self._num_classes = config["model"]["bbox_classes"]
        self._enc_image_size = config["model"]["enc_image_size"]

        self._max_pred_len = config["predict"]["max_steps"]

        self._tag_transformer = Tag_Transformer(
            device,
            tag_vocab_size,
            td_encode,
            self._decoder_dim,
            self._enc_layers,
            self._dec_layers,
            self._enc_image_size,
            n_heads=self._n_heads,
        ).to(device)

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

        # Stage 2: Cache tag IDs as device tensors (avoid dict hits + reallocs in loop)
        wm_tag = word_map["word_map_tag"]
        self._ids = {k: torch.tensor(v, device=device, dtype=torch.long)
                     for k, v in wm_tag.items() if isinstance(v, int)}

        # Sets for quick membership tests in the loop
        _emit_names = ["fcel", "ecel", "ched", "rhed", "srow", "nl", "ucel"]
        self._emit_ids = torch.stack([self._ids[n] for n in _emit_names if n in self._ids]) \
            if any(n in self._ids for n in _emit_names) else torch.empty(0, dtype=torch.long, device=device)

        _skip_names = ["nl", "ucel", "xcel"]
        self._skip_ids = torch.stack([self._ids[n] for n in _skip_names if n in self._ids]) \
            if any(n in self._ids for n in _skip_names) else torch.empty(0, dtype=torch.long, device=device)

        self._batched_decoder = BatchedTableDecoderV2(self, self._device)

        # Enable fast kernels where safe
        if device == 'cuda':
            # Don't enable benchmark globally - it interferes with CUDA Graphs!
            # torch.backends.cudnn.benchmark = True  # REMOVED - causes slowdown with graphs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")  # Ampere+ only

        # Optimization 8: Optionally disable gradients globally for inference
        if os.getenv("DISABLE_GRAD", "1") == "1":
            torch.set_grad_enabled(False)

        # Block size for encoder batching
        self._encoder_block_bs = int(os.getenv("ENCODER_BLOCK_BS", "32"))

    def _encode_in_blocks(self, imgs: torch.Tensor, block_bs: int = 32) -> torch.Tensor:
        """
        Simple batched encoding without graphs or compilation.
        Just ensures channels_last format and runs through encoder.
        """
        B0, C, H, W = imgs.shape
        device = imgs.device

        # Ensure channels_last format for optimal performance
        imgs_cl = imgs if imgs.is_contiguous(memory_format=torch.channels_last) \
            else imgs.contiguous(memory_format=torch.channels_last)
        
        # Simply run through encoder - no graphs, no compilation
        return self._encoder(imgs_cl)

    @torch.inference_mode()
    def predict(self, imgs, max_steps, k, return_attention=False):
        """
        Stage 3: batched encoder + batched AR decoder.
        imgs: [B,3,448,448]
        returns: list of (seq, outputs_class, outputs_coord)
        """
        B = imgs.size(0)
        self._encoder.eval()
        self._tag_transformer.eval()
        self._bbox_decoder.eval()

        # Use proper timer based on device
        is_cuda = str(self._device).startswith('cuda')
        timer = _CudaTimer() if is_cuda else _CPUTimer()

        # ===== ENCODER TIMING =====
        with timer.time_section('encoder_forward'):
            enc_out_batch = self._encode_in_blocks(imgs, block_bs=self._encoder_block_bs)  # [B,C,H,W] - NCHW format

        # ===== MEMORY PREPARATION =====
        with timer.time_section('tag_input_filter'):
            filtered_nchw = self._tag_transformer._input_filter(enc_out_batch)  # [B,C,h,w] NCHW

        with timer.time_section('memory_reshape'):
            filtered_nhwc = filtered_nchw.permute(0, 2, 3, 1)  # [B,h,w,C] NHWC
            B_, h, w, C = filtered_nhwc.shape
            mem = filtered_nhwc.reshape(B_, h * w, C).permute(1, 0, 2).contiguous()  # [B,h*w,C] -> [h*w,B,C] = [S,B,C]

        # ===== TAG TRANSFORMER ENCODER =====
        with timer.time_section('tag_encoder'):
            mem_enc = self._tag_transformer._encoder(mem, mask=None)  # [S,B,C]

        # ===== BATCHED DECODER =====
        with timer.time_section('batched_ar_decoder'):
            results = self._batched_decoder.predict_batched(enc_out_batch, mem_enc, max_steps)

        # Finalize and print timing if profiling enabled
        if self._prof:
            timer.finalize()
            total_time = (timer.get_time('encoder_forward') + timer.get_time('tag_input_filter') +
                         timer.get_time('memory_reshape') + timer.get_time('tag_encoder') +
                         timer.get_time('batched_ar_decoder'))
            print(f"\n=== TableModel Timing (B={B}) ===")
            print(f"  encoder_forward:    {timer.get_time('encoder_forward'):7.1f} ms")
            print(f"  tag_input_filter:   {timer.get_time('tag_input_filter'):7.1f} ms")
            print(f"  memory_reshape:     {timer.get_time('memory_reshape'):7.1f} ms")
            print(f"  tag_encoder:        {timer.get_time('tag_encoder'):7.1f} ms")
            print(f"  batched_ar_decoder: {timer.get_time('batched_ar_decoder'):7.1f} ms")

            # Print decoder breakdown if available
            if timer.get_time('ar_loop') > 0:
                print(f"\n  === Decoder Breakdown ===")
                print(f"    ar_loop:          {timer.get_time('ar_loop'):7.1f} ms")
                print(f"    bbox_decode:      {timer.get_time('bbox_decode'):7.1f} ms")

            print(f"  ---------------------------")
            print(f"  TOTAL:              {total_time:7.1f} ms\n")

        return results

    def _log(self):
        # Setup a custom logger
        return s.get_custom_logger(self.__class__.__name__, LOG_LEVEL)

    def mergebboxes(self, bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
        """Merge two bboxes using pure tensor ops to avoid CPU sync"""
        new_w = (bbox2[0] + bbox2[2] / 2) - (bbox1[0] - bbox1[2] / 2)
        new_h = (bbox2[1] + bbox2[3] / 2) - (bbox1[1] - bbox1[3] / 2)

        new_left = bbox1[0] - bbox1[2] / 2
        new_top = torch.minimum(bbox2[1] - bbox2[3] / 2, bbox1[1] - bbox1[3] / 2)

        new_cx = new_left + new_w / 2
        new_cy = new_top + new_h / 2

        return torch.stack([new_cx, new_cy, new_w, new_h])  # stays on same device/dtype

    def _flatten_hw_to_sbc(self, hwc):
        """Utility: [B,H,W,C] -> [S,B,C] for transformer"""
        B, H, W, C = hwc.shape
        return hwc.flatten(1, 2).permute(1, 0, 2)  # no .contiguous() - avoid copy
