#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import logging

import torch
import torch.nn as nn

import docling_ibm_models.tableformer.settings as s
from docling_ibm_models.tableformer.models.common.base_model import BaseModel
from docling_ibm_models.tableformer.utils.app_profiler import AggProfiler

from fork.table.encoder04_rs import Encoder04
from fork.table.bbox_decoder_rs import BBoxDecoder
from fork.table.transformer_rs import Tag_Transformer

LOG_LEVEL = logging.WARN
# LOG_LEVEL = logging.INFO
# LOG_LEVEL = logging.DEBUG


class TableModel04_rs(BaseModel, nn.Module):
    r"""
    TableNet04Model encoder, dual-decoder model with OTSL+ support
    """

    def __init__(self, config, init_data, device):
        super(TableModel04_rs, self).__init__(config, init_data, device)

        self._prof = config["predict"].get("profiling", False)
        self._device = device
        # Extract the word_map from the init_data
        word_map = init_data["word_map"]

        # Encoder
        self._enc_image_size = config["model"]["enc_image_size"]
        self._encoder_dim = config["model"]["hidden_dim"]
        self._encoder = Encoder04(self._enc_image_size, self._encoder_dim).to(device)

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

    @torch.inference_mode()
    def predict(self, imgs, max_steps, k, return_attention=False):
        """
        Stage 1 batched inference: batch the image encoder + tag-encoder.
        imgs: FloatTensor [B, 3, 448, 448]
        Returns: list of (seq, outputs_class, outputs_coord) for each item.
        """
        B = imgs.size(0)
        
        # Set modules to eval mode for deterministic behavior
        self._encoder.eval()
        self._tag_transformer.eval()
        self._bbox_decoder.eval()
        
        # Stage 1: Batch the image encoder for all images at once
        # enc_out_batch: [B, H, W, C] where H=W=28, C=256
        AggProfiler().begin("model_encoder", self._prof)
        enc_out_batch = self._encoder(imgs)
        AggProfiler().end("model_encoder", self._prof)
        
        # Stage 1: Apply input_filter to entire batch and prepare transformer memory
        # Apply CNN filter: [B,C,H,W] -> [B,h,w,C] where h=w=28, C=512  
        AggProfiler().begin("model_tag_transformer_input_filter", self._prof)
        filtered_batch = self._tag_transformer._input_filter(
            enc_out_batch.permute(0, 3, 1, 2)
        ).permute(0, 2, 3, 1)
        AggProfiler().end("model_tag_transformer_input_filter", self._prof)
        
        # Flatten spatial dims and transpose for transformer: [B,h,w,C] -> [S,B,C]
        memory_batch = self._flatten_hw_to_sbc(filtered_batch)
        
        # Stage 1: Run transformer encoder once for the entire batch with no mask
        AggProfiler().begin("model_tag_transformer_encoder", self._prof)
        encoder_out_batch = self._tag_transformer._encoder(memory_batch, mask=None)
        AggProfiler().end("model_tag_transformer_encoder", self._prof)
        
        # Stage 2: Per-item AR decode with GPU-only loop
        results = []
        for i in range(B):
            # Extract per-item encoder outputs
            enc_out_i = enc_out_batch[i:i+1]  # [1,H,W,C] for bbox decoder
            encoder_out_i = encoder_out_batch[:, i:i+1, :]  # [S,1,C] for tag decoder
            
            # Use Stage 2 GPU-only AR loop
            seq, outputs_class, outputs_coord = self._predict_single_with_precomputed_stage2(
                enc_out_i, encoder_out_i, max_steps
            )
            results.append((seq, outputs_class, outputs_coord))
        return results
    
    def _flatten_hw_to_sbc(self, hwc):
        """Utility: [B,H,W,C] -> [S,B,C] for transformer"""
        B, H, W, C = hwc.shape
        return hwc.flatten(1, 2).permute(1, 0, 2).contiguous()
    
    def _predict_single_with_precomputed_stage2(self, enc_out: torch.Tensor, memory_enc: torch.Tensor, max_steps: int):
        """
        Stage 2: Device-only AR loop with preallocated buffers.
        enc_out   : [1,H,W,C]  (raw encoder output for bbox head)
        memory_enc: [S,1,C]    (tag-transformer encoder output)
        """
        device = self._device
        tt = self._tag_transformer
        prof = AggProfiler()

        # --- Preallocate decoded tags: [Tmax+1, 1] ---
        Tmax = self._max_pred_len
        decoded_tags = torch.empty((Tmax + 1, 1), dtype=torch.long, device=device)
        start_id = self._ids["<start>"]; end_id = self._ids["<end>"]
        decoded_tags[0, 0] = start_id
        t = 0

        # --- Flags on device ---
        skip_next   = torch.tensor(True,  device=device)
        prev_ucel   = torch.tensor(False, device=device)
        first_lcel  = torch.tensor(True,  device=device)
        line_num    = torch.zeros((), dtype=torch.long, device=device)

        # --- Optional ids (may be None) ---
        lcel_id = self._ids.get("lcel")
        fcel_id = self._ids.get("fcel")
        xcel_id = self._ids.get("xcel")
        ucel_id = self._ids.get("ucel")
        nl_id   = self._ids.get("nl")

        # --- Ragged buffers for bbox head ---
        tag_H_buf: list[torch.Tensor] = []
        bboxes_to_merge: dict[int, int] = {}
        cur_bbox_ind = 0
        bbox_ind = 0
        open_span_start = None

        cache = None

        # --- AR loop ---
        for _ in range(Tmax):
            # [t+1,1,D]
            tgt = tt._positional_encoding(tt._embedding(decoded_tags[:t+1]))

            prof.begin("model_tag_transformer_decoder", self._prof)
            decoded, cache = tt._decoder(tgt, memory=memory_enc, cache=cache, memory_key_padding_mask=None)
            prof.end("model_tag_transformer_decoder", self._prof)

            last_H = decoded[-1, :, :]          # [1,D]
            logits = tt._fc(last_H)             # [1,V]
            new_tag = torch.argmax(logits, dim=1)  # [1] Long (on device)

            # ---- Structure corrections (GPU) ----
            if xcel_id is not None and lcel_id is not None:
                new_tag = torch.where((line_num == 0) & (new_tag == xcel_id),
                                      lcel_id.expand_as(new_tag), new_tag)

            if prev_ucel and lcel_id is not None and fcel_id is not None:
                new_tag = torch.where(new_tag == lcel_id,
                                      fcel_id.expand_as(new_tag), new_tag)

            # ---- Append token ----
            t += 1
            decoded_tags[t, 0] = new_tag

            # ---- Termination ----
            if (new_tag == end_id).item():      # minimal unavoidable sync
                break

            # ---- BBox emission decisions (mostly GPU) ----
            emit_bbox = (~skip_next) & (self._emit_ids.numel() > 0) \
                        & torch.isin(new_tag, self._emit_ids)
            is_lcel = (lcel_id is not None) and (new_tag == lcel_id)

            # Append features for bbox head when emit OR first_lcel happens
            if emit_bbox.item() or (is_lcel and first_lcel).item():
                tag_H_buf.append(last_H)  # GPU tensor per cell
                
                # Handle span tracking
                if is_lcel and first_lcel:
                    open_span_start = cur_bbox_ind
                    bboxes_to_merge[open_span_start] = -1
                    cur_bbox_ind += 1
                elif not first_lcel and open_span_start is not None:
                    # Close the span
                    bboxes_to_merge[open_span_start] = cur_bbox_ind
                    open_span_start = None
                    cur_bbox_ind += 1
                elif emit_bbox.item():
                    cur_bbox_ind += 1

            # Update flags
            if is_lcel:
                first_lcel = torch.tensor(False, device=device)
            else:
                first_lcel = torch.tensor(True, device=device)

            # Update line number if nl
            if nl_id is not None:
                line_num = line_num + (new_tag == nl_id).to(line_num.dtype)

            # Next-step skip mask
            if self._skip_ids.numel() > 0:
                skip_next = torch.isin(new_tag, self._skip_ids)
            else:
                skip_next = torch.tensor(False, device=device)

            # prev_ucel
            prev_ucel = (ucel_id is not None) and (new_tag == ucel_id)

        # --- Materialize sequence ---
        seq = decoded_tags[:t+1, 0].tolist()

        # --- BBox head (unchanged arch) ---
        prof.begin("model_bbox_decoder", self._prof)
        if self._bbox and len(tag_H_buf) > 0:
            cls_logits, coords = self._bbox_decoder.inference(enc_out, tag_H_buf)
        else:
            cls_logits = torch.empty(0, device=device)
            coords     = torch.empty(0, device=device)
        prof.end("model_bbox_decoder", self._prof)

        # --- Merge spans safely (treat -1 as "no merge") ---
        out_cls, out_coord, skip = [], [], set()
        N = len(coords) if coords is not None else 0
        for i in range(N):
            if i in skip: continue
            j = bboxes_to_merge.get(i, None)
            if j is None or j < 0 or j >= N:
                out_cls.append(cls_logits[i]); out_coord.append(coords[i])
            else:
                skip.add(j)
                out_cls.append(cls_logits[i])
                out_coord.append(self.mergebboxes(coords[i], coords[j]))

        out_cls   = torch.stack(out_cls)   if out_cls   else torch.empty(0, device=device)
        out_coord = torch.stack(out_coord) if out_coord else torch.empty(0, device=device)

        return seq, out_cls, out_coord

    def _predict_single_with_precomputed(self, enc_out, encoder_out, max_steps, k, return_attention=False):
        """
        Single-item inference with precomputed encoder outputs (Stage 1 optimization).
        enc_out: [1,H,W,C] from image encoder (for bbox decoder)
        encoder_out: [S,1,C] from tag transformer encoder
        """
        AggProfiler().begin("predict_total", self._prof)

        word_map = self._init_data["word_map"]["word_map_tag"]
        
        # Skip encoder steps - use precomputed outputs
        decoded_tags = (
            torch.LongTensor([word_map["<start>"]]).to(self._device).unsqueeze(1)
        )
        output_tags = []
        cache = None
        tag_H_buf = []

        skip_next_tag = True
        prev_tag_ucel = False
        line_num = 0

        # Populate bboxes_to_merge, indexes of first lcel, and last cell in a span
        first_lcel = True
        bboxes_to_merge = {}
        cur_bbox_ind = -1
        bbox_ind = 0

        # Create dummy encoder_mask for compatibility (not used with precomputed encoder_out)
        encoder_mask = None

        # Autoregressive decoder loop (unchanged from original)
        while len(output_tags) < self._max_pred_len:
            decoded_embedding = self._tag_transformer._embedding(decoded_tags)
            decoded_embedding = self._tag_transformer._positional_encoding(
                decoded_embedding
            )
            AggProfiler().begin("model_tag_transformer_decoder", self._prof)
            decoded, cache = self._tag_transformer._decoder(
                decoded_embedding,
                encoder_out,  # Use precomputed encoder output
                cache,
                memory_key_padding_mask=None,  # No padding mask for dense image features
            )
            AggProfiler().end("model_tag_transformer_decoder", self._prof)
            # Grab last feature to produce token
            AggProfiler().begin("model_tag_transformer_fc", self._prof)
            logits = self._tag_transformer._fc(decoded[-1, :, :])  # 1, vocab_size
            AggProfiler().end("model_tag_transformer_fc", self._prof)
            new_tag = logits.argmax(1).item()

            # STRUCTURE ERROR CORRECTION
            # Correction for first line xcel...
            if line_num == 0:
                if new_tag == word_map["xcel"]:
                    new_tag = word_map["lcel"]

            # Correction for ucel, lcel sequence...
            if prev_tag_ucel:
                if new_tag == word_map["lcel"]:
                    new_tag = word_map["fcel"]

            # End of generation
            if new_tag == word_map["<end>"]:
                output_tags.append(new_tag)
                decoded_tags = torch.cat(
                    [
                        decoded_tags,
                        torch.LongTensor([new_tag]).unsqueeze(1).to(self._device),
                    ],
                    dim=0,
                )  # current_output_len, 1
                break
            output_tags.append(new_tag)

            # BBOX PREDICTION
            # MAKE SURE TO SYNC NUMBER OF CELLS WITH NUMBER OF BBOXes
            if not skip_next_tag:
                if new_tag in [
                    word_map["fcel"],
                    word_map["ecel"],
                    word_map["ched"],
                    word_map["rhed"],
                    word_map["srow"],
                    word_map["nl"],
                    word_map["ucel"],
                ]:
                    # GENERATE BBOX HERE TOO (All other cases)...
                    tag_H_buf.append(decoded[-1, :, :])
                    if first_lcel is not True:
                        # Mark end index for horizontal cell bbox merge
                        bboxes_to_merge[cur_bbox_ind] = bbox_ind
                    bbox_ind += 1

            # Treat horisontal span bboxes...
            if new_tag != word_map["lcel"]:
                first_lcel = True
            else:
                if first_lcel:
                    # GENERATE BBOX HERE (Beginning of horisontal span)...
                    tag_H_buf.append(decoded[-1, :, :])
                    first_lcel = False
                    # Mark start index for cell bbox merge
                    cur_bbox_ind = bbox_ind
                    bboxes_to_merge[cur_bbox_ind] = -1
                    bbox_ind += 1

            if new_tag in [word_map["nl"], word_map["ucel"], word_map["xcel"]]:
                skip_next_tag = True
            else:
                skip_next_tag = False

            # Register ucel in sequence...
            if new_tag == word_map["ucel"]:
                prev_tag_ucel = True
            else:
                prev_tag_ucel = False

            decoded_tags = torch.cat(
                [
                    decoded_tags,
                    torch.LongTensor([new_tag]).unsqueeze(1).to(self._device),
                ],
                dim=0,
            )  # current_output_len, 1
        seq = decoded_tags.squeeze().tolist()

        if self._bbox:
            AggProfiler().begin("model_bbox_decoder", self._prof)
            outputs_class, outputs_coord = self._bbox_decoder.inference(
                enc_out, tag_H_buf  # Use precomputed enc_out
            )
            AggProfiler().end("model_bbox_decoder", self._prof)
        else:
            outputs_class, outputs_coord = None, None

        # Guard .to() calls and actually use the return values
        if outputs_class is not None:
            outputs_class = outputs_class.to(self._device)
        if outputs_coord is not None:
            outputs_coord = outputs_coord.to(self._device)

        ########################################################################################
        # Merge First and Last predicted BBOX for each span, according to bboxes_to_merge
        ########################################################################################

        outputs_class1 = []
        outputs_coord1 = []
        boxes_to_skip = []

        for box_ind in range(len(outputs_coord)):
            box1 = outputs_coord[box_ind].to(self._device)
            cls1 = outputs_class[box_ind].to(self._device)
            if box_ind in bboxes_to_merge:
                j = bboxes_to_merge[box_ind]
                if j >= 0:  # Valid merge target
                    box2 = outputs_coord[j].to(self._device)
                    boxes_to_skip.append(j)
                    boxm = self.mergebboxes(box1, box2)
                    outputs_coord1.append(boxm)
                    outputs_class1.append(cls1)
                else:  # Open span (-1): keep box1 as-is
                    outputs_coord1.append(box1)
                    outputs_class1.append(cls1)
            else:
                if box_ind not in boxes_to_skip:
                    outputs_coord1.append(box1)
                    outputs_class1.append(cls1)

        if len(outputs_coord1) > 0:
            outputs_coord1 = torch.stack(outputs_coord1)
        else:
            outputs_coord1 = torch.empty(0)
        if len(outputs_class1) > 0:
            outputs_class1 = torch.stack(outputs_class1)
        else:
            outputs_class1 = torch.empty(0)

        outputs_class = outputs_class1
        outputs_coord = outputs_coord1

        # Do the rest of the steps...
        AggProfiler().end("predict_total", self._prof)
        num_tab_cells = seq.count(4) + seq.count(5)
        num_rows = seq.count(9)
        self._log().info(
            "OTSL predicted table cells#: {}; rows#: {}".format(num_tab_cells, num_rows)
        )
        return seq, outputs_class, outputs_coord

    def _predict_single(self, imgs, max_steps, k, return_attention=False):
        r"""
        Inference.
        The input image must be preprocessed and transformed.

        Parameters
        ----------
        img : tensor FloatTensor - torch.Size([1, 3, 448, 448])
            Input image for the inference

        Returns
        -------
        seq : list
            Predictions for the tags as indices over the word_map
        outputs_class : tensor(x, 3)
            Classes of predicted bboxes. x is the number of bboxes. There are 3 bbox classes

        outputs_coord : tensor(x, 4)
            Coords of predicted bboxes. x is the number of bboxes. Each bbox is in [cxcywh] format
        """
        AggProfiler().begin("predict_total", self._prof)

        # Invoke encoder
        self._tag_transformer.eval()
        enc_out = self._encoder(imgs)
        AggProfiler().end("model_encoder", self._prof)

        word_map = self._init_data["word_map"]["word_map_tag"]
        n_heads = self._tag_transformer._n_heads
        # [1, 28, 28, 512]
        encoder_out = self._tag_transformer._input_filter(
            enc_out.permute(0, 3, 1, 2)
        ).permute(0, 2, 3, 1)

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        enc_inputs = encoder_out.view(batch_size, -1, encoder_dim).to(self._device)
        enc_inputs = enc_inputs.permute(1, 0, 2)
        positions = enc_inputs.shape[0]

        # No mask needed for dense image features (Stage 1 consistency)
        encoder_mask = None

        # Invoking tag transformer encoder before the loop to save time
        AggProfiler().begin("model_tag_transformer_encoder", self._prof)
        encoder_out = self._tag_transformer._encoder(enc_inputs, mask=None)
        AggProfiler().end("model_tag_transformer_encoder", self._prof)

        decoded_tags = (
            torch.LongTensor([word_map["<start>"]]).to(self._device).unsqueeze(1)
        )
        output_tags = []
        cache = None
        tag_H_buf = []

        skip_next_tag = True
        prev_tag_ucel = False
        line_num = 0

        # Populate bboxes_to_merge, indexes of first lcel, and last cell in a span
        first_lcel = True
        bboxes_to_merge = {}
        cur_bbox_ind = -1
        bbox_ind = 0

        # i = 0
        while len(output_tags) < self._max_pred_len:
            decoded_embedding = self._tag_transformer._embedding(decoded_tags)
            decoded_embedding = self._tag_transformer._positional_encoding(
                decoded_embedding
            )
            AggProfiler().begin("model_tag_transformer_decoder", self._prof)
            decoded, cache = self._tag_transformer._decoder(
                decoded_embedding,
                encoder_out,
                cache,
                memory_key_padding_mask=None,  # No padding mask for dense image features
            )
            AggProfiler().end("model_tag_transformer_decoder", self._prof)
            # Grab last feature to produce token
            AggProfiler().begin("model_tag_transformer_fc", self._prof)
            logits = self._tag_transformer._fc(decoded[-1, :, :])  # 1, vocab_size
            AggProfiler().end("model_tag_transformer_fc", self._prof)
            new_tag = logits.argmax(1).item()

            # STRUCTURE ERROR CORRECTION
            # Correction for first line xcel...
            if line_num == 0:
                if new_tag == word_map["xcel"]:
                    new_tag = word_map["lcel"]

            # Correction for ucel, lcel sequence...
            if prev_tag_ucel:
                if new_tag == word_map["lcel"]:
                    new_tag = word_map["fcel"]

            # End of generation
            if new_tag == word_map["<end>"]:
                output_tags.append(new_tag)
                decoded_tags = torch.cat(
                    [
                        decoded_tags,
                        torch.LongTensor([new_tag]).unsqueeze(1).to(self._device),
                    ],
                    dim=0,
                )  # current_output_len, 1
                break
            output_tags.append(new_tag)

            # BBOX PREDICTION

            # MAKE SURE TO SYNC NUMBER OF CELLS WITH NUMBER OF BBOXes
            if not skip_next_tag:
                if new_tag in [
                    word_map["fcel"],
                    word_map["ecel"],
                    word_map["ched"],
                    word_map["rhed"],
                    word_map["srow"],
                    word_map["nl"],
                    word_map["ucel"],
                ]:
                    # GENERATE BBOX HERE TOO (All other cases)...
                    tag_H_buf.append(decoded[-1, :, :])
                    if first_lcel is not True:
                        # Mark end index for horizontal cell bbox merge
                        bboxes_to_merge[cur_bbox_ind] = bbox_ind
                    bbox_ind += 1

            # Treat horisontal span bboxes...
            if new_tag != word_map["lcel"]:
                first_lcel = True
            else:
                if first_lcel:
                    # GENERATE BBOX HERE (Beginning of horisontal span)...
                    tag_H_buf.append(decoded[-1, :, :])
                    first_lcel = False
                    # Mark start index for cell bbox merge
                    cur_bbox_ind = bbox_ind
                    bboxes_to_merge[cur_bbox_ind] = -1
                    bbox_ind += 1

            if new_tag in [word_map["nl"], word_map["ucel"], word_map["xcel"]]:
                skip_next_tag = True
            else:
                skip_next_tag = False

            # Register ucel in sequence...
            if new_tag == word_map["ucel"]:
                prev_tag_ucel = True
            else:
                prev_tag_ucel = False

            decoded_tags = torch.cat(
                [
                    decoded_tags,
                    torch.LongTensor([new_tag]).unsqueeze(1).to(self._device),
                ],
                dim=0,
            )  # current_output_len, 1
        seq = decoded_tags.squeeze().tolist()

        if self._bbox:
            AggProfiler().begin("model_bbox_decoder", self._prof)
            outputs_class, outputs_coord = self._bbox_decoder.inference(
                enc_out, tag_H_buf
            )
            AggProfiler().end("model_bbox_decoder", self._prof)
        else:
            outputs_class, outputs_coord = None, None

        outputs_class.to(self._device)
        outputs_coord.to(self._device)

        ########################################################################################
        # Merge First and Last predicted BBOX for each span, according to bboxes_to_merge
        ########################################################################################

        outputs_class1 = []
        outputs_coord1 = []
        boxes_to_skip = []

        for box_ind in range(len(outputs_coord)):
            box1 = outputs_coord[box_ind].to(self._device)
            cls1 = outputs_class[box_ind].to(self._device)
            if box_ind in bboxes_to_merge:
                j = bboxes_to_merge[box_ind]
                if j >= 0:  # Valid merge target
                    box2 = outputs_coord[j].to(self._device)
                    boxes_to_skip.append(j)
                    boxm = self.mergebboxes(box1, box2)
                    outputs_coord1.append(boxm)
                    outputs_class1.append(cls1)
                else:  # Open span (-1): keep box1 as-is
                    outputs_coord1.append(box1)
                    outputs_class1.append(cls1)
            else:
                if box_ind not in boxes_to_skip:
                    outputs_coord1.append(box1)
                    outputs_class1.append(cls1)

        if len(outputs_coord1) > 0:
            outputs_coord1 = torch.stack(outputs_coord1)
        else:
            outputs_coord1 = torch.empty(0)
        if len(outputs_class1) > 0:
            outputs_class1 = torch.stack(outputs_class1)
        else:
            outputs_class1 = torch.empty(0)

        outputs_class = outputs_class1
        outputs_coord = outputs_coord1

        # Do the rest of the steps...
        AggProfiler().end("predict_total", self._prof)
        num_tab_cells = seq.count(4) + seq.count(5)
        num_rows = seq.count(9)
        self._log().info(
            "OTSL predicted table cells#: {}; rows#: {}".format(num_tab_cells, num_rows)
        )
        return seq, outputs_class, outputs_coord
