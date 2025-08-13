#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import logging

import torch
import torch.nn as nn

import docling_ibm_models.tableformer.settings as s
import docling_ibm_models.tableformer.utils.utils as u

# from scipy.optimize import linear_sum_assignment

LOG_LEVEL = logging.INFO


class CellAttention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, tag_decoder_dim, language_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param tag_decoder_dim: size of tag decoder's RNN
        :param language_dim: size of language model's RNN
        :param attention_dim: size of the attention network
        """
        super(CellAttention, self).__init__()
        # linear layer to transform encoded image
        self._encoder_att = nn.Linear(encoder_dim, attention_dim)
        # linear layer to transform tag decoder output
        self._tag_decoder_att = nn.Linear(tag_decoder_dim, attention_dim)
        # linear layer to transform language models output
        self._language_att = nn.Linear(language_dim, attention_dim)
        # linear layer to calculate values to be softmax-ed
        self._full_att = nn.Linear(attention_dim, 1)
        self._relu = nn.ReLU()
        self._softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def _log(self):
        # Setup a custom logger
        return s.get_custom_logger(self.__class__.__name__, LOG_LEVEL)

    def forward(self, encoder_out, decoder_hidden, language_out):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (1, num_pixels, encoder_dim)
        :param decoder_hidden: tag decoder output, a tensor of dimension [(num_cells,
                               tag_decoder_dim)]
        :param language_out: language model output, a tensor of dimension (num_cells,
                               language_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self._encoder_att(encoder_out)  # (1, num_pixels, attention_dim)
        att2 = self._tag_decoder_att(decoder_hidden)  # (num_cells, tag_decoder_dim)
        att3 = self._language_att(language_out)  # (num_cells, attention_dim)
        att = self._full_att(
            self._relu(att1 + att2.unsqueeze(1) + att3.unsqueeze(1))
        ).squeeze(2)
        alpha = self._softmax(att)  # (num_cells, num_pixels)
        # (num_cells, encoder_dim)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return attention_weighted_encoding, alpha


class BBoxDecoder(nn.Module):
    """
    CellDecoder generates cell content
    """

    def __init__(
        self,
        device,
        attention_dim,
        embed_dim,
        tag_decoder_dim,
        decoder_dim,
        num_classes,
        encoder_dim=512,
        dropout=0.5,
        cnn_layer_stride=1,
    ):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param tag_decoder_dim: size of tag decoder's RNN
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        :param mini_batch_size: batch size of cells to reduce GPU memory usage
        """
        super(BBoxDecoder, self).__init__()
        self._device = device
        self._encoder_dim = encoder_dim
        self._attention_dim = attention_dim
        self._embed_dim = embed_dim
        self._decoder_dim = decoder_dim
        self._dropout = dropout
        self._num_classes = num_classes

        if cnn_layer_stride is not None:
            self._input_filter = u.resnet_block(stride=cnn_layer_stride)
        # attention network
        self._attention = CellAttention(
            encoder_dim, tag_decoder_dim, decoder_dim, attention_dim
        )
        # decoder LSTMCell
        self._init_h = nn.Linear(encoder_dim, decoder_dim)

        # linear layer to create a sigmoid-activated gate
        self._f_beta = nn.Linear(decoder_dim, encoder_dim)
        self._sigmoid = nn.Sigmoid()
        self._dropout = nn.Dropout(p=self._dropout)
        self._class_embed = nn.Linear(512, self._num_classes + 1)
        self._bbox_embed = u.MLP(512, 256, 4, 3)

    def _init_hidden_state(self, encoder_out, batch_size):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self._init_h(mean_encoder_out).expand(batch_size, -1)
        return h

    def _log(self):
        # Setup a custom logger
        return s.get_custom_logger(self.__class__.__name__, LOG_LEVEL)

    @torch.inference_mode()
    def inference(self, enc_out_nchw, tag_H, tile_pixels: int = 128, use_amp: bool = False):
        """
        Vectorized, memory-efficient inference for ONE table.
        - No new parameters.
        - Accepts NCHW to avoid extra permutes.
        - Uses streaming softmax over pixels to avoid [N,P,A] blow-up.

        Args
        ----
        enc_out_nchw : Tensor [1, C, H, W]
        tag_H        : List[Tensor] each [1, D] or [D]
        tile_pixels  : int, pixel tile size along P=H*W
        use_amp      : whether to autocast to bf16/fp16 (try after parity check)

        Returns
        -------
        logits_cls : [N, num_classes+1]
        boxes      : [N, 4] in cxcywh
        """
        assert enc_out_nchw.dim() == 4 and enc_out_nchw.size(0) == 1, \
            "bbox inference expects a single table (B=1)."

        device = enc_out_nchw.device
        dtype = enc_out_nchw.dtype

        # Optional autocast (turn on after verifying parity)
        autocast_cm = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if use_amp else \
            torch.cpu.amp.autocast(enabled=False)

        with autocast_cm:
            # 1) Optional conv filter (kept to preserve weights/behavior)
            if hasattr(self, "_input_filter"):
                # enc_out_nchw: [1,C,H,W] -> same layout
                B, C, H, W = enc_out_nchw.shape
                assert B == 1 and C == 256, f"Expected [1,256,H,W] NCHW input, got [{B},{C},{H},{W}]"
                enc_out_nchw = self._input_filter(enc_out_nchw)

            # 2) Flatten pixels once: [1,C,H,W] -> [P,C]
            B, C, H, W = enc_out_nchw.shape
            P = H * W
            enc_flat = enc_out_nchw.reshape(1, C, P).permute(0, 2, 1).reshape(P, C).contiguous()  # [P,C]

            # 3) Stack tag hidden states -> [N, D]
            if len(tag_H) == 0:
                empty = torch.empty(0, device=device, dtype=dtype)
                return empty, empty

            tag_H_stacked = []
            for t in tag_H:
                t = t.squeeze(0) if (t.dim() == 2 and t.size(0) == 1) else t.reshape(-1)
                tag_H_stacked.append(t)
            tag_H_stacked = torch.stack(tag_H_stacked, dim=0).to(device=device, dtype=dtype)  # [N, D]
            N = tag_H_stacked.size(0)

            # 4) Precompute linear pieces (no broadcasts yet)
            # att layers (shared with original CellAttention to keep weights identical)
            att_enc = self._attention._encoder_att  # Linear(C -> A)
            att_tag = self._attention._tag_decoder_att  # Linear(D -> A)
            att_lang = self._attention._language_att  # Linear(dec_dim -> A)
            att_out = self._attention._full_att  # Linear(A -> 1)
            relu = self._attention._relu

            # (a) Per-pixel embedding (P,A) – compute once
            S_p = att_enc(enc_flat)  # [P, A]

            # (b) Per-cell embeddings (N,A)
            R_n = att_tag(tag_H_stacked)  # [N, A]

            # (c) Init language hidden h0 from pixel mean, expand to N
            mean_enc = enc_flat.mean(dim=0, keepdim=True)  # [1, C]
            h0 = self._init_hidden_state(mean_enc.unsqueeze(0), N)  # [N, dec_dim]
            L_n = att_lang(h0)  # [N, A]

            Rn_plus_Ln = R_n + L_n  # [N, A]
            w = att_out.weight.view(-1)  # [A]
            b = att_out.bias  # [1] or []

            # 5) Streaming softmax over pixels (two-pass; numerically stable)
            # First pass: max over P per row(n)
            row_max = torch.full((N,), -float("inf"), device=device, dtype=S_p.dtype)
            for start in range(0, P, tile_pixels):
                stop = min(start + tile_pixels, P)
                # compute Z_tile: [N, T] via (N,T,A) but only for small T
                Sp_tile = S_p[start:stop]  # [T, A]
                # broadcast add -> [N, T, A]
                sum_NTA = (Rn_plus_Ln.unsqueeze(1) + Sp_tile.unsqueeze(0))  # [N, T, A]
                relu_NTA = relu(sum_NTA)  # [N, T, A]
                # project to 1: (N,T)
                Z_tile = relu_NTA.matmul(w)  # [N, T]
                if b is not None:
                    Z_tile = Z_tile + b
                # rowwise max update
                tile_max, _ = Z_tile.max(dim=1)
                row_max = torch.maximum(row_max, tile_max)

            # Second pass: accumulate exp and weighted sum of encodings
            row_sum = torch.zeros(N, device=device, dtype=S_p.dtype)  # denominator
            awe = torch.zeros(N, C, device=device, dtype=S_p.dtype)  # numerator for weighted sum

            for start in range(0, P, tile_pixels):
                stop = min(start + tile_pixels, P)
                Sp_tile = S_p[start:stop]  # [T, A]
                enc_tile = enc_flat[start:stop]  # [T, C]

                sum_NTA = (Rn_plus_Ln.unsqueeze(1) + Sp_tile.unsqueeze(0))  # [N, T, A]
                relu_NTA = relu(sum_NTA)  # [N, T, A]
                Z_tile = relu_NTA.matmul(w)  # [N, T]
                if b is not None:
                    Z_tile = Z_tile + b

                # exp with max subtraction
                Z_tile = Z_tile - row_max.unsqueeze(1)  # [N, T]
                E_tile = torch.exp(Z_tile)  # [N, T]

                # denom
                row_sum += E_tile.sum(dim=1)  # [N]

                # numerator for awe: (N,T) @ (T,C) -> (N,C)
                awe += E_tile.matmul(enc_tile)  # [N, C]

            # finalize attention-weighted encoding
            awe = awe / row_sum.unsqueeze(1)  # [N, C]

            # 6) Gate + combine EXACTLY like original
            gate = self._sigmoid(self._f_beta(h0))  # [N, C]
            awe = gate * awe  # [N, C]
            h = awe * h0  # [N, C]
            h = self._dropout(h)

            # 7) Heads (unchanged)
            logits_cls = self._class_embed(h)  # [N, num_classes+1]
            boxes = self._bbox_embed(h).sigmoid()  # [N, 4]
            return logits_cls, boxes

    # def inference(self, encoder_out, tag_H):
    #     """
    #     Inference on test images with beam search
    #     """
    #     if hasattr(self, "_input_filter"):
    #         encoder_out = self._input_filter(encoder_out.permute(0, 3, 1, 2)).permute(
    #             0, 2, 3, 1
    #         )
    #
    #     encoder_dim = encoder_out.size(3)
    #
    #     # Flatten encoding (1, num_pixels, encoder_dim)
    #     encoder_out = encoder_out.view(1, -1, encoder_dim)
    #
    #     num_cells = len(tag_H)
    #     predictions_bboxes = []
    #     predictions_classes = []
    #
    #     for c_id in range(num_cells):
    #         # Start decoding
    #         h = self._init_hidden_state(encoder_out, 1)
    #         cell_tag_H = tag_H[c_id]
    #         awe, _ = self._attention(encoder_out, cell_tag_H, h)
    #         gate = self._sigmoid(self._f_beta(h))
    #         awe = gate * awe
    #         h = awe * h
    #
    #         predictions_bboxes.append(self._bbox_embed(h).sigmoid())
    #         predictions_classes.append(self._class_embed(h))
    #     if len(predictions_bboxes) > 0:
    #         predictions_bboxes = torch.stack([x[0] for x in predictions_bboxes])
    #     else:
    #         predictions_bboxes = torch.empty(0)
    #
    #     if len(predictions_classes) > 0:
    #         predictions_classes = torch.stack([x[0] for x in predictions_classes])
    #     else:
    #         predictions_classes = torch.empty(0)
    #
    #     return predictions_classes, predictions_bboxes
