#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import logging

import torch
import torch.nn as nn

import docling_ibm_models.tableformer.settings as s
import docling_ibm_models.tableformer.utils.utils as u

LOG_LEVEL = logging.INFO


class CellAttention(nn.Module):
    """
    Attention Network - optimized for batch processing.
    """

    def __init__(self, encoder_dim, tag_decoder_dim, language_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param tag_decoder_dim: size of tag decoder's RNN
        :param language_dim: size of language model's RNN
        :param attention_dim: size of the attention network
        """
        super(CellAttention, self).__init__()
        self._encoder_att = nn.Linear(encoder_dim, attention_dim)
        self._tag_decoder_att = nn.Linear(tag_decoder_dim, attention_dim)
        self._language_att = nn.Linear(language_dim, attention_dim)
        self._full_att = nn.Linear(attention_dim, 1)
        self._relu = nn.ReLU()

    def _log(self):
        return s.get_custom_logger(self.__class__.__name__, LOG_LEVEL)

    def forward(self, encoder_out, decoder_hidden, language_out):
        """
        Vectorized forward propagation for batch processing.
        :param encoder_out: encoded images, a tensor of dimension (1, num_pixels, encoder_dim)
        :param decoder_hidden: tag decoder output, a tensor of dimension (num_cells, tag_decoder_dim)
        :param language_out: language model output, a tensor of dimension (num_cells, language_dim)
        :return: attention weighted encoding, weights
        """
        # encoder_out: [1, P, encoder_dim]
        # decoder_hidden: [N, tag_decoder_dim]
        # language_out: [N, language_dim]
        
        N = decoder_hidden.size(0)  # num_cells
        P = encoder_out.size(1)     # num_pixels
        
        att1 = self._encoder_att(encoder_out)          # [1, P, attention_dim]
        att2 = self._tag_decoder_att(decoder_hidden)   # [N, attention_dim]
        att3 = self._language_att(language_out)        # [N, attention_dim]
        
        # Broadcast to [N, P, attention_dim]
        att = self._full_att(self._relu(
            att1.expand(N, -1, -1) + att2[:, None, :] + att3[:, None, :]
        )).squeeze(2)  # [N, P]
        
        alpha = torch.softmax(att, dim=1)  # [N, P]
        
        # Weighted sum using bmm for efficiency
        attention_weighted_encoding = torch.bmm(
            alpha.unsqueeze(1), 
            encoder_out.expand(N, -1, -1)
        ).squeeze(1)  # [N, encoder_dim]
        
        return attention_weighted_encoding, alpha


class BBoxDecoder(nn.Module):
    """
    Optimized BBoxDecoder with vectorized inference - NO NEW PARAMETERS
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
        
        # EXACT SAME architecture as original - no new layers!
        self._attention = CellAttention(
            encoder_dim, tag_decoder_dim, decoder_dim, attention_dim
        )
        self._init_h = nn.Linear(encoder_dim, decoder_dim)
        self._f_beta = nn.Linear(decoder_dim, encoder_dim)
        self._sigmoid = nn.Sigmoid()
        self._dropout = nn.Dropout(p=dropout)
        self._class_embed = nn.Linear(512, self._num_classes + 1)  # Keep 512 to match checkpoint
        self._bbox_embed = u.MLP(512, 256, 4, 3)  # Keep 512 to match checkpoint

    def _init_hidden_state(self, encoder_out, batch_size):
        """Initialize hidden state for batch_size cells"""
        mean_encoder_out = encoder_out.mean(dim=1)  # [1, encoder_dim]
        h = self._init_h(mean_encoder_out)  # [1, decoder_dim]
        if batch_size > 1:
            h = h.expand(batch_size, -1)  # [batch_size, decoder_dim]
        return h

    def _log(self):
        return s.get_custom_logger(self.__class__.__name__, LOG_LEVEL)

    @torch.inference_mode()
    def inference(self, encoder_out, tag_H):
        """
        Vectorized inference - process all cells at once WITHOUT adding new parameters
        :param encoder_out: [1, H, W, C] encoded features
        :param tag_H: list of tensors with shape [1, tag_decoder_dim] or [tag_decoder_dim]
        :return: (predictions_classes, predictions_bboxes)
        """
        if hasattr(self, "_input_filter"):
            encoder_out = self._input_filter(encoder_out.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # Flatten encoding to [1, P, C]
        B, H, W, C = encoder_out.shape
        assert B == 1, "BBoxDecoder.inference expects a single table per call"
        encoder_out = encoder_out.reshape(1, H * W, C).contiguous()  # [1, P, C]
        P = encoder_out.size(1)
        device = encoder_out.device
        dtype = encoder_out.dtype  # Match dtype for AMP compatibility

        N = len(tag_H)
        if N == 0:
            empty = torch.empty(0, device=device, dtype=dtype)
            return empty, empty

        # Stack tag hidden states -> [N, tag_decoder_dim]
        tag_H_stacked = []
        for t in tag_H:
            if t.dim() == 2 and t.size(0) == 1:
                t = t.squeeze(0)
            elif t.dim() > 2:
                t = t.reshape(-1)  # Use reshape for safety with non-contiguous tensors
            tag_H_stacked.append(t)
        tag_H_stacked = torch.stack(tag_H_stacked, dim=0).to(device=device, dtype=dtype).contiguous()  # [N, tag_decoder_dim]

        # ---- Vectorized attention using EXISTING CellAttention layers ----
        # Access the SAME parameters to keep state_dict unchanged
        att_enc = self._attention._encoder_att(encoder_out)       # [1, P, A]
        att_tag = self._attention._tag_decoder_att(tag_H_stacked) # [N, A]

        # Init language hidden h0 from mean encoder; shape [N, decoder_dim]
        h0 = self._init_hidden_state(encoder_out, N)              # [N, dec_dim]
        att_lang = self._attention._language_att(h0)              # [N, A]

        # Combine and score -> [N, P]
        att = self._attention._full_att(
            self._attention._relu(
                att_enc.expand(N, -1, -1) + att_tag[:, None, :] + att_lang[:, None, :]
            )
        ).squeeze(2)                                              # [N, P]

        # Softmax over pixels
        alpha = torch.softmax(att, dim=1)                         # [N, P]

        # Weighted sum via bmm - avoid expand if N==1
        # encoder_out: [1, P, C], alpha: [N, P] -> bmm: [N, 1, P] @ [N, P, C] => [N, C]
        enc_exp = encoder_out if N == 1 else encoder_out.expand(N, -1, -1)
        awe = torch.bmm(alpha.unsqueeze(1), enc_exp).squeeze(1)  # [N, C]

        # Gate + combine EXACTLY like original
        gate = self._sigmoid(self._f_beta(h0))                    # [N, C]
        awe = gate * awe                                          # [N, C]
        h = awe * h0                                               # [N, C]

        # Apply dropout
        h = self._dropout(h)

        # Generate predictions for all cells at once
        logits_cls = self._class_embed(h)                         # [N, num_classes+1]
        boxes = self._bbox_embed(h).sigmoid()                     # [N, 4] cxcywh

        return logits_cls, boxes

    def inference_original(self, encoder_out, tag_H):
        """
        Keep original implementation for comparison/fallback
        """
        if hasattr(self, "_input_filter"):
            encoder_out = self._input_filter(encoder_out.permute(0, 3, 1, 2)).permute(
                0, 2, 3, 1
            )

        encoder_dim = encoder_out.size(3)
        encoder_out = encoder_out.view(1, -1, encoder_dim)

        num_cells = len(tag_H)
        predictions_bboxes = []
        predictions_classes = []

        for c_id in range(num_cells):
            h = self._init_hidden_state(encoder_out, 1)
            cell_tag_H = tag_H[c_id]
            awe, _ = self._attention(encoder_out, cell_tag_H, h)
            gate = self._sigmoid(self._f_beta(h))
            awe = gate * awe
            h = awe * h

            predictions_bboxes.append(self._bbox_embed(h).sigmoid())
            predictions_classes.append(self._class_embed(h))
            
        if len(predictions_bboxes) > 0:
            predictions_bboxes = torch.stack([x[0] for x in predictions_bboxes])
        else:
            predictions_bboxes = torch.empty(0)

        if len(predictions_classes) > 0:
            predictions_classes = torch.stack([x[0] for x in predictions_classes])
        else:
            predictions_classes = torch.empty(0)

        return predictions_classes, predictions_bboxes