# fork/table/compat_kv_remap.py
import torch


def remap_mha_to_explicit_qkv(sd: dict, *,
                              dec_prefix="_tag_transformer._decoder.layers.",
                              num_layers: int,
                              d_model: int) -> dict:
    """
    Convert old nn.MultiheadAttention packed params to explicit q/k/v/o for the new KV decoder.

    Old keys (per layer i):
      {P}i.self_attn.in_proj_weight  [3*D, D]
      {P}i.self_attn.in_proj_bias    [3*D]
      {P}i.self_attn.out_proj.weight [D, D]
      {P}i.self_attn.out_proj.bias   [D]
      {P}i.multihead_attn.in_proj_weight  [3*D, D]
      {P}i.multihead_attn.in_proj_bias    [3*D]
      {P}i.multihead_attn.out_proj.weight [D, D]
      {P}i.multihead_attn.out_proj.bias   [D]

    New keys expected:
      {P}i.self_proj.q.weight / .bias
      {P}i.self_proj.k.weight / .bias
      {P}i.self_proj.v.weight / .bias
      {P}i.self_proj.o.weight / .bias
      {P}i.cross_q.weight / .bias
      {P}i.cross_k.weight / .bias
      {P}i.cross_v.weight / .bias
      {P}i.out.weight / .bias

    FFN (linear1/linear2) and norms keep their original names → no remap needed.
    """
    sd = dict(sd)  # shallow copy

    def _chunk_qkv(w_or_b):
        # split along dim 0 into (q, k, v)
        return torch.chunk(w_or_b, 3, dim=0)

    for i in range(num_layers):
        base = f"{dec_prefix}{i}."

        # ---- self-attn ----
        key_w = base + "self_attn.in_proj_weight"
        key_b = base + "self_attn.in_proj_bias"
        if key_w in sd:
            W = sd.pop(key_w)                 # [3D, D]
            B = sd.pop(key_b) if key_b in sd else None
            qW, kW, vW = _chunk_qkv(W)
            if B is not None:
                qB, kB, vB = _chunk_qkv(B)
            sd[base + "self_proj.q.weight"] = qW
            sd[base + "self_proj.k.weight"] = kW
            sd[base + "self_proj.v.weight"] = vW
            if B is not None:
                sd[base + "self_proj.q.bias"] = qB
                sd[base + "self_proj.k.bias"] = kB
                sd[base + "self_proj.v.bias"] = vB

        key = base + "self_attn.out_proj.weight"
        if key in sd:
            sd[base + "self_proj.o.weight"] = sd.pop(key)
        key = base + "self_attn.out_proj.bias"
        if key in sd:
            sd[base + "self_proj.o.bias"] = sd.pop(key)

        # ---- cross-attn (encoder-decoder) ----
        key_w = base + "multihead_attn.in_proj_weight"
        key_b = base + "multihead_attn.in_proj_bias"
        if key_w in sd:
            W = sd.pop(key_w)
            B = sd.pop(key_b) if key_b in sd else None
            qW, kW, vW = _chunk_qkv(W)
            if B is not None:
                qB, kB, vB = _chunk_qkv(B)
            sd[base + "cross_q.weight"] = qW
            sd[base + "cross_k.weight"] = kW
            sd[base + "cross_v.weight"] = vW
            if B is not None:
                sd[base + "cross_q.bias"] = qB
                sd[base + "cross_k.bias"] = kB
                sd[base + "cross_v.bias"] = vB

        key = base + "multihead_attn.out_proj.weight"
        if key in sd:
            sd[base + "out.weight"] = sd.pop(key)
        key = base + "multihead_attn.out_proj.bias"
        if key in sd:
            sd[base + "out.bias"] = sd.pop(key)

    # sanity: embedding/out head untouched; norms/linear1/linear2 keep names
    return sd

