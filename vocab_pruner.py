"""
Vocabulary pruning for TableFormer decoder.
Reduces vocabulary size from thousands to just the tags actually used.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional


def get_tableformer_keep_tokens() -> List[str]:
    """
    Get the minimal set of tokens needed for table structure prediction.
    These are the only tokens the decoder ever emits.
    """
    # Special tokens (must be first for stable IDs)
    SPECIALS = ["<pad>", "<start>", "<end>", "<unk>"]
    
    # OTSL table structure tags
    OTSL_TAGS = [
        "nl",     # newline
        "fcel",   # first cell
        "ecel",   # empty cell  
        "lcel",   # last cell (in span)
        "xcel",   # continuation cell
        "ucel",   # upper cell (vertical span)
        "ched",   # column header
        "rhed",   # row header
        "srow",   # section row
    ]
    
    # HTML tags used in conversion
    HTML_TAGS = [
        "<tr>", "</tr>",           # table row
        "<td>", "</td>",           # table data cell
        "<thead>", "</thead>",     # table header
        "<tbody>", "</tbody>",     # table body
        "rowspan", "colspan",      # span attributes
        ">",                       # closing bracket for spans
    ]
    
    return SPECIALS + OTSL_TAGS + HTML_TAGS


def build_compact_maps(word_map: dict, keep_tokens: List[str]) -> Tuple[Dict, Dict, Dict]:
    """
    Build mappings between original and compact vocabularies.
    
    Args:
        word_map: Original word map with "word_map_tag" key
        keep_tokens: List of tokens to keep (order defines new IDs)
        
    Returns:
        Tuple of (old_to_new_id_map, new_word_map_tag, new_rev_word_map)
    """
    original_wm = word_map["word_map_tag"]  # token -> old_id
    
    # Check all tokens exist
    missing = [t for t in keep_tokens if t not in original_wm]
    if missing:
        # Some tokens might not exist, that's ok - skip them
        keep_tokens = [t for t in keep_tokens if t in original_wm]
        print(f"⚠️ Skipping missing tokens: {missing}")
    
    # Build mappings
    old_ids = [original_wm[t] for t in keep_tokens]
    old_to_new = {old: new for new, old in enumerate(old_ids)}
    new_word_map_tag = {t: i for i, t in enumerate(keep_tokens)}
    new_rev_word_map = {i: t for t, i in new_word_map_tag.items()}
    
    print(f"✅ Vocabulary pruned: {len(original_wm)} → {len(keep_tokens)} tokens")
    
    return old_to_new, new_word_map_tag, new_rev_word_map


def prune_decoder_vocab(model, word_map: dict, keep_tokens: Optional[List[str]] = None, 
                        device=None) -> Tuple:
    """
    Prune the decoder vocabulary to only keep necessary tokens.
    
    Args:
        model: TableModel04_rs instance
        word_map: Original word map dictionary
        keep_tokens: Tokens to keep (uses default set if None)
        device: Target device
        
    Returns:
        Tuple of (model, pruned_word_map)
    """
    if keep_tokens is None:
        keep_tokens = get_tableformer_keep_tokens()
    
    # Build compact mappings
    old_to_new, new_word_map_tag, new_rev = build_compact_maps(word_map, keep_tokens)
    
    # Get the modules to prune
    tag_transformer = model._tag_transformer
    old_embedding = tag_transformer._embedding
    old_fc = tag_transformer._fc
    
    # Get device
    dev = device or old_embedding.weight.device
    
    # Create index tensor for slicing
    original_wm = word_map["word_map_tag"]
    old_ids = [original_wm[t] for t in keep_tokens if t in original_wm]
    old_ids_tensor = torch.tensor(old_ids, dtype=torch.long, device=dev)
    
    # Extract and create new layers
    with torch.no_grad():
        # Slice embedding rows
        new_embed_weight = old_embedding.weight.index_select(0, old_ids_tensor)
        new_vocab_size = len(old_ids)
        embed_dim = new_embed_weight.size(1)
        
        # Create new embedding layer
        # Find padding index in new vocab
        pad_idx = None
        if "<pad>" in new_word_map_tag:
            pad_idx = new_word_map_tag["<pad>"]
        
        new_embedding = nn.Embedding(
            new_vocab_size, embed_dim, 
            padding_idx=pad_idx
        )
        new_embedding.weight.copy_(new_embed_weight)
        
        # Slice FC layer (output projection)
        new_fc_weight = old_fc.weight.index_select(0, old_ids_tensor)
        new_fc_bias = None
        if old_fc.bias is not None:
            new_fc_bias = old_fc.bias.index_select(0, old_ids_tensor)
        
        # Create new FC layer
        hidden_dim = new_fc_weight.size(1)
        new_fc = nn.Linear(hidden_dim, new_vocab_size, bias=(old_fc.bias is not None))
        new_fc.weight.copy_(new_fc_weight)
        if new_fc_bias is not None:
            new_fc.bias.copy_(new_fc_bias)
    
    # Install new layers
    tag_transformer._embedding = new_embedding.to(dev)
    tag_transformer._fc = new_fc.to(dev)
    
    # Update word maps
    pruned_word_map = dict(word_map)  # shallow copy
    pruned_word_map["word_map_tag"] = new_word_map_tag
    
    # Store mappings in model for inference
    model._compact_rev_word_map = new_rev
    model._id_map_old_to_new = old_to_new
    model._pruned_vocab_size = new_vocab_size
    
    # Update any cached special token IDs
    if hasattr(model, '_init_data'):
        model._init_data["word_map"]["word_map_tag"] = new_word_map_tag
    
    print(f"🚀 Decoder vocabulary pruned: {len(word_map['word_map_tag'])} → {new_vocab_size}")
    print(f"   Embedding: {old_embedding.weight.shape} → {new_embedding.weight.shape}")
    print(f"   FC layer:  {old_fc.weight.shape} → {new_fc.weight.shape}")
    
    return model, pruned_word_map


def validate_pruned_model(model, test_sequence: List[int], old_word_map: dict) -> bool:
    """
    Validate that pruned model can handle a test sequence.
    
    Args:
        model: Pruned model
        test_sequence: Sequence of token IDs (in old vocabulary)
        old_word_map: Original word map
        
    Returns:
        True if validation passes
    """
    if not hasattr(model, '_id_map_old_to_new'):
        print("Model not pruned, skipping validation")
        return True
    
    # Map old IDs to new IDs
    old_to_new = model._id_map_old_to_new
    
    try:
        new_sequence = []
        for old_id in test_sequence:
            if old_id in old_to_new:
                new_sequence.append(old_to_new[old_id])
            else:
                # Token not in pruned vocab - this would be an error
                old_token = None
                for token, tid in old_word_map["word_map_tag"].items():
                    if tid == old_id:
                        old_token = token
                        break
                print(f"❌ Token '{old_token}' (ID {old_id}) not in pruned vocabulary!")
                return False
        
        print(f"✅ Validation passed: {len(test_sequence)} tokens mapped successfully")
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False