from __future__ import annotations
import json, hashlib, os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional


@dataclass
class Tolerances:
    bbox_abs: float = 1.0     # pixels tolerance
    bbox_rel: float = 0.01    # 1% relative tolerance
    iou_min: float = 0.98     # accept if IoU >= this
    text_case_insensitive: bool = False

# ---------------- Utils -----------------

def _round(x: Optional[float], nd=2) -> Optional[float]:
    return None if x is None else round(float(x), nd)

def _bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    denom = area_a + area_b - inter
    return (inter / denom) if denom > 0 else 0.0

def _bbox_close(a, b, tol: Tolerances) -> bool:
    if _bbox_iou(a, b) >= tol.iou_min:
        return True
    for v, w in zip(a, b):
        if abs(v - w) <= tol.bbox_abs:
            continue
        rel = abs(v - w) / max(1.0, abs(w))
        if rel <= tol.bbox_rel:
            continue
        return False
    return True

def _norm_text(t: Optional[str], ci: bool) -> str:
    if not t:
        return ""
    t = t.strip()
    return t.lower() if ci else t

# -------------- Canonicalization --------

def _canon_table(tbl: Dict[str, Any]) -> Dict[str, Any]:
    """Make a compact, stable dict for hashing and diff."""
    d = tbl if isinstance(tbl, dict) else tbl.model_dump()
    out = {
        "id": d.get("id"),
        "page_no": d.get("page_no"),
        "num_rows": int(d.get("num_rows", 0)),
        "num_cols": int(d.get("num_cols", 0)),
        "otsl_seq": d.get("otsl_seq", []),
        "cells": []
    }
    for c in d.get("table_cells", []):
        cd = c if isinstance(c, dict) else c.model_dump()
        bb = None
        if cd.get("bbox") is not None:
            source = cd["bbox"] if isinstance(cd["bbox"], dict) else cd["bbox"].model_dump()
            bb = {"l": _round(source["l"]), "t": _round(source["t"]), "r": _round(source["r"]), "b": _round(source["b"])}
        out["cells"].append({
            "sr": int(cd["start_row_offset_idx"]),
            "er": int(cd["end_row_offset_idx"]),
            "sc": int(cd["start_col_offset_idx"]),
            "ec": int(cd["end_col_offset_idx"]),
            "row_span": int(cd.get("row_span", cd["end_row_offset_idx"] - cd["start_row_offset_idx"])),
            "col_span": int(cd.get("col_span", cd["end_col_offset_idx"] - cd["start_col_offset_idx"])),
            "col_header": bool(cd.get("column_header", False)),
            "row_header": bool(cd.get("row_header", False)),
            "row_section": bool(cd.get("row_section", False)),
            "bbox": bb,
            "token": cd.get("bbox", {}).get("token") if isinstance(cd.get("bbox"), dict) else None,
        })
    out["cells"].sort(key=lambda z: (z["sr"], z["sc"], z["er"], z["ec"]))
    return out

def _hash_table(canon: Dict[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(str(canon["page_no"]).encode())
    h.update(str(canon["num_rows"]).encode()); h.update(str(canon["num_cols"]).encode())
    for cell in canon["cells"]:
        h.update(f'{cell["sr"]},{cell["sc"]},{cell["er"]},{cell["ec"]},{cell["row_span"]},{cell["col_span"]},{int(cell["col_header"])},{int(cell["row_header"])},{int(cell["row_section"])}'.encode())
        if cell["bbox"]:
            bb = cell["bbox"]
            h.update(f'{bb["l"]},{bb["t"]},{bb["r"]},{bb["b"]}'.encode())
        if cell.get("token"):
            h.update(cell["token"].encode(errors="ignore"))
    return h.hexdigest()[:16]

def _stable_table_id(canon: Dict[str, Any]) -> str:
    """Generate stable ID based on content, not detection order."""
    h = hashlib.sha256()
    h.update(str(canon["page_no"]).encode())
    h.update(str(canon["num_rows"]).encode())
    h.update(str(canon["num_cols"]).encode())
    # Sort cells to ensure stable ordering
    cell_sigs = []
    for cell in canon["cells"]:
        cell_sigs.append(f'{cell["sr"]},{cell["sc"]},{cell["er"]},{cell["ec"]}')
    cell_sigs.sort()
    for sig in cell_sigs:
        h.update(sig.encode())
    return h.hexdigest()[:8]

def _table_similarity(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    """Compute structural similarity between tables (0.0 to 1.0)."""
    if a["page_no"] != b["page_no"]:
        return 0.0
    if a["num_rows"] != b["num_rows"] or a["num_cols"] != b["num_cols"]:
        return 0.0
    
    # Compare cell grid structure (ignore bboxes/tokens for stability)
    a_cells = {(c["sr"], c["sc"], c["er"], c["ec"]) for c in a["cells"]}
    b_cells = {(c["sr"], c["sc"], c["er"], c["ec"]) for c in b["cells"]}
    
    if not a_cells or not b_cells:
        return 1.0 if a_cells == b_cells else 0.0
    
    # Jaccard similarity on cell indices
    intersection = len(a_cells & b_cells)
    union = len(a_cells | b_cells)
    return intersection / union if union > 0 else 0.0

def _serialize_doc(doc_id: str, pages: List[Any]) -> Dict[str, Any]:
    tables = []
    for p in pages:
        ts = getattr(p.predictions, "tablestructure", None)
        if ts and getattr(ts, "table_map", None):
            for tbl in ts.table_map.values():
                tables.append(_canon_table(tbl))
    entries = [{
        "doc_id": doc_id,
        "page_no": t["page_no"],
        "table_id": t["id"],
        "stable_id": _stable_table_id(t),
        "hash": _hash_table(t),
        "table": t
    } for t in tables]
    entries.sort(key=lambda e: (e["page_no"], str(e["table_id"])))
    return {"doc_id": doc_id, "tables": entries}

# -------------- Matching ----------------

def _match_tables(baseline_tables: List[Dict], current_tables: List[Dict]) -> Tuple[List[Tuple[Dict, Dict]], List[Dict], List[Dict]]:
    """
    Robust two-stage table matching to avoid false ADDED/REMOVED due to ID instability.
    
    Returns:
        (matched_pairs, unmatched_baseline, unmatched_current)
    """
    # Group by page
    
    b_by_page = defaultdict(list)
    c_by_page = defaultdict(list)
    
    for t in baseline_tables:
        b_by_page[t["page_no"]].append(t)
    for t in current_tables:
        c_by_page[t["page_no"]].append(t)
    
    matched_pairs = []
    unmatched_baseline = []
    unmatched_current = []
    
    # Process each page independently
    for page_no in sorted(set(b_by_page.keys()) | set(c_by_page.keys())):
        b_tables = b_by_page.get(page_no, [])
        c_tables = c_by_page.get(page_no, [])
        
        # Stage A: Exact match by stable_id (fast path)
        b_by_stable = {t.get("stable_id", f"legacy_{t['table_id']}"): t for t in b_tables}
        c_by_stable = {t.get("stable_id", f"legacy_{t['table_id']}"): t for t in c_tables}
        
        b_unused = list(b_tables)
        c_unused = list(c_tables)
        
        # Match by stable_id first
        for stable_id in set(b_by_stable.keys()) & set(c_by_stable.keys()):
            b_table = b_by_stable[stable_id]
            c_table = c_by_stable[stable_id]
            matched_pairs.append((b_table, c_table))
            if b_table in b_unused:
                b_unused.remove(b_table)
            if c_table in c_unused:
                c_unused.remove(c_table)
        
        # Stage B: Greedy matching by similarity for unmatched tables
        while b_unused and c_unused:
            best_pair = None
            best_score = 0.0
            
            for b_table in b_unused:
                for c_table in c_unused:
                    score = _table_similarity(b_table["table"], c_table["table"])
                    if score > best_score and score >= 0.98:  # High threshold for grid match
                        best_score = score
                        best_pair = (b_table, c_table)
            
            if best_pair:
                matched_pairs.append(best_pair)
                b_unused.remove(best_pair[0])
                c_unused.remove(best_pair[1])
            else:
                break  # No more good matches
        
        # Remaining tables are truly unmatched
        unmatched_baseline.extend(b_unused)
        unmatched_current.extend(c_unused)
    
    return matched_pairs, unmatched_baseline, unmatched_current

# -------------- Diffs -------------------

def _compare_tables(base: Dict[str, Any], curr: Dict[str, Any], tol: Tolerances) -> List[str]:
    msgs = []
    if base["num_rows"] != curr["num_rows"]:
        msgs.append(f'num_rows {base["num_rows"]} -> {curr["num_rows"]}')
    if base["num_cols"] != curr["num_cols"]:
        msgs.append(f'num_cols {base["num_cols"]} -> {curr["num_cols"]}')
    if base.get("otsl_seq") != curr.get("otsl_seq"):
        msgs.append("otsl_seq changed")

    B = {(c["sr"], c["sc"], c["er"], c["ec"]): c for c in base["cells"]}
    C = {(c["sr"], c["sc"], c["er"], c["ec"]): c for c in curr["cells"]}
    ks = sorted(set(B.keys()) | set(C.keys()))
    for k in ks:
        b, c = B.get(k), C.get(k)
        if b is None:
            msgs.append(f"cell added at {k}")
            continue
        if c is None:
            msgs.append(f"cell removed at {k}")
            continue
        for fld in ("row_span", "col_span", "col_header", "row_header", "row_section"):
            if b[fld] != c[fld]:
                msgs.append(f"cell {k}: {fld} {b[fld]} -> {c[fld]}")
        bb, cb = b["bbox"], c["bbox"]
        if bb and cb:
            a = (bb["l"], bb["t"], bb["r"], bb["b"])
            d = (cb["l"], cb["t"], cb["r"], cb["b"])
            if not _bbox_close(a, d, tol):
                msgs.append(f"cell {k}: bbox {a} -> {d}")
        elif (bb is None) != (cb is None):
            msgs.append(f"cell {k}: bbox presence changed")
        bt = _norm_text(b.get("token"), tol.text_case_insensitive)
        ct = _norm_text(c.get("token"), tol.text_case_insensitive)
        if bt != ct:
            msgs.append(f"cell {k}: token text changed")
    return msgs

# -------------- Public Runner -----------

class TableRegressionRunner:
    """
    Stateless-ish runner you call once per document at the *end* of a GPU pipeline.
    mode: 'baseline' to write baseline, 'compare' to diff against baseline.
    It ALWAYS prints a summary when run in either mode.
    """
    def __init__(self,
                 out_dir: str | Path = "./tf_regression",
                 mode: str = "compare",            # 'baseline' | 'compare'
                 tolerances: Optional[Tolerances] = None):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.tol = tolerances or Tolerances()

    def run(self, doc_id: str, pages: List[Any]) -> Dict[str, Any]:
        """
        Returns: {"status": "baseline_written"|"matched"|"diffs", "diffs": [...], "baseline": path, "current": path}
        Always prints a human-readable summary.
        """
        payload = _serialize_doc(doc_id, pages)
        base_path = self.dir / f"{doc_id}.json"
        curr_path = self.dir / f"{doc_id}.current.json"

        if self.mode == "baseline":
            with open(base_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"🟢 Baseline written: {base_path}")
            # also write a copy as current for convenience
            with open(curr_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            return {"status": "baseline_written", "diffs": [], "baseline": str(base_path), "current": str(curr_path)}

        # compare mode
        if not base_path.exists():
            # still write current for inspection
            with open(curr_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"⚠️  No baseline at {base_path}. Wrote current run to {curr_path}. "
                  f"Run in baseline mode once to establish goldens.")
            return {"status": "no_baseline", "diffs": [], "baseline": str(base_path), "current": str(curr_path)}

        with open(base_path, "r", encoding="utf-8") as f:
            baseline = json.load(f)

        # Robust table matching to avoid false ADDED/REMOVED from ID instability
        matched_pairs, unmatched_baseline, unmatched_current = _match_tables(
            baseline["tables"], payload["tables"]
        )
        
        diffs: List[str] = []
        
        # Compare matched pairs
        for b_entry, c_entry in matched_pairs:
            if b_entry["hash"] == c_entry["hash"]:
                continue  # Identical, skip detailed comparison
            
            # Deep compare the tables
            msgs = _compare_tables(b_entry["table"], c_entry["table"], self.tol)
            for m in msgs:
                # Use original table_id for display, but note they're matched by structure
                display_id = f"{b_entry['table_id']}->{c_entry['table_id']}" if b_entry['table_id'] != c_entry['table_id'] else b_entry['table_id']
                diffs.append(f"p{b_entry['page_no']} table {display_id}: {m}")
        
        # Report truly unmatched tables
        for b_entry in unmatched_baseline:
            diffs.append(f"p{b_entry['page_no']} table {b_entry['table_id']}: REMOVED (no structural match)")
        
        for c_entry in unmatched_current:
            diffs.append(f"p{c_entry['page_no']} table {c_entry['table_id']}: ADDED (no structural match)")

        # always write current
        with open(curr_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        if not diffs:
            print("✅ Tables match baseline.")
            print(f"   Baseline: {base_path}")
            return {"status": "matched", "diffs": [], "baseline": str(base_path), "current": str(curr_path)}

        print("❌ Differences from baseline:")
        for d in diffs:
            print(f"   • {d}")
        print(f"   Baseline: {base_path}")
        print(f"   Current : {curr_path}")
        return {"status": "diffs", "diffs": diffs, "baseline": str(base_path), "current": str(curr_path)}
