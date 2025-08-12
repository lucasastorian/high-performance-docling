# layout_regression_runner.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import json, re

# ---------- Config ----------

@dataclass
class LayoutTol:
    iou_threshold: float = 0.7   # consider "same cluster" if IoU>=th
    keep_confidence: bool = False  # include confidence in serialization (usually no)

# ---------- Utils ----------

def _safe_id(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', s)

def _iou(a: Tuple[float,float,float,float], b: Tuple[float,float,float,float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = aa + bb - inter
    return inter / denom if denom > 0 else 0.0

def _bbox_tuple(bb) -> Tuple[float,float,float,float]:
    if hasattr(bb, "as_tuple"):
        return tuple(map(float, bb.as_tuple()))
    if isinstance(bb, dict):
        return float(bb["l"]), float(bb["t"]), float(bb["r"]), float(bb["b"])
    return tuple(map(float, bb))

# ---------- Canonicalization ----------

def _canon_page_layout(page) -> Dict[str, Any]:
    # expects page.predictions.layout.clusters
    clusters = getattr(getattr(page.predictions, "layout", None), "clusters", []) or []
    rows: List[Dict[str, Any]] = []
    for cl in clusters:
        label = getattr(cl, "label", None)
        # label to string (DocItemLabel or str)
        lbl = str(label.name if hasattr(label, "name") else str(label)).lower()
        l,t,r,b = _bbox_tuple(cl.bbox)
        rec = {"label": lbl, "bbox": [l, t, r, b]}
        conf = getattr(cl, "confidence", None)
        if conf is not None:
            rec["confidence"] = float(conf)
        rows.append(rec)
    # stable order: by label, then y, then x
    rows.sort(key=lambda x: (x["label"], x["bbox"][1], x["bbox"][0]))
    return {"page_no": page.page_no, "clusters": rows}

def serialize_layout_doc(doc_id: str, pages: List[Any], tol: LayoutTol) -> Dict[str, Any]:
    return {
        "doc_id": doc_id,
        "tol_iou": tol.iou_threshold,
        "pages": [_canon_page_layout(p) for p in pages]
    }

# ---------- Diff ----------

def _greedy_match(A: List[Dict[str, Any]], B: List[Dict[str, Any]], iou_th: float) -> Tuple[List[Tuple[int,int]], List[int], List[int]]:
    """
    Greedy per-label IoU matching. Returns (matches, a_unmatched, b_unmatched)
    where matches is list of (i,j) indices.
    """
    # group by label to avoid cross-type matches
    from collections import defaultdict
    idxA = defaultdict(list)
    idxB = defaultdict(list)
    for i, a in enumerate(A): idxA[a["label"]].append(i)
    for j, b in enumerate(B): idxB[b["label"]].append(j)

    matches, usedA, usedB = [], set(), set()
    for lbl in sorted(set(idxA.keys()) | set(idxB.keys())):
        LA = idxA.get(lbl, [])
        LB = idxB.get(lbl, [])
        # compute all IoUs for this label
        edges = []
        for i in LA:
            ai = tuple(A[i]["bbox"])
            for j in LB:
                bj = tuple(B[j]["bbox"])
                edges.append(( _iou(ai, bj), i, j))
        # sort by IoU desc and greedily pick
        edges.sort(reverse=True, key=lambda x: x[0])
        for iou, i, j in edges:
            if iou < iou_th: break
            if i in usedA or j in usedB: continue
            usedA.add(i); usedB.add(j); matches.append((i,j))
    a_un = [i for i in range(len(A)) if i not in usedA]
    b_un = [j for j in range(len(B)) if j not in usedB]
    return matches, a_un, b_un

def compare_layout_docs(baseline: Dict[str, Any], current: Dict[str, Any], tol: LayoutTol) -> List[str]:
    msgs: List[str] = []
    Bp = {p["page_no"]: p for p in baseline["pages"]}
    Cp = {p["page_no"]: p for p in current["pages"]}
    for page_no in sorted(set(Bp.keys()) | set(Cp.keys())):
        A = Bp.get(page_no, {"clusters":[]})["clusters"]
        C = Cp.get(page_no, {"clusters":[]})["clusters"]
        m, a_un, c_un = _greedy_match(A, C, tol.iou_threshold)
        if a_un or c_un:
            msgs.append(f"p{page_no}: clusters {len(A)} → {len(C)}, matched {len(m)} @IoU≥{tol.iou_threshold}")
        # optional: flag significant bbox shifts among matched pairs
        for (i,j) in m:
            ai, cj = A[i], C[j]
            iou = _iou(tuple(ai["bbox"]), tuple(cj["bbox"]))
            if iou < 0.9:  # warn on large movement even if above threshold
                msgs.append(f"p{page_no}: '{ai['label']}' bbox shift IoU={iou:.2f}")
    return msgs

# ---------- Runner ----------

class LayoutRegressionRunner:
    """
    Usage:
      runner = LayoutRegressionRunner()
      runner.run(url_or_id, pages, mode="baseline")  # writes baseline + prints
      runner.run(url_or_id, pages, mode="compare")   # diffs vs baseline + prints
    """
    def __init__(self, out_dir: str | Path = "./tf_regression_layout", tol: Optional[LayoutTol] = None):
        self.dir = Path(out_dir); self.dir.mkdir(parents=True, exist_ok=True)
        self.tol = tol or LayoutTol()

    def run(self, doc_id: str, pages: List[Any], mode: str = "compare") -> Dict[str, Any]:
        did = _safe_id(doc_id)
        base_path = self.dir / f"{did}.json"
        cur_path  = self.dir / f"{did}.current.json"
        payload = serialize_layout_doc(did, pages, self.tol)

        if mode == "baseline":
            with open(base_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            # also write current for convenience
            with open(cur_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"🟢 Layout baseline written: {base_path}")
            return {"status":"baseline_written", "baseline": str(base_path), "current": str(cur_path)}

        # compare
        if not base_path.exists():
            with open(cur_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"⚠️  No layout baseline at {base_path}. Wrote current: {cur_path}")
            return {"status":"no_baseline", "baseline": str(base_path), "current": str(cur_path)}

        with open(base_path, "r", encoding="utf-8") as f:
            baseline = json.load(f)

        diffs = compare_layout_docs(baseline, payload, self.tol)
        with open(cur_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        if not diffs:
            print("✅ Layout matches baseline.")
            print(f"   Baseline: {base_path}")
            return {"status":"matched", "baseline": str(base_path), "current": str(cur_path)}

        print("❌ Layout differences:")
        for d in diffs: print("   •", d)
        print(f"   Baseline: {base_path}")
        print(f"   Current : {cur_path}")
        return {"status":"diffs", "diffs": diffs, "baseline": str(base_path), "current": str(cur_path)}
