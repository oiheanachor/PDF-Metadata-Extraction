#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REV Extractor — Fixed Version (Compatible with rev_extractor_updated_v2_patched.py)
Fixes critical type mismatch bug, maintains proven scoring, adds notes attribute
"""

from __future__ import annotations
import argparse, logging, re, math, csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import fitz  # PyMuPDF
from tqdm import tqdm

LOG = logging.getLogger("rev_extractor_fixed")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ----------------------------- Patterns & Constants -----------------------------

# Enhanced to support letter+dot (E., F.)
REV_VALUE_RE = re.compile(r"^(?:[A-Z]{1,2}|\.[A-Z]{1,2}|[A-Z]{1,2}\.|\.?\d{1,3}-\d{1,3}\.?|-+|_+|\.-+|\._+)$")

def canonicalize_rev_value(v: str) -> str:
    """Canonicalise REV values while preserving meaning."""
    s = norm_val(v)
    if s in {"", "NO_REV", "NONE", "N/A"}:
        return "NO_REV"
    if re.fullmatch(r"-{1,}", s):
        return "-"
    if re.fullmatch(r"_{1,}", s):
        return "_"
    if re.fullmatch(r"\.-{1,}", s):
        return ".-"
    if re.fullmatch(r"\._{1,}", s):
        return "._"
    return s

def is_plausible_rev_value(v: str) -> bool:
    """Domain plausibility checks."""
    s = canonicalize_rev_value(v)
    if s == "NO_REV":
        return True
    if s in {"-", "_", ".-", "._"}:
        return True
    # Handle dot+letter and letter+dot
    if re.fullmatch(r"[A-Z]{1,2}", s) or re.fullmatch(r"\.[A-Z]{1,2}", s) or re.fullmatch(r"[A-Z]{1,2}\.", s):
        core = s.replace(".", "")
        if len(core) == 2:
            return core[0] in {"A", "B", "C"}
        return True
    m = re.fullmatch(r"(\d{1,3})-(\d{1,3})", s)
    if m:
        return m.group(2) == "0"
    return False

def is_suspicious_rev_value(v: str) -> bool:
    s = norm_val(v)
    if re.fullmatch(r"\d{1,4}", s):
        return True
    s2 = canonicalize_rev_value(s)
    return bool(REV_VALUE_RE.fullmatch(s2) and not is_plausible_rev_value(s2))

REV_TOKEN_RE = re.compile(r"^rev\.?$", re.IGNORECASE)

TITLE_ANCHORS = {"DWG", "DWG.", "DWGNO", "SHEET", "SCALE", "WEIGHT", "SIZE", "TITLE", "DRAWN", "CHECKED"}
REV_TABLE_HEADERS = {
    "REVISIONS", "REVISION", "DESCRIPTION", "DESCRIPTIONS",
    "EC", "DFT", "APPR", "APPD", "DATE", "CHKD", "DRAWN",
    "CHECKED", "APPROVED", "DRAWING", "CHANGE", "ECN"
}

DEFAULT_BR_X = 0.68
DEFAULT_BR_Y = 0.72
DEFAULT_EDGE_MARGIN = 0.018

# ----------------------------- Data Structures ---------------------------------

@dataclass
class Token:
    text: str
    conf: Optional[float]
    x: float
    y: float
    w: float
    h: float

@dataclass
class PageResult:
    tokens: List[Token]
    text: str
    engine: str

@dataclass
class RevHit:
    """
    Compatible with rev_extractor_updated_v2_patched.py wrapper
    """
    file: str
    page: int
    value: str
    engine: str
    score: float
    context_snippet: str
    notes: str = ""  # Added for wrapper compatibility

# ----------------------------- Utilities ---------------------------------------

def norm_val(v: Any) -> str:
    """Normalize token text."""
    if v is None:
        return ""
    s = str(v).replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = (s.replace("–", "-").replace("—", "-").replace("−", "-")
           .replace("‑", "-").replace("﹣", "-").replace("－", "-"))
    s = s.replace("‾", "_").replace("¯", "_")
    if re.fullmatch(r"\.?[A-Za-z]{1,3}\.?", s):
        s = s.upper()
    return s

def in_bottom_right(x: float, y: float, width: float, height: float) -> bool:
    return x > width * 0.55 and y > height * 0.60

def in_bottom_right_strict(x: float, y: float, width: float, height: float, brx: float, bry: float) -> bool:
    return x >= width * brx and y >= height * bry

def in_top_half(y: float, height: float) -> bool:
    return y < height * 0.5

def is_far_from_edges(x: float, y: float, width: float, height: float, edge_margin: float) -> bool:
    xm = width * edge_margin
    ym = height * edge_margin
    return (x > xm) and (x < width - xm) and (y > ym) and (y < height - ym)

def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def context_snippet_from_tokens(tokens: List[Token], center: Tuple[float, float], radius: float = 160) -> str:
    close = [t.text for t in tokens if distance((t.x, t.y), center) <= radius]
    s = " ".join(close)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:80]

# ----------------------------- Native Tokenization ------------------------------

def get_native_tokens(pdf_path: Path, page_index0: int) -> PageResult:
    tokens: List[Token] = []
    text_parts: List[str] = []
    with fitz.open(pdf_path) as doc:
        page = doc[page_index0]
        for x0, y0, x1, y1, txt, *_ in page.get_text("words"):
            txt_clean = txt.strip()
            if not txt_clean:
                continue
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            tokens.append(Token(text=txt_clean, conf=None, x=cx, y=cy, w=(x1-x0), h=(y1-y0)))
            text_parts.append(txt_clean)
    return PageResult(tokens=tokens, text=" ".join(text_parts), engine="native")

# ----------------------------- Revision Table Detection ------------------------

def is_in_revision_table(token: Token, all_tokens: List[Token], page_w: float, page_h: float) -> bool:
    """Detect if token is in revision table."""
    if in_bottom_right_strict(token.x, token.y, page_w, page_h, 0.68, 0.72):
        return False
    if not in_top_half(token.y, page_h):
        return False
    nearby = [t for t in all_tokens if distance((t.x, t.y), (token.x, token.y)) <= 400]
    table_header_count = sum(1 for t in nearby if norm_val(t.text).upper() in REV_TABLE_HEADERS)
    if table_header_count >= 3:
        return True
    if table_header_count >= 2 and token.x > page_w * 0.5:
        return True
    return False

# ----------------------------- Candidate Assembly ------------------------------

def _sort_by_x(tokens: List[Token]) -> List[Token]:
    return sorted(tokens, key=lambda t: (t.y, t.x))

def assemble_inline_candidates(neighborhood: List[Token], line_tol: float = 0.85, gap_tol: float = 0.60) -> List[str]:
    """Build candidate strings by concatenating adjacent tokens."""
    if not neighborhood:
        return []
    by_lines: List[List[Token]] = []
    toks = _sort_by_x(neighborhood)
    for t in toks:
        placed = False
        for line in by_lines:
            anchor = line[0]
            same_line = abs(t.y - anchor.y) <= max(anchor.h, t.h) * line_tol
            if same_line:
                line.append(t)
                placed = True
                break
        if not placed:
            by_lines.append([t])

    cands: set = set()
    for line in by_lines:
        line = sorted(line, key=lambda t: t.x)
        if not line:
            continue
        avg_h = sum(t.h for t in line) / len(line)
        max_gap = avg_h * gap_tol
        texts = [norm_val(t.text) for t in line]
        xs = [t.x for t in line]
        for i in range(len(line)-1):
            if abs(xs[i+1] - xs[i]) <= max_gap:
                cands.add(texts[i] + texts[i+1])
        for i in range(len(line)-2):
            if abs(xs[i+1] - xs[i]) <= max_gap and abs(xs[i+2] - xs[i+1]) <= max_gap:
                cands.add(texts[i] + texts[i+1] + texts[i+2])
    return list(cands)

# ----------------------------- Scoring ------------------------------------------

def _nearby_anchor_bonus(tokens_in_zone: List[Token], center_xy: Tuple[float, float], radius=220) -> int:
    return sum(1 for a in tokens_in_zone
               if norm_val(a.text).upper() in TITLE_ANCHORS and distance((a.x, a.y), center_xy) <= radius)

def score_candidates_bottom_right_first(
    tokens: List[Token], page_w: float, page_h: float,
    brx: float, bry: float, blocklist: Optional[set] = None,
    edge_margin: float = DEFAULT_EDGE_MARGIN
) -> Optional[Tuple[str, float, Tuple[float, float], str]]:
    """
    Returns (value, score, center, context) or None
    """
    block = {t.upper() for t in (blocklist or set())}

    br_tokens = [
        t for t in tokens
        if in_bottom_right_strict(t.x, t.y, page_w, page_h, brx, bry)
        and is_far_from_edges(t.x, t.y, page_w, page_h, edge_margin)
    ]
    if not br_tokens:
        return None

    br_rev_labels = [t for t in br_tokens if REV_TOKEN_RE.match(norm_val(t.text))]

    # Pattern detection
    def is_hyphen_code(s: str) -> bool:
        return bool(re.fullmatch(r"\d{1,2}-\d{1,2}", s))
    def is_double_letter(s: str) -> bool:
        return bool(re.fullmatch(r"[A-Z]{2}", s))
    def is_single_letter(s: str) -> bool:
        return bool(re.fullmatch(r"[A-Z]", s))
    def is_dot_letter(s: str) -> bool:
        return bool(re.fullmatch(r"\.[A-Z]{1,2}", s) or re.fullmatch(r"[A-Z]{1,2}\.", s))

    def base_score_for(v: str) -> float:
        if is_dot_letter(v):      return 25.0  # Boost for dot+letter
        if is_hyphen_code(v):     return 40.0
        if is_double_letter(v):   return 14.0
        if is_single_letter(v):   return 4.0
        return 8.0

    def neighborhood_around(cx: float, cy: float, radius: float = 300.0) -> List[Token]:
        return [t for t in br_tokens if distance((t.x, t.y), (cx, cy)) <= radius]

    cands: List[Tuple[float, str, Tuple[float,float]]] = []

    def consider_token_or_assembled(ref_xy: Tuple[float,float], neigh: List[Token], label_token: Optional[Token]):
        for t in neigh:
            v = norm_val(t.text)
            if not REV_VALUE_RE.match(v):
                continue
            if v.upper() in block:
                continue
            if is_in_revision_table(t, tokens, page_w, page_h):
                continue
            
            d = distance((t.x, t.y), ref_xy) + 1e-3
            score = base_score_for(v) + 1000.0 / d
            
            if label_token is not None:
                if abs(t.y - label_token.y) <= max(label_token.h, t.h) * 0.8:
                    score += 6.0
                if t.x > label_token.x:
                    score += 8.0
            
            if in_bottom_right(t.x, t.y, page_w, page_h): 
                score += 3.0
            
            score += _nearby_anchor_bonus(br_tokens, (t.x, t.y)) * 1.2
            
            if is_dot_letter(v):
                LOG.debug(f"  Dot+letter: {v} (score: {score:.1f})")
            
            cands.append((score, v, (t.x, t.y)))

        assembled = assemble_inline_candidates(neigh, line_tol=0.85, gap_tol=0.60)
        for s in assembled:
            s_norm = norm_val(s)
            if not REV_VALUE_RE.match(s_norm):
                continue
            if s_norm.upper() in block:
                continue
            score = base_score_for(s_norm) + 1000.0 / 30.0
            if label_token is not None:
                score += 6.0
            cands.append((score, s_norm, ref_xy))

    if br_rev_labels:
        for r in br_rev_labels:
            neigh = neighborhood_around(r.x, r.y, radius=300.0)
            consider_token_or_assembled((r.x, r.y), neigh, r)
    else:
        anchor_xy = (page_w * 0.92, page_h * 0.90)
        neigh = neighborhood_around(anchor_xy[0], anchor_xy[1], radius=320.0)
        consider_token_or_assembled(anchor_xy, neigh, None)

    if not cands:
        for t in br_tokens:
            if norm_val(t.text).upper() == "OF":
                return ("OF", 0.05, (t.x, t.y), context_snippet_from_tokens(tokens, (t.x, t.y)))
        return None

    any_hyphen = any(re.fullmatch(r"\d{1,2}-\d{1,2}", v) for _, v, _ in cands)
    if any_hyphen:
        cands = [(s - (6.0 if re.fullmatch(r"[A-Z]", v) else 0.0), v, xy) for (s, v, xy) in cands]

    best = max(cands, key=lambda c: c[0])
    score, v, center = best
    ctx = context_snippet_from_tokens(tokens, center, radius=160)
    
    if is_dot_letter(v):
        LOG.info(f"  ✓ Extracted dot+letter: {v}")
    
    return (v, score, center, ctx)

def score_candidates_global(tokens: List[Token], page_w: float, page_h: float) -> Optional[Tuple[str, float, Tuple[float, float], str]]:
    """Global fallback. Returns tuple or None."""
    anchor_tokens = [t for t in tokens if norm_val(t.text).upper() in TITLE_ANCHORS]
    rev_tokens = [t for t in tokens if REV_TOKEN_RE.match(norm_val(t.text))]
    if not rev_tokens:
        return None

    def nearby_anchor_bonus(center_xy, radius=220):
        return sum(1 for a in anchor_tokens if distance((a.x, a.y), center_xy) <= radius)

    cands = []
    for r in rev_tokens:
        r_word = norm_val(r.text).lower()
        is_revision_word = r_word.startswith("revision")
        neighborhood = [t for t in tokens if distance((t.x, t.y), (r.x, r.y)) <= 280]
        
        nearby = [t for t in tokens if distance((t.x, t.y), (r.x, r.y)) <= 350]
        table_header_count = sum(1 for t in nearby if norm_val(t.text).upper() in REV_TABLE_HEADERS)
        is_likely_revision_table = table_header_count >= 2
        
        for t in neighborhood:
            v = norm_val(t.text)
            if not REV_VALUE_RE.match(v):
                continue
            if is_in_revision_table(t, tokens, page_w, page_h):
                continue
            
            d = distance((t.x, t.y), (r.x, r.y)) + 1e-3
            same_line = abs(t.y - r.y) <= max(r.h, t.h) * 0.8
            to_right = t.x > r.x
            
            score = 1000.0 / d
            if same_line:
                score += 8.0
            if to_right:
                score += 6.0
            score += nearby_anchor_bonus((t.x, t.y)) * 1.5
            
            if is_likely_revision_table or is_revision_word:
                score *= 0.1
            if in_bottom_right(t.x, t.y, page_w, page_h):
                score += 4.0
            
            cands.append((score, v, (t.x, t.y)))

    if not cands:
        return None

    best = max(cands, key=lambda c: c[0])
    score, v, center = best
    ctx = context_snippet_from_tokens(tokens, center, radius=160)
    return (v, score, center, ctx)

# ----------------------------- Page Analysis ------------------------------------

def analyze_page_native(
    pdf_path: Path, page_index0: int, brx: float, bry: float, blocklist: set, edge_margin: float
) -> Optional[Tuple[str, str, float, str]]:
    """Returns (engine, value, score, context) or None"""
    native = get_native_tokens(pdf_path, page_index0)
    with fitz.open(pdf_path) as doc:
        pw, ph = doc[page_index0].rect.width, doc[page_index0].rect.height

    # Try all corners
    if native.tokens:
        best_suspicious = None
        for corner in ("br", "bl", "tr", "tl"):
            # Determine ROI
            if corner == "br":
                roi_tokens = [t for t in native.tokens if in_bottom_right_strict(t.x, t.y, pw, ph, brx, bry)]
            elif corner == "bl":
                left_w = pw * (1.0 - brx)
                roi_tokens = [t for t in native.tokens if t.x <= left_w and t.y >= ph * bry]
            elif corner == "tr":
                top_h = ph * (1.0 - bry)
                roi_tokens = [t for t in native.tokens if t.x >= pw * brx and t.y <= top_h]
            else:  # tl
                left_w = pw * (1.0 - brx)
                top_h = ph * (1.0 - bry)
                roi_tokens = [t for t in native.tokens if t.x <= left_w and t.y <= top_h]
            
            if not roi_tokens:
                continue
            
            res = score_candidates_bottom_right_first(roi_tokens, pw, ph, 0.0, 0.0, blocklist, edge_margin)
            if not res:
                continue
            
            v, score, _, ctx = res
            if not is_suspicious_rev_value(v) and is_plausible_rev_value(v):
                return (f"native_{corner}", canonicalize_rev_value(v), score, ctx)
            if best_suspicious is None or score > best_suspicious[2]:
                best_suspicious = (f"native_{corner}", canonicalize_rev_value(v), score, ctx)
        
        if best_suspicious:
            return best_suspicious

    # Global fallback
    if native.tokens:
        res = score_candidates_global(native.tokens, pw, ph)
        if res:
            v, score, _, ctx = res
            return ("native_global", canonicalize_rev_value(v), score, ctx)

    return None

# ----------------------------- File Processing ----------------------------------

def process_pdf_native(pdf_path: Path, brx: float, bry: float, blocklist: set, edge_margin: float) -> Optional[RevHit]:
    """
    Process all pages, return best REV hit.
    Compatible with rev_extractor_updated_v2_patched.py wrapper.
    """
    hits: Dict[int, RevHit] = {}
    with fitz.open(pdf_path) as d:
        n = len(d)
    
    for i in range(n):
        res = analyze_page_native(pdf_path, i, brx, bry, blocklist, edge_margin)
        if not res:
            continue
        engine, value, score, ctx = res
        page_no = i + 1
        prev = hits.get(page_no)
        if not prev or score > prev.score:
            hits[page_no] = RevHit(
                file=pdf_path.name,
                page=page_no,
                value=value,
                engine=engine,
                score=score,
                context_snippet=ctx,
                notes=""  # Empty notes for native extraction
            )
    
    if not hits:
        return None
    
    best = max(hits.values(), key=lambda h: h.score)
    return best

# ----------------------------- Main ---------------------------------------------

def main():
    """Standalone CLI - for testing without wrapper."""
    p = argparse.ArgumentParser(description="REV Extractor - Fixed & Compatible")
    p.add_argument("input_folder", type=Path)
    p.add_argument("-o", "--output", type=Path, default=Path("rev_results_fixed.csv"))
    p.add_argument("--brx", type=float, default=DEFAULT_BR_X)
    p.add_argument("--bry", type=float, default=DEFAULT_BR_Y)
    p.add_argument("--edge-margin", type=float, default=DEFAULT_EDGE_MARGIN)
    args = p.parse_args()

    pdfs = sorted(args.input_folder.glob("*.pdf"))
    if not pdfs:
        LOG.warning(f"No PDFs in {args.input_folder}")
        return

    LOG.info(f"Processing {len(pdfs)} PDFs")

    blocklist = {"EC", "DF", "DT", "AP", "ID", "NO", "IN", "ON", "BY"}
    rows = []
    stats = {"native": 0, "failed": 0, "dot_letter": 0}

    for pdf_path in tqdm(pdfs, desc="Processing"):
        try:
            result = process_pdf_native(pdf_path, args.brx, args.bry, blocklist, args.edge_margin)
            
            if result and result.value != "NO_REV":
                stats["native"] += 1
                
                if re.fullmatch(r"\.[A-Z]{1,2}", result.value) or re.fullmatch(r"[A-Z]{1,2}\.", result.value):
                    stats["dot_letter"] += 1
                
                rows.append({
                    "file": result.file,
                    "value": result.value,
                    "engine": result.engine,
                    "score": f"{result.score:.2f}",
                    "context": result.context_snippet
                })
            else:
                stats["failed"] += 1
                rows.append({
                    "file": pdf_path.name,
                    "value": "NO_REV",
                    "engine": "failed",
                    "score": "0",
                    "context": ""
                })

        except Exception as e:
            LOG.error(f"Failed {pdf_path.name}: {e}")
            stats["failed"] += 1
            rows.append({
                "file": pdf_path.name,
                "value": "ERROR",
                "engine": "error",
                "score": "0",
                "context": str(e)[:50]
            })

    # Write CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['file', 'value', 'engine', 'score', 'context'])
        writer.writeheader()
        writer.writerows(rows)

    LOG.info(f"\n{'='*60}")
    LOG.info(f"Results: {args.output}")
    LOG.info(f"Total: {len(rows)} | Native Success: {stats['native']} | Failed: {stats['failed']}")
    LOG.info(f"Dot+letter formats: {stats['dot_letter']}")
    LOG.info(f"Success Rate: {stats['native']/len(pdfs)*100:.1f}%")
    LOG.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()
