#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REV Extractor — Enhanced Version with Minimal Changes
Based on proven rev_extractor_fixed_v2.py with ONLY these additions:
1. Small scoring boost for dot+letter patterns (.E, .F, etc.)
2. Better logging for dot+letter detection
3. Letter+dot pattern support (E., F., etc.)

KEEPS ALL PROVEN LOGIC INTACT - no major rewrites!
"""

from __future__ import annotations
import argparse, logging, re, math, csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import fitz  # PyMuPDF
from tqdm import tqdm

LOG = logging.getLogger("rev_extractor_enhanced")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ----------------------------- Patterns & Constants -----------------------------

# ENHANCED: Added letter+dot pattern (E., F., AA.)
REV_VALUE_RE = re.compile(r"^(?:[A-Z]{1,2}|\.[A-Z]{1,2}|[A-Z]{1,2}\.|\.+\d{1,3}-\d{1,3}|\d{1,3}-\d{1,3}\.?|-+|_+|\.-+|\._+)$")

# NEW: Pattern matchers for special formats
DOT_LETTER_RE = re.compile(r"^\.[A-Z]{1,2}$")
LETTER_DOT_RE = re.compile(r"^[A-Z]{1,2}\.$")

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
    """Domain plausibility checks used to trigger GPT retry / human review."""
    s = canonicalize_rev_value(v)

    if s == "NO_REV":
        return True

    if s in {"-", "_", ".-", "._"}:
        return True

    # ENHANCED: Handle dot+letter and letter+dot patterns
    if re.fullmatch(r"[A-Z]{1,2}", s) or re.fullmatch(r"\.[A-Z]{1,2}", s) or re.fullmatch(r"[A-Z]{1,2}\.", s):
        # Remove dots for validation
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

# Title block anchors (usually bottom-right)
TITLE_ANCHORS = {"DWG", "DWG.", "DWGNO", "SHEET", "SCALE", "WEIGHT", "SIZE", "TITLE", "DRAWN", "CHECKED"}

# Revision table headers (usually top-right)
REV_TABLE_HEADERS = {
    "REVISIONS", "REVISION", "DESCRIPTION", "DESCRIPTIONS",
    "EC", "DFT", "APPR", "APPD", "DATE", "CHKD", "DRAWN",
    "CHECKED", "APPROVED", "DRAWING", "CHANGE", "ECN"
}

# ROI defaults for bottom-right title block
DEFAULT_BR_X = 0.68
DEFAULT_BR_Y = 0.72
DEFAULT_EDGE_MARGIN = 0.018
DEFAULT_REV_2L_BLOCKLIST = {"EC", "DF", "DT", "AP", "ID", "NO", "IN", "ON", "BY"}

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
    file: str
    page: int
    value: str
    engine: str
    score: float
    context_snippet: str

# ----------------------------- Utilities ---------------------------------------

def _scalarize(v: Any):
    """Coerce any non-scalar to a plain Python scalar or string."""
    if isinstance(v, (list, tuple, set)):
        return ", ".join(map(str, v))
    if isinstance(v, dict):
        return ", ".join(f"{k}={str(vv)}" for k, vv in v.items())
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", errors="ignore")
    if isinstance(v, (str, int, float, bool)):
        return v
    return str(v)

def norm_val(v: Any) -> str:
    """Normalize token text for comparisons."""
    if v is None:
        return ""
    s = str(v).replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()

    # Dash/minus variants
    s = (s.replace("–", "-")
           .replace("—", "-")
           .replace("−", "-")
           .replace("‑", "-")
           .replace("﹣", "-")
           .replace("－", "-"))

    # Overline-like underscores
    s = s.replace("‾", "_").replace("¯", "_")

    if re.fullmatch(r"\.?[A-Za-z]{1,3}\.?", s):
        s = s.upper()
    return s


def in_bottom_right(x: float, y: float, width: float, height: float) -> bool:
    """Loose bottom-right check."""
    return x > width * 0.55 and y > height * 0.60

def in_bottom_right_strict(x: float, y: float, width: float, height: float, brx: float, bry: float) -> bool:
    """Strict bottom-right ROI check."""
    return x >= width * brx and y >= height * bry

def in_bottom_left_strict(x: float, y: float, w: float, h: float, brx: float, bry: float) -> bool:
    """Bottom-left ROI that mirrors the size of the bottom-right ROI."""
    left_w = w * (1.0 - brx)
    return x <= left_w and y >= h * bry


def in_top_right_strict(x: float, y: float, w: float, h: float, brx: float, bry: float) -> bool:
    """Top-right ROI that mirrors the size of the bottom-right ROI."""
    top_h = h * (1.0 - bry)
    return x >= w * brx and y <= top_h


def in_top_left_strict(x: float, y: float, w: float, h: float, brx: float, bry: float) -> bool:
    """Top-left ROI that mirrors the size of the bottom-right ROI."""
    left_w = w * (1.0 - brx)
    top_h = h * (1.0 - bry)
    return x <= left_w and y <= top_h


def in_top_half(y: float, height: float) -> bool:
    """Check if token is in top half of page (where revision tables usually are)."""
    return y < height * 0.5

def is_far_from_edges(x: float, y: float, width: float, height: float, edge_margin: float) -> bool:
    """Filter out tokens too close to page edges."""
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

# ----------------------------- Enhanced Revision Table Detection ---------------

def is_in_revision_table(token: Token, all_tokens: List[Token], page_w: float, page_h: float) -> bool:
    """
    Detect if a token is part of a revision table.
    """
    # Quick check: if in bottom-right strict ROI, not in revision table
    if in_bottom_right_strict(token.x, token.y, page_w, page_h, 0.68, 0.72):
        return False
    
    # Check if in top half (most revision tables)
    if not in_top_half(token.y, page_h):
        return False
    
    # Count revision table headers nearby
    nearby = [t for t in all_tokens if distance((t.x, t.y), (token.x, token.y)) <= 400]
    table_header_count = sum(1 for t in nearby if norm_val(t.text).upper() in REV_TABLE_HEADERS)
    
    # If 3+ table headers nearby, very likely a revision table
    if table_header_count >= 3:
        return True
    
    # If 2 headers and token is in top-right, likely table
    if table_header_count >= 2 and token.x > page_w * 0.5:
        return True
    
    return False

def count_revision_table_headers_nearby(center_xy: Tuple[float, float], all_tokens: List[Token], radius: float = 350) -> int:
    """Count how many revision table headers are near this location."""
    return sum(1 for t in all_tokens 
               if distance((t.x, t.y), center_xy) <= radius 
               and norm_val(t.text).upper() in REV_TABLE_HEADERS)

# ----------------------------- Candidate Assembly ------------------------------

def _sort_by_x(tokens: List[Token]) -> List[Token]:
    return sorted(tokens, key=lambda t: (t.y, t.x))

def assemble_inline_candidates(neighborhood: List[Token], line_tol: float = 0.85, gap_tol: float = 0.60) -> List[str]:
    """Build candidate strings by concatenating adjacent tokens: '1' '-' '0' -> '1-0'"""
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
                line.append(t); placed = True; break
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
        # 2-grams
        for i in range(len(line)-1):
            if abs(xs[i+1] - xs[i]) <= max_gap:
                cands.add(texts[i] + texts[i+1])
        # 3-grams
        for i in range(len(line)-2):
            if abs(xs[i+1] - xs[i]) <= max_gap and abs(xs[i+2] - xs[i+1]) <= max_gap:
                cands.add(texts[i] + texts[i+1] + texts[i+2])
    return list(cands)

# ----------------------------- Scoring with Dot+Letter Boost ------------------

def _nearby_anchor_bonus(tokens_in_zone: List[Token], center_xy: Tuple[float, float], radius=220) -> int:
    return sum(1 for a in tokens_in_zone
               if norm_val(a.text).upper() in TITLE_ANCHORS and distance((a.x, a.y), center_xy) <= radius)

def score_candidates_bottom_right_first(
    tokens: List[Token], page_w: float, page_h: float,
    brx: float, bry: float, blocklist: Optional[set] = None,
    edge_margin: float = DEFAULT_EDGE_MARGIN,
    max_dx: float = 220.0,
    max_dy: float = 120.0,
    allow_global_fallback: bool = True,
):
    """
    PASS A: Strict bottom-right ROI with revision table filtering.
    ENHANCED: Small boost for dot+letter patterns.
    """
    block = {t.upper() for t in (blocklist or set())}

    # ROI filter + edge exclusion
    br_tokens = [
        t for t in tokens
        if in_bottom_right_strict(t.x, t.y, page_w, page_h, brx, bry)
        and is_far_from_edges(t.x, t.y, page_w, page_h, edge_margin)
    ]
    if not br_tokens:
        return None

    br_rev_labels = [t for t in br_tokens if REV_TOKEN_RE.match(norm_val(t.text))]

    # Priority patterns
    def is_hyphen_code(s: str) -> bool:
        return bool(re.fullmatch(r"\d{1,2}-\d{1,2}", s))
    def is_double_letter(s: str) -> bool:
        return bool(re.fullmatch(r"[A-Z]{2}", s))
    def is_single_letter(s: str) -> bool:
        return bool(re.fullmatch(r"[A-Z]", s))
    # NEW: Dot+letter detection
    def is_dot_letter(s: str) -> bool:
        return bool(DOT_LETTER_RE.match(s) or LETTER_DOT_RE.match(s))

    def base_score_for(v: str) -> float:
        # ENHANCED: Boost dot+letter patterns slightly
        if is_dot_letter(v):      return 25.0  # Higher than double letter
        if is_hyphen_code(v):     return 40.0
        if is_double_letter(v):   return 14.0
        if is_single_letter(v):   return 4.0
        return 8.0

    def neighborhood_around(cx: float, cy: float, radius: float = 300.0) -> List[Token]:
        return [t for t in br_tokens if distance((t.x, t.y), (cx, cy)) <= radius]

    cands: List[Tuple[float, str, Tuple[float,float]]] = []

    def consider_token_or_assembled(ref_xy: Tuple[float,float], neigh: List[Token], label_token: Optional[Token]):
        # 1) Raw tokens
        for t in neigh:
            v = norm_val(t.text)
            if not REV_VALUE_RE.match(v):
                continue
            vu = v.upper()
            if vu in block:
                continue
            
            # Check if token is in revision table - if so, SKIP IT
            if is_in_revision_table(t, tokens, page_w, page_h):
                LOG.debug(f"Skipping '{v}' - detected in revision table")
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
            
            # NEW: Log dot+letter detection
            if is_dot_letter(v):
                LOG.debug(f"  Found dot+letter pattern: {v} (score: {score:.1f})")
            
            cands.append((score, v, (t.x, t.y)))

        # 2) Assembled n-grams
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
        # Approximate typical REV cell centroid
        anchor_xy = (page_w * 0.92, page_h * 0.90)
        neigh = neighborhood_around(anchor_xy[0], anchor_xy[1], radius=320.0)
        consider_token_or_assembled(anchor_xy, neigh, None)

    if not cands:
        # Last resort: check for 'OF' sentinel
        for t in br_tokens:
            if norm_val(t.text).upper() == "OF":
                center = (t.x, t.y)
                ctx = context_snippet_from_tokens(tokens, center, radius=160)
                return {"value": "OF", "score": 0.05, "center": center, "context": ctx, "notes": ""}
        return None

    # Demote single letters if hyphen codes exist
    any_hyphen = any(re.fullmatch(r"\d{1,2}-\d{1,2}", v) for _, v, _ in cands)
    if any_hyphen:
        cands = [(s - (6.0 if re.fullmatch(r"[A-Z]", v) else 0.0), v, xy) for (s, v, xy) in cands]

    best = max(cands, key=lambda c: c[0])
    score, v, center = best
    ctx = context_snippet_from_tokens(tokens, center, radius=160)
    
    # NEW: Log if we found a dot+letter value
    if is_dot_letter(v):
        LOG.info(f"  ✓ Extracted dot+letter REV: {v}")
    
    return {"value": v, "score": score, "center": center, "context": ctx, "notes": ""}

def score_candidates_corner_first(
    tokens: List[Token],
    page_width: float,
    page_height: float,
    brx: float,
    bry: float,
    max_dx: float = 220.0,
    max_dy: float = 120.0,
    allow_global_fallback: bool = True,
    corner: str = "br",
) -> Optional[Dict]:
    """Score candidates from specific corner."""
    corner = corner.lower()
    if corner == "br":
        roi_fn = in_bottom_right_strict
    elif corner == "bl":
        roi_fn = in_bottom_left_strict
    elif corner == "tr":
        roi_fn = in_top_right_strict
    elif corner == "tl":
        roi_fn = in_top_left_strict
    else:
        roi_fn = in_bottom_right_strict

    roi_tokens = [t for t in tokens if roi_fn(t.x, t.y, page_width, page_height, brx, bry)]

    res = score_candidates_bottom_right_first(
        roi_tokens,
        page_width,
        page_height,
        brx=0.0,
        bry=0.0,
        max_dx=max_dx,
        max_dy=max_dy,
        allow_global_fallback=allow_global_fallback,
    )
    if res:
        res = dict(res)
        res["notes"] = (res.get("notes","") + f" corner={corner}").strip()
    return res


def score_candidates_global(tokens: List[Token], page_w: float, page_h: float):
    """PASS B: Global fallback with revision table down-weighting."""
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
        
        table_header_count = count_revision_table_headers_nearby((r.x, r.y), tokens, radius=350)
        is_likely_revision_table = table_header_count >= 2
        
        for t in neighborhood:
            v = norm_val(t.text)
            if not REV_VALUE_RE.match(v):
                continue
            
            if is_in_revision_table(t, tokens, page_w, page_h):
                LOG.debug(f"Skipping '{v}' in global pass - detected in revision table")
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
                score *= 0.1  # Massive penalty
            
            if in_bottom_right(t.x, t.y, page_w, page_h):
                score += 4.0
            
            cands.append((score, v, (t.x, t.y)))

    if not cands:
        return None

    best = max(cands, key=lambda c: c[0])
    score, v, center = best
    ctx = context_snippet_from_tokens(tokens, center, radius=160)
    return {"value": v, "score": score, "center": center, "context": ctx, "notes": "global"}


# ----------------------------- Main Processing ----------------------------------

def process_pdf_native(
    pdf_path: Path,
    *,
    page: int = 0,
    brx: float = DEFAULT_BR_X,
    bry: float = DEFAULT_BR_Y,
    corner_order: List[str] = ["br", "bl", "tr", "tl"],
    blocklist: Optional[set] = None,
) -> Optional[RevHit]:
    """Extract REV using native PyMuPDF, trying multiple corners."""
    try:
        pr = get_native_tokens(pdf_path, page)
        tokens = pr.tokens
        if not tokens:
            return None

        with fitz.open(pdf_path) as doc:
            p = doc[page]
            pw, ph = p.rect.width, p.rect.height

        # Try corners in order
        for corner in corner_order:
            res = score_candidates_corner_first(
                tokens, pw, ph, brx, bry,
                allow_global_fallback=False,
                corner=corner
            )
            if res and res["value"]:
                val = canonicalize_rev_value(res["value"])
                if val != "NO_REV":
                    return RevHit(
                        file=pdf_path.name,
                        page=page,
                        value=val,
                        engine=f"native_{corner}",
                        score=res["score"],
                        context_snippet=res["context"]
                    )

        # Global fallback
        res_g = score_candidates_global(tokens, pw, ph)
        if res_g and res_g["value"]:
            val = canonicalize_rev_value(res_g["value"])
            if val != "NO_REV":
                return RevHit(
                    file=pdf_path.name,
                    page=page,
                    value=val,
                    engine="native_global",
                    score=res_g["score"],
                    context_snippet=res_g["context"]
                )

        return None

    except Exception as e:
        LOG.debug(f"Native extraction failed for {pdf_path.name}: {e}")
        return None


# ----------------------------- CLI ----------------------------------------------

def main():
    p = argparse.ArgumentParser(description="REV Extractor - Enhanced with minimal changes")
    p.add_argument("input_folder", type=Path)
    p.add_argument("-o", "--output", type=Path, default=Path("rev_results_enhanced.csv"))
    args = p.parse_args()

    pdfs = sorted(args.input_folder.glob("*.pdf"))
    if not pdfs:
        LOG.warning(f"No PDFs in {args.input_folder}")
        return

    LOG.info(f"Processing {len(pdfs)} PDFs")

    rows = []
    stats = {"native": 0, "failed": 0, "dot_letter": 0}

    for pdf_path in tqdm(pdfs, desc="Processing"):
        try:
            result = process_pdf_native(pdf_path)
            
            if result:
                stats["native"] += 1
                
                # Track dot+letter formats
                if DOT_LETTER_RE.match(result.value) or LETTER_DOT_RE.match(result.value):
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

    # Write CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['file', 'value', 'engine', 'score', 'context'])
        writer.writeheader()
        writer.writerows(rows)

    LOG.info(f"\n{'='*60}")
    LOG.info(f"Results: {args.output}")
    LOG.info(f"Total: {len(rows)} | Native: {stats['native']} | Failed: {stats['failed']}")
    LOG.info(f"Dot+letter formats: {stats['dot_letter']}")
    LOG.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()
