#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REV Extractor — Fixed Version with Stronger Revision Table Filtering
Fixes issue where revision tables (top-right) beat title block (bottom-right)
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

REV_VALUE_RE = re.compile(r"^(?:[A-Z]{1,2}|\.[A-Z]{1,2}|\d{1,3}-\d{1,3}|-+|_+|\.-+|\._+)$")


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

    if re.fullmatch(r"[A-Z]{1,2}", s) or re.fullmatch(r"\.[A-Z]{1,2}", s):
        core = s[1:] if s.startswith(".") else s
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

# Revision table headers (usually top-right) - EXPANDED LIST
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
    """Normalize token text for comparisons.

    - Collapse whitespace
    - Normalise dash variants to '-'
    - Keep '.', '-', '_' because they are valid REV values
    - Uppercase alpha tokens
    """
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

    if re.fullmatch(r"\.?[A-Za-z]{1,3}", s):
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


def score_candidates_corner_first(
    tokens: List[NativeToken],
    page_width: float,
    page_height: float,
    brx: float,
    bry: float,
    max_dx: float = 220.0,
    max_dy: float = 120.0,
    allow_global_fallback: bool = True,
    corner: str = "br",
) -> Optional[ExtractResult]:
    """Like score_candidates_bottom_right_first, but for any corner ROI.

    corner: 'br' | 'bl' | 'tr' | 'tl'
    """
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

    roi_tokens = [t for t in tokens if roi_fn(t["x"], t["y"], page_width, page_height, brx, bry)]

    # Reuse the proven bottom-right scorer on the filtered token set.
    # Because the scorer itself is anchor-based (REV label proximity), it works for any ROI.
    res = score_candidates_bottom_right_first(
        roi_tokens,
        page_width,
        page_height,
        brx=0.0,  # already filtered, keep everything
        bry=0.0,
        max_dx=max_dx,
        max_dy=max_dy,
        allow_global_fallback=allow_global_fallback,
    )
    if res:
        res = dict(res)
        res["notes"] = (res.get("notes","") + f" corner={corner}").strip()
    return res


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
    Revision tables typically:
    - Are in top half or top-right of page
    - Have multiple revision table headers nearby (REVISIONS, DESCRIPTION, DATE, EC, etc.)
    - Have a columnar structure
    """
    # Quick check: if in bottom-right strict ROI, not in revision table
    if in_bottom_right_strict(token.x, token.y, page_w, page_h, 0.68, 0.72):
        return False
    
    # Check if in top half (most revision tables)
    if not in_top_half(token.y, page_h):
        return False
    
    # Count revision table headers nearby (large radius to catch table)
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

    cands: set[str] = set()
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

# ----------------------------- Scoring with Revision Table Filtering -----------

def _nearby_anchor_bonus(tokens_in_zone: List[Token], center_xy: Tuple[float, float], radius=220) -> int:
    return sum(1 for a in tokens_in_zone
               if norm_val(a.text).upper() in TITLE_ANCHORS and distance((a.x, a.y), center_xy) <= radius)

def score_candidates_bottom_right_first(
    tokens: List[Token], page_w: float, page_h: float,
    brx: float, bry: float, blocklist: Optional[set] = None,
    edge_margin: float = DEFAULT_EDGE_MARGIN
):
    """
    PASS A: Strict bottom-right ROI with STRONG revision table filtering.
    Returns (value, score, center, context) or None.
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

    def base_score_for(v: str) -> float:
        if is_hyphen_code(v):   return 40.0
        if is_double_letter(v): return 14.0
        if is_single_letter(v): return 4.0
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
            
            # CRITICAL FIX: Check if token is in revision table - if so, SKIP IT ENTIRELY
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
                return ("OF", 0.05, center, ctx)
        return None

    # Demote single letters if hyphen codes exist
    any_hyphen = any(re.fullmatch(r"\d{1,2}-\d{1,2}", v) for _, v, _ in cands)
    if any_hyphen:
        cands = [(s - (6.0 if re.fullmatch(r"[A-Z]", v) else 0.0), v, xy) for (s, v, xy) in cands]

    best = max(cands, key=lambda c: c[0])
    score, v, center = best
    ctx = context_snippet_from_tokens(tokens, center, radius=160)
    return (v, score, center, ctx)

def score_candidates_global(tokens: List[Token], page_w: float, page_h: float):
    """
    PASS B: Global fallback with STRONG revision table down-weighting.
    """
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
        
        # CRITICAL FIX: Count table headers more aggressively
        table_header_count = count_revision_table_headers_nearby((r.x, r.y), tokens, radius=350)
        is_likely_revision_table = table_header_count >= 2
        
        for t in neighborhood:
            v = norm_val(t.text)
            if not REV_VALUE_RE.match(v):
                continue
            
            # CRITICAL FIX: Skip if in revision table
            if is_in_revision_table(t, tokens, page_w, page_h):
                LOG.debug(f"Skipping '{v}' in global pass - detected in revision table")
                continue
            
            d = distance((t.x, t.y), (r.x, r.y)) + 1e-3
            same_line = abs(t.y - r.y) <= max(r.h, t.h) * 0.8
            to_right = t.x > r.x
            base = 1000.0 / d
            
            if same_line: base += 4.0
            if to_right:  base += 6.0
            if in_bottom_right(t.x, t.y, page_w, page_h): base += 5.0
            base += nearby_anchor_bonus((t.x, t.y)) * 1.5
            if t.conf is not None: base += (t.conf - 0.5) * 2.0
            
            # STRONGER down-weighting
            if is_revision_word: base -= 4.0  # Increased from -2.0
            if is_likely_revision_table: base -= 20.0  # Dramatically increased from -6.0
            
            cands.append((base, v, (t.x, t.y)))

    if not cands:
        return None

    # Strongly prefer bottom-right
    br_cands = [c for c in cands if in_bottom_right(c[2][0], c[2][1], page_w, page_h)]
    pool = br_cands if br_cands else cands
    
    score, v, center = max(pool, key=lambda c: c[0])
    ctx = context_snippet_from_tokens(tokens, center, radius=160)
    return (v, score, center, ctx)

# ----------------------------- Page Analyzers ----------------------------------

def analyze_page_native(
    pdf_path: Path, page_index0: int, brx: float, bry: float, blocklist: set, edge_margin: float
) -> Optional[Tuple[str, str, float, str]]:
    """Returns (engine, value, score, context) or None"""
    native = get_native_tokens(pdf_path, page_index0)
    with fitz.open(pdf_path) as doc:
        pw, ph = doc[page_index0].rect.width, doc[page_index0].rect.height

    # Pass A: Strict corner ROIs (handles rotated drawings where title block is not bottom-right)
    if native.tokens:
        best_suspicious = None
        for corner in ("br", "bl", "tr", "tl"):
            res = score_candidates_corner_first(native.tokens, pw, ph, brx, bry, allow_global_fallback=False, corner=corner)
            if not res:
                continue
            v, score, _, ctx = res
            if not is_suspicious_rev_value(v) and is_plausible_rev_value(v):
                return (f"native_{corner}", canonicalize_rev_value(v), score, ctx)
            # keep best suspicious as a last resort (GPT can override later)
            if best_suspicious is None or score > best_suspicious[2]:
                best_suspicious = (f"native_{corner}", canonicalize_rev_value(v), score, ctx)
        if best_suspicious:
            return best_suspicious


    # Pass B: Global fallback
    if native.tokens:
        res = score_candidates_global(native.tokens, pw, ph)
        if res:
            v, score, _, ctx = res
            return ("native", v, score, ctx)

    # Lightweight textual fallback
    if native.text:
        m = re.search(r"(?i)\brev(?:ision)?\b\s*[:#\-]?\s*([A-Za-z]{1,2}|\d{1,2}-\d{1,2})\b", native.text)
        if m:
            return ("native_text", norm_val(m.group(1)), 0.3, native.text[:80])

    return None

# ----------------------------- File Processing ---------------------------------

def _normalize_output_value(v: str) -> str:
    """Map 'OF' → 'EMPTY', otherwise return normalized."""
    vu = norm_val(v).upper()
    if vu == "OF":
        return "EMPTY"
    return norm_val(v)

def process_pdf_native(pdf_path: Path, brx: float, bry: float, blocklist: set, edge_margin: float) -> Optional[RevHit]:
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
            hits[page_no] = RevHit(file=pdf_path.name, page=page_no, value=value,
                                   engine=engine, score=score, context_snippet=ctx)
    if not hits:
        return None
    best = max(hits.values(), key=lambda h: getattr(h, 'score', 0))
    return best

def iter_pdfs(folder: Path) -> Iterable[Path]:
    seen = set()
    for p in folder.iterdir():
        try:
            if p.is_file() and p.suffix.lower() == ".pdf":
                rp = p.resolve()
                if rp not in seen:
                    seen.add(rp)
                    yield p
        except Exception:
            continue

def run_pipeline(input_folder: Path, output_csv: Path,
                 brx: float, bry: float, rev_2l_blocklist: set,
                 edge_margin: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    pdfs = list(iter_pdfs(input_folder))
    if not pdfs:
        LOG.warning(f"No PDFs found in {input_folder}")

    for p in tqdm(pdfs, desc="Scanning PDFs"):
        try:
            native_best = process_pdf_native(p, brx, bry, rev_2l_blocklist, edge_margin)

            if native_best:
                value = _normalize_output_value(native_best.value)
                rows.append({"file": p.name, "value": value, "engine": native_best.engine})
            else:
                rows.append({"file": p.name, "value": "", "engine": ""})

        except Exception as e:
            LOG.warning(f"Failed {p.name}: {e}")
            rows.append({"file": p.name, "value": "", "engine": ""})

    # Write CSV
    try:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, 'w', newline='', encoding='utf-8-sig') as outf:
            writer = csv.writer(outf)
            writer.writerow(['file', 'value', 'engine'])
            for r in rows:
                fs = _scalarize(r.get('file', ''))
                vs = _scalarize(r.get('value', ''))
                es = _scalarize(r.get('engine', ''))
                writer.writerow([fs, vs, es])
        LOG.info(f"Wrote CSV to {output_csv.resolve()} with {len(rows)} rows")
    except Exception as e:
        LOG.error(f"Failed to write CSV: {e}")

    return rows

def parse_args(argv=None):
    a = argparse.ArgumentParser(description="Extract REV values (fixed revision table filtering).")
    a.add_argument("input_folder", type=Path)
    a.add_argument("-o","--output", type=Path, default=Path("rev_results.csv"))
    a.add_argument("--br-x", type=float, default=DEFAULT_BR_X)
    a.add_argument("--br-y", type=float, default=DEFAULT_BR_Y)
    a.add_argument("--edge-margin", type=float, default=DEFAULT_EDGE_MARGIN)
    a.add_argument("--rev-2l-blocklist", type=str,
                   default=",".join(sorted(DEFAULT_REV_2L_BLOCKLIST)))
    return a.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    blocklist = {s.strip().upper() for s in args.rev_2l_blocklist.split(",") if s.strip()}
    return run_pipeline(
        input_folder=args.input_folder,
        output_csv=args.output,
        brx=args.br_x,
        bry=args.br_y,
        rev_2l_blocklist=blocklist,
        edge_margin=args.edge_margin
    )

if __name__ == "__main__":
    main()
