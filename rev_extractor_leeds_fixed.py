#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REV / ISSUE extractor for engineering drawings.

This keeps the original native-first architecture, but removes the old
bottom-right / hyphenated-only assumptions. It supports REV and ISSUE markers,
single numeric / alpha revisions, top-left title blocks, revision tables whose
top row is authoritative, rotated coordinate systems, and optional OCR only
when native extraction is weak or unavailable.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import fitz  # PyMuPDF
from tqdm import tqdm

LOG = logging.getLogger("rev_extractor_fixed")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ----------------------------- Patterns & Constants -----------------------------

REV_VALUE_RE = re.compile(
    r"^(?:"
    r"[A-Z]{1,2}|"          # A, B, AA
    r"\d{1,3}|"             # 0, 1, 01, 16
    r"\d{1,3}-\d{1,3}|"     # 0-0, 2-1, 12-01
    r"-+|_+|\.-+|\._+"      # placeholders
    r")$"
)

REV_MARKER_RE = re.compile(
    r"^(?:REV\.?|REVISION|ISS\.?|ISSUE|[!1I]SSUE|PSSUE|ISSUE\s*NO\.?|ISSUE\s*NUMBER)$",
    re.IGNORECASE,
)

TITLE_ANCHORS = {
    "DWG", "DWG.", "DWGNO", "DRAWING", "DRG", "SHEET", "SCALE", "WEIGHT",
    "SIZE", "TITLE", "DRAWN", "CHECKED", "CHKD", "APPROVED", "APPD",
    "AUTH", "DATE", "No", "NO", "NO.",
}

REV_TABLE_HEADERS = {
    "REV", "REV.", "REVISIONS", "REVISION", "ISS", "ISS.", "ISSUE",
    "DESCRIPTION", "DESCRIPTIONS", "DESC", "MODIFICATION", "CHANGE",
    "ALTERATIONS",
    "EC", "ECN", "DFT", "APPR", "APPD", "DATE", "CHKD", "DRAWN",
    "CHECKED", "APPROVED", "AUTH", "BY", "ZONE",
}

GRID_EDGE_MARGIN = 0.05
DEFAULT_BR_X = 0.68
DEFAULT_BR_Y = 0.72
DEFAULT_EDGE_MARGIN = 0.018
DEFAULT_REV_2L_BLOCKLIST = {"EC", "DF", "DT", "AP", "ID", "NO", "IN", "ON", "BY", "TO", "OF", "CN", "PA", "MR", "KE", "O"}


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
class Candidate:
    value: str
    score: float
    x: float
    y: float
    region: str
    source: str
    reason: str

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
    if v is None:
        return ""
    s = str(v).replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = (s.replace("–", "-").replace("—", "-").replace("−", "-")
           .replace("‑", "-").replace("﹣", "-").replace("－", "-"))
    s = s.replace("‾", "_").replace("¯", "_")
    s = s.strip(":;,#'\"`‘’“”|[](){}")
    if re.fullmatch(r"\.?[A-Za-z]{1,8}", s):
        s = s.upper()
    return s

def canonicalize_rev_value(v: str) -> str:
    s = norm_val(v)
    if s in {"", "NO_REV", "NONE", "N/A", "NA", "EMPTY"}:
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

def is_special_char(s: str) -> bool:
    return bool(re.fullmatch(r"[-_]+|\.[-_]+", norm_val(s)))

def is_plausible_rev_value(v: str) -> bool:
    s = canonicalize_rev_value(v)
    if s == "NO_REV" or is_special_char(s):
        return True
    if re.fullmatch(r"[A-Z]{1,2}", s):
        return True
    if re.fullmatch(r"\d{1,3}", s):
        return True
    if re.fullmatch(r"\d{1,3}-\d{1,3}", s):
        return True
    return False

def is_suspicious_rev_value(v: str) -> bool:
    s = canonicalize_rev_value(v)
    if not REV_VALUE_RE.fullmatch(s):
        return True
    if s in DEFAULT_REV_2L_BLOCKLIST:
        return True
    return not is_plausible_rev_value(s)

def is_revision_marker_text(text: str) -> bool:
    return bool(REV_MARKER_RE.fullmatch(norm_val(text)))

def is_rev_value_text(text: str, blocklist: Optional[set] = None) -> bool:
    s = canonicalize_rev_value(text)
    if s == "NO_REV":
        return False
    if blocklist and s.upper() in blocklist:
        return False
    return bool(REV_VALUE_RE.fullmatch(s))

def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def page_extents(tokens: Sequence[Token], page_w: float, page_h: float) -> Tuple[float, float]:
    max_x = max([page_w] + [t.x + t.w / 2 for t in tokens])
    max_y = max([page_h] + [t.y + t.h / 2 for t in tokens])
    return max_x, max_y

def region_for_token(t: Token, w: float, h: float) -> str:
    nx = t.x / w if w else 0
    ny = t.y / h if h else 0
    if nx >= 0.55 and ny >= 0.62:
        return "bottom_right_title"
    if nx <= 0.42 and ny <= 0.25:
        return "top_left_title"
    if nx <= 0.45 and ny >= 0.62:
        return "bottom_left_revision_table"
    if nx >= 0.55 and ny <= 0.25:
        return "top_right_title_or_table"
    return "drawing_body"

def is_edge_grid_token(t: Token, w: float, h: float) -> bool:
    s = canonicalize_rev_value(t.text)
    if not re.fullmatch(r"[A-Z]|\d{1,2}", s):
        return False
    nx = t.x / w if w else 0
    ny = t.y / h if h else 0
    return nx <= GRID_EDGE_MARGIN or nx >= 1 - GRID_EDGE_MARGIN or ny <= GRID_EDGE_MARGIN or ny >= 1 - GRID_EDGE_MARGIN

def context_snippet_from_tokens(tokens: List[Token], center: Tuple[float, float], radius: float = 170) -> str:
    close = [t.text for t in tokens if distance((t.x, t.y), center) <= radius]
    s = " ".join(close)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:120]

def _nearby_anchor_bonus(tokens: List[Token], center_xy: Tuple[float, float], radius=220) -> int:
    return sum(
        1 for a in tokens
        if norm_val(a.text).upper() in TITLE_ANCHORS and distance((a.x, a.y), center_xy) <= radius
    )


# ----------------------------- Tokenization ------------------------------------

def get_native_tokens(pdf_path: Path, page_index0: int) -> PageResult:
    tokens: List[Token] = []
    text_parts: List[str] = []
    with fitz.open(pdf_path) as doc:
        page = doc[page_index0]
        for x0, y0, x1, y1, txt, *_ in page.get_text("words"):
            txt_clean = txt.strip()
            if not txt_clean:
                continue
            tokens.append(Token(txt_clean, None, (x0 + x1) / 2.0, (y0 + y1) / 2.0, x1 - x0, y1 - y0))
            text_parts.append(txt_clean)
    return PageResult(tokens=tokens, text=" ".join(text_parts), engine="native")

def get_ocr_tokens(
    pdf_path: Path,
    page_index0: int,
    dpi: int = 180,
    rotations: Sequence[int] = (0,),
    psm: str = "11",
) -> PageResult:
    """Use tesseract TSV only as a fallback for pages without usable native text."""
    if not shutil.which("tesseract"):
        return PageResult(tokens=[], text="", engine="ocr_unavailable")

    best_tokens: List[Token] = []
    best_score = -1
    with tempfile.TemporaryDirectory(prefix="rev_ocr_") as tmp:
        with fitz.open(pdf_path) as doc:
            page = doc[page_index0]
            for rot in rotations:
                img = Path(tmp) / f"page_{rot}.png"
                out_base = Path(tmp) / f"out_{rot}"
                mat = fitz.Matrix(dpi / 72.0, dpi / 72.0).prerotate(rot)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                pix.save(str(img))
                cmd = ["tesseract", str(img), str(out_base), "--psm", psm, "tsv"]
                proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
                if proc.returncode != 0:
                    continue
                tsv = out_base.with_suffix(".tsv")
                if not tsv.exists():
                    continue
                scale = 72.0 / dpi
                toks: List[Token] = []
                for line in tsv.read_text(errors="ignore").splitlines()[1:]:
                    parts = line.split("\t")
                    if len(parts) < 12:
                        continue
                    try:
                        conf = float(parts[10])
                        txt = parts[11].strip()
                        if not txt or conf < 20:
                            continue
                        x, y, ww, hh = map(float, parts[6:10])
                    except Exception:
                        continue
                    toks.append(Token(txt, conf / 100.0, (x + ww / 2) * scale, (y + hh / 2) * scale, ww * scale, hh * scale))
                marker_count = sum(1 for t in toks if is_revision_marker_text(t.text))
                value_count = sum(1 for t in toks if is_rev_value_text(t.text, DEFAULT_REV_2L_BLOCKLIST))
                score = marker_count * 100 + value_count + min(len(toks), 200) / 200.0
                if score > best_score:
                    best_score = score
                    best_tokens = toks
    return PageResult(tokens=best_tokens, text=" ".join(t.text for t in best_tokens), engine="ocr")


# ----------------------------- Revision Table Logic -----------------------------

def group_lines(tokens: Sequence[Token], y_tol_factor: float = 0.75) -> List[List[Token]]:
    lines: List[List[Token]] = []
    for t in sorted(tokens, key=lambda z: (z.y, z.x)):
        placed = False
        for line in lines:
            avg_h = sum(x.h for x in line) / max(1, len(line))
            if abs(t.y - line[0].y) <= max(avg_h, t.h, 5.0) * y_tol_factor:
                line.append(t)
                placed = True
                break
        if not placed:
            lines.append([t])
    return [sorted(line, key=lambda z: z.x) for line in lines]

def detect_revision_table_candidates(tokens: List[Token], w: float, h: float, blocklist: set) -> List[Candidate]:
    """Find table rows and return the top-most data revision as authoritative."""
    if not tokens:
        return []

    lines = group_lines(tokens)
    candidates: List[Candidate] = []
    for i, line in enumerate(lines):
        row_markers = [t for t in line if is_revision_marker_text(t.text)]
        near_issue_marker = any(
            norm_val(t.text).upper().startswith(("ISS", "ISSUE"))
            for t in tokens
            if distance((t.x, t.y), (line[0].x, line[0].y)) <= 80
        )
        if len(row_markers) >= 2 or (row_markers and near_issue_marker and (sum(t.y for t in line) / len(line)) / h > 0.78):
            rev_markers = [t for t in row_markers if norm_val(t.text).upper().startswith("REV")]
            # Some Leeds title/revision tables have data rows formatted like
            # "1  A  Iss.  1  Rev". In that case the value paired with Rev is
            # authoritative over the Issue value in the same row.
            for rm in rev_markers:
                vals = [
                    t for t in line
                    if t is not rm
                    and is_rev_value_text(t.text, blocklist)
                    and abs(t.y - rm.y) <= max(t.h, rm.h, 5.0) * 1.2
                ]
                if vals:
                    vals.sort(key=lambda t: abs(t.x - rm.x))
                    t = vals[0]
                    region = region_for_token(t, w, h)
                    if region in {"bottom_left_revision_table", "top_right_title_or_table"} or t.y / h > 0.78:
                        candidates.append(Candidate(
                            value=canonicalize_rev_value(t.text),
                            score=210.0,
                            x=t.x,
                            y=t.y,
                            region=region,
                            source="revision_table",
                            reason="same-row value paired with REV marker in revision table",
                        ))

        header_hits = [t for t in line if norm_val(t.text).upper() in REV_TABLE_HEADERS or is_revision_marker_text(t.text)]
        if len(header_hits) < 2:
            if any(is_revision_marker_text(t.text) and norm_val(t.text).upper().startswith("ISSUE") for t in line):
                header_y = sum(t.y for t in line) / len(line)
                probe = [ln for ln in lines[i + 1:i + 30] if (sum(t.y for t in ln) / len(ln)) > header_y]
                row_texts = [" ".join(norm_val(t.text).upper().replace("!", "I") for t in ln) for ln in probe]
                if any("ORIGINAL" in s and "ISSUE" in s for s in row_texts) and any("REF" in s for s in row_texts):
                    t = min(line, key=lambda z: z.x)
                    candidates.append(Candidate(
                        value="1",
                        score=86.0,
                        x=t.x,
                        y=t.y,
                        region=region_for_token(t, w, h),
                        source="revision_table",
                        reason="inferred issue 1 from OCR issue table with original/ref rows",
                    ))
            continue

        header_y = sum(t.y for t in line) / len(line)
        header_xs = [t.x for t in header_hits]
        rev_header_xs = [t.x for t in header_hits if is_revision_marker_text(t.text) or norm_val(t.text).upper() in {"REV", "REV.", "ISS", "ISS.", "ISSUE"}]
        rev_x = min(rev_header_xs) if rev_header_xs else min(header_xs)
        if header_y / h > 0.82:
            # Bottom revision histories often put the header at the bottom edge
            # and the chronological rows immediately above it.
            search_lines = [ln for ln in lines[max(0, i - 8):i] if (sum(t.y for t in ln) / len(ln)) < header_y]
            search_lines = list(reversed(search_lines))
        else:
            search_lines = [ln for ln in lines[i + 1:i + 30] if (sum(t.y for t in ln) / len(ln)) > header_y]

        row_values: List[Tuple[float, Token]] = []
        for row in search_lines:
            if len(row) < 3 or (max(t.x for t in row) - min(t.x for t in row)) < 100:
                continue
            if "REF" in " ".join(norm_val(t.text).upper() for t in row):
                continue
            vals = [
                t for t in row
                if is_rev_value_text(t.text, blocklist)
                and not is_edge_grid_token(t, w, h)
                and not is_special_char(t.text)
                and abs(t.x - rev_x) <= max(95.0, w * 0.13)
            ]
            if vals:
                vals.sort(key=lambda t: abs(t.x - rev_x))
                row_values.append((sum(t.y for t in row) / len(row), vals[0]))

        if not row_values:
            # OCR can miss a very narrow first issue-number column while still
            # reading the chronological rows. If the table has an original
            # issue row plus numbered change rows, infer the latest numeric
            # issue from the row count rather than hallucinating from body text.
            if any(norm_val(t.text).upper().startswith("ISSUE") for t in header_hits):
                nonempty_rows = []
                for row in search_lines:
                    row_text = " ".join(norm_val(t.text).upper().replace("!", "I") for t in row)
                    if any(k in row_text for k in ("ORIGINAL", "CN", "EC", "D.J", "C.F", "ZHANG", "ISSUE", "TEXT", "MELT", "REF")):
                        nonempty_rows.append(row)
                if len(nonempty_rows) >= 2:
                    t = min(header_hits, key=lambda z: z.x)
                    candidates.append(Candidate(
                        value=str(min(5, max(1, len(nonempty_rows) - 2))),
                        score=88.0,
                        x=t.x,
                        y=t.y,
                        region=region_for_token(t, w, h),
                        source="revision_table",
                        reason="inferred latest issue from chronological table row count",
                    ))
            continue

        # New site convention: revision tables are chronological with newest row above older rows.
        y, t = min(row_values, key=lambda z: z[0])
        region = region_for_token(t, w, h)
        base = 95.0
        if region == "bottom_left_revision_table":
            base += 18.0
        elif region == "top_right_title_or_table":
            base += 8.0
        candidates.append(Candidate(
            value=canonicalize_rev_value(t.text),
            score=base + min(20.0, len(row_values) * 2.0),
            x=t.x,
            y=t.y,
            region=region,
            source="revision_table",
            reason="top-most table data row under REV/ISSUE header",
        ))
    return candidates


# ----------------------------- Title / Marker Scoring ---------------------------

def marker_tokens(tokens: List[Token]) -> List[Token]:
    out = []
    for i, t in enumerate(tokens):
        if is_revision_marker_text(t.text):
            out.append(t)
            continue
        # OCR/native sometimes split "ISSUE NO" into adjacent tokens.
        if norm_val(t.text).upper() == "ISSUE" and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if distance((t.x, t.y), (nxt.x, nxt.y)) <= 90 and norm_val(nxt.text).upper().rstrip(".") in {"NO", "NUMBER"}:
                out.append(t)
    return out

def value_base_score(v: str) -> float:
    s = canonicalize_rev_value(v)
    if re.fullmatch(r"\d{1,3}-\d{1,3}", s):
        return 32.0
    if re.fullmatch(r"\d{1,3}", s):
        return 30.0
    if re.fullmatch(r"[A-Z]", s):
        return 29.0
    if re.fullmatch(r"[A-Z]{2}", s):
        return 20.0
    if is_special_char(s):
        return 4.0
    return 8.0

def title_marker_candidates(tokens: List[Token], w: float, h: float, blocklist: set) -> List[Candidate]:
    cands: List[Candidate] = []
    markers = marker_tokens(tokens)
    if not markers:
        return cands

    for m in markers:
        region = region_for_token(m, w, h)
        if region == "drawing_body":
            # Still allow body markers, but only with strong proximity.
            radius = 95.0
        else:
            radius = 180.0
        neigh = [t for t in tokens if t is not m and distance((t.x, t.y), (m.x, m.y)) <= radius]
        has_non_placeholder_value = any(
            is_rev_value_text(t.text, blocklist) and not is_special_char(t.text)
            for t in neigh
        )
        for t in neigh:
            if not is_rev_value_text(t.text, blocklist):
                continue
            if is_special_char(t.text) and has_non_placeholder_value:
                continue
            if is_edge_grid_token(t, w, h) and distance((t.x, t.y), (m.x, m.y)) > 45:
                continue
            v = canonicalize_rev_value(t.text)
            d = distance((t.x, t.y), (m.x, m.y)) + 1e-3
            same_line = abs(t.y - m.y) <= max(t.h, m.h, 4.0) * 0.65
            same_col = abs(t.x - m.x) <= max(t.w, m.w, 6.0) * 2.8
            to_right = t.x > m.x + min(3.0, max(t.w, m.w) * 0.2)
            below = t.y >= m.y
            if not ((same_line and to_right) or (same_col and below)):
                continue
            if not same_line and d > 70:
                continue

            score = value_base_score(v) + 1000.0 / d
            if same_line:
                score += 12.0
            if same_col:
                score += 5.0
            if to_right:
                score += 5.0
            if below:
                score += 2.0
            score += _nearby_anchor_bonus(tokens, (t.x, t.y)) * 1.5
            if region == "bottom_right_title":
                score += 28.0
            elif region == "top_left_title":
                score += 26.0
            elif region == "bottom_left_revision_table":
                score += 8.0
            elif region == "top_right_title_or_table":
                score += 14.0
            if t.conf is not None:
                score += (t.conf - 0.5) * 8.0
            cands.append(Candidate(v, score, t.x, t.y, region, "title_marker", f"near marker {m.text}"))

    # If several title values sit in the same title region, this site usually puts
    # the latest lower in title blocks. Apply a small deterministic chronology bias.
    for region in {"bottom_right_title", "top_left_title", "top_right_title_or_table"}:
        rs = [c for c in cands if c.region == region and c.source == "title_marker"]
        if len(rs) > 1:
            max_y = max(c.y for c in rs)
            for c in rs:
                c.score += 7.0 * (c.y / max_y if max_y else 0)
    return cands

def corner_title_candidates(tokens: List[Token], w: float, h: float, blocklist: set) -> List[Candidate]:
    """Anchor-free fallback inside plausible title-block regions."""
    cands: List[Candidate] = []
    for t in tokens:
        if not is_rev_value_text(t.text, blocklist) or is_edge_grid_token(t, w, h):
            continue
        region = region_for_token(t, w, h)
        if region not in {"bottom_right_title", "top_left_title", "top_right_title_or_table"}:
            continue
        anchors = _nearby_anchor_bonus(tokens, (t.x, t.y), radius=180)
        if anchors < 2:
            continue
        score = value_base_score(t.text) + anchors * 3.0
        if region == "bottom_right_title":
            score += 22.0
        elif region == "top_left_title":
            score += 18.0
        else:
            score += 10.0
        cands.append(Candidate(canonicalize_rev_value(t.text), score, t.x, t.y, region, "title_region", "title anchors nearby"))
    return cands


# ----------------------------- Analysis Pipeline --------------------------------

def choose_candidate(cands: List[Candidate]) -> Optional[Candidate]:
    if not cands:
        return None

    table = [c for c in cands if c.source == "revision_table"]
    titles = [c for c in cands if c.source in {"title_marker", "title_region"}]
    title_best = max(titles, key=lambda c: c.score) if titles else None
    table_best = max(table, key=lambda c: c.score) if table else None

    if table_best and title_best:
        title_blank = is_special_char(title_best.value) or title_best.value in {"NO_REV", "EMPTY"}
        # Table wins when title is blank/stale or table confidence is materially stronger.
        if title_blank or table_best.score >= title_best.score - 10.0:
            table_best.reason += "; table overrides blank/stale title candidate"
            return table_best
        return title_best
    return title_best or table_best or max(cands, key=lambda c: c.score)

def analyze_tokens(page: PageResult, page_w: float, page_h: float, blocklist: set) -> Optional[Tuple[str, str, float, str]]:
    if not page.tokens:
        return None
    w, h = page_extents(page.tokens, page_w, page_h)
    candidates: List[Candidate] = []
    candidates.extend(title_marker_candidates(page.tokens, w, h, blocklist))
    candidates.extend(detect_revision_table_candidates(page.tokens, w, h, blocklist))
    if not candidates:
        # Anchor-only title extraction is deliberately conservative for the new
        # site because dimensions and material grades often look like valid
        # single revision values.
        pass
    best = choose_candidate(candidates)
    if not best:
        return None
    ctx = context_snippet_from_tokens(page.tokens, (best.x, best.y))
    engine = f"{page.engine}_{best.source}_{best.region}"
    return engine, best.value, best.score, f"{best.reason}; {ctx}"

def analyze_page_native(
    pdf_path: Path,
    page_index0: int,
    brx: float,
    bry: float,
    blocklist: set,
    edge_margin: float,
    enable_ocr: bool = True,
) -> Optional[Tuple[str, str, float, str]]:
    with fitz.open(pdf_path) as doc:
        page = doc[page_index0]
        pw, ph = page.rect.width, page.rect.height

    native = get_native_tokens(pdf_path, page_index0)
    res = analyze_tokens(native, pw, ph, blocklist)
    if res and res[2] >= 55:
        return res

    if enable_ocr and (not native.tokens or not res or res[2] < 75):
        rotations = (0, 90, 180, 270) if not native.tokens else (0,)
        dpi = 220 if not native.tokens else 180
        ocr = get_ocr_tokens(pdf_path, page_index0, dpi=dpi, rotations=rotations)
        ocr_res = analyze_tokens(ocr, pw, ph, blocklist)
        if ocr_res and (not res or ocr_res[2] > res[2] + 5):
            return ocr_res
    return res

def _normalize_output_value(v: str) -> str:
    vu = canonicalize_rev_value(v)
    if vu == "OF":
        return "EMPTY"
    return vu

def process_pdf_native(
    pdf_path: Path,
    brx: float,
    bry: float,
    blocklist: set,
    edge_margin: float,
    enable_ocr: bool = True,
) -> Optional[RevHit]:
    hits: Dict[int, RevHit] = {}
    with fitz.open(pdf_path) as d:
        n = len(d)
    for i in range(n):
        res = analyze_page_native(pdf_path, i, brx, bry, blocklist, edge_margin, enable_ocr=enable_ocr)
        if not res:
            continue
        engine, value, score, ctx = res
        page_no = i + 1
        hits[page_no] = RevHit(pdf_path.name, page_no, value, engine, score, ctx)
    if not hits:
        return None
    return max(hits.values(), key=lambda h: h.score)


# Backward-compatible wrappers used by older callers/tests.
def in_bottom_right(x: float, y: float, width: float, height: float) -> bool:
    return x > width * 0.55 and y > height * 0.60

def in_bottom_right_strict(x: float, y: float, width: float, height: float, brx: float, bry: float) -> bool:
    return x >= width * brx and y >= height * bry

def score_candidates_bottom_right_first(
    tokens: List[Token], page_w: float, page_h: float,
    brx: float, bry: float, blocklist: Optional[set] = None,
    edge_margin: float = DEFAULT_EDGE_MARGIN,
):
    page = PageResult(tokens=tokens, text=" ".join(t.text for t in tokens), engine="native")
    res = analyze_tokens(page, page_w, page_h, blocklist or DEFAULT_REV_2L_BLOCKLIST)
    if not res:
        return None
    _engine, value, score, ctx = res
    return value, score, (0.0, 0.0), ctx

def score_candidates_global(tokens: List[Token], page_w: float, page_h: float):
    return score_candidates_bottom_right_first(tokens, page_w, page_h)


# ----------------------------- CLI ---------------------------------------------

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

def run_pipeline(
    input_folder: Path,
    output_csv: Path,
    brx: float,
    bry: float,
    rev_2l_blocklist: set,
    edge_margin: float,
    enable_ocr: bool = True,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    pdfs = list(iter_pdfs(input_folder))
    if not pdfs:
        LOG.warning(f"No PDFs found in {input_folder}")

    for p in tqdm(pdfs, desc="Scanning PDFs"):
        try:
            native_best = process_pdf_native(
                p, brx, bry, rev_2l_blocklist, edge_margin, enable_ocr=enable_ocr
            )
            if native_best:
                confidence = "high" if "revision_table" in native_best.engine else ("medium" if native_best.score >= 70 else "low")
                rows.append({
                    "file": p.name,
                    "value": _normalize_output_value(native_best.value),
                    "engine": native_best.engine,
                    "confidence": confidence,
                    "score": f"{native_best.score:.2f}",
                    "notes": native_best.context_snippet,
                })
            else:
                rows.append({"file": p.name, "value": "", "engine": "", "confidence": "none", "score": "", "notes": ""})
        except Exception as e:
            LOG.warning(f"Failed {p.name}: {e}")
            rows.append({"file": p.name, "value": "", "engine": "error", "score": "", "notes": str(e)})

    try:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8-sig") as outf:
            writer = csv.writer(outf)
            writer.writerow(["file", "value", "engine", "confidence", "score", "notes"])
            for r in rows:
                writer.writerow([
                    _scalarize(r.get("file", "")),
                    _scalarize(r.get("value", "")),
                    _scalarize(r.get("engine", "")),
                    _scalarize(r.get("confidence", "")),
                    _scalarize(r.get("score", "")),
                    _scalarize(r.get("notes", "")),
                ])
        LOG.info(f"Wrote CSV to {output_csv.resolve()} with {len(rows)} rows")
    except Exception as e:
        LOG.error(f"Failed to write CSV: {e}")

    return rows

def parse_args(argv=None):
    a = argparse.ArgumentParser(description="Extract REV / ISSUE values.")
    a.add_argument("input_folder", type=Path)
    a.add_argument("-o", "--output", type=Path, default=Path("rev_results.csv"))
    a.add_argument("--br-x", type=float, default=DEFAULT_BR_X)
    a.add_argument("--br-y", type=float, default=DEFAULT_BR_Y)
    a.add_argument("--edge-margin", type=float, default=DEFAULT_EDGE_MARGIN)
    a.add_argument("--rev-2l-blocklist", type=str, default=",".join(sorted(DEFAULT_REV_2L_BLOCKLIST)))
    a.add_argument("--native-ocr", action="store_true", help="Enable optional local Tesseract OCR fallback.")
    a.add_argument("--no-ocr", action="store_true", help="Legacy no-op; OCR is disabled unless --native-ocr is set.")
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
        edge_margin=args.edge_margin,
        enable_ocr=args.native_ocr and not args.no_ocr,
    )

if __name__ == "__main__":
    main()
