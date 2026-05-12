#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
new_site_rev_extractor.py
=========================

Production-grade revision / issue extractor for the **new engineering site**.

This module is a refactor of `rev_extractor_fixed_v2.py` + `rev_extractor_updated_v2.py`.
The legacy scripts were tuned to a previous site where:

  * revisions sat in the bottom-right title block,
  * markers were almost exclusively `REV` / `REV.`,
  * single-character numeric values were noise,
  * revision tables were dangerous (stale history).

For the new site those assumptions break:

  * markers include `ISSUE`, `ISS`, `ISS.`, `Issue No.`, alongside `REV` variants,
  * single characters (`A`, `1`, `5`, `B` ...) are first-class values,
  * the value can live in the bottom-right title block, the top-left issue table,
    or be drawn from the bottom-left revision table when the title block is blank,
  * a sizeable minority of drawings has *content-rotated* layout (image rotated 90
    on the page even though `page.rotation == 0`),
  * a small minority is scanned (raster PDF, no native text).

Public entry points
-------------------
* :func:`extract_rev`            — single-file extraction
* :func:`run_pipeline`           — batch processing with CSV output and parallelism
* :class:`AzureVisionExtractor`  — Azure GPT-4 Vision wrapper with strict JSON,
                                   schema validation, retry, and region-zoom

Pipeline modes
--------------
The module supports three operating modes via the ``mode`` parameter:

* ``"pymupdf_only"`` — native PyMuPDF extraction only. Free, fast, handles the
  ~95% majority of digitally-generated drawings on this site. Fails cleanly on
  scanned PDFs and leaves them flagged for review.

* ``"gpt_only"`` — Azure GPT-4 Vision for every file. Most accurate and the
  recommended mode when token cost is not a constraint. The GPT pipeline is
  exhaustively prompted with the new-site value set, region rules, chronology
  rules, and known false friends. JSON output is strictly schema-validated with
  one automatic retry on malformed responses, and a targeted region-zoom pass
  triggers when the primary response returns ``confidence != high``.

* ``"pymupdf_with_gpt_fallback"`` (default) — native PyMuPDF first; GPT-4 Vision
  invoked only when native extraction returns no value or low-confidence value.
  Best cost/accuracy balance.

Native extraction internals
---------------------------
1. **Rotation-aware tokenisation.** PyMuPDF reports word coordinates in
   mediabox space; we apply the page rotation matrix so all downstream logic
   operates in visual coordinates.
2. **Layout fingerprinting.** ``"modern"`` (Rotork) vs ``"classic"`` (Exeeco)
   biases candidate scoring but does not gate it; we always try every
   plausible region.
3. **Marker discovery** across ``Rev``, ``Rev.``, ``Revision``, ``Iss``,
   ``Iss.``, ``Issue``, ``Issue No`` with OCR-safe normalisation.
4. **Noise filters:**
   - template-version stamps pinned to absolute page corners,
   - inline references like ``CONTRACT G/T12345 REV 8``,
   - cluster-based detection of grid letters (A-H column / 1-8 row),
   - hard blocklist for values that are overwhelmingly false friends
     on this site (``O``/``I``, ``OF``/``NO``, ``BS``/``CT``, etc.).
5. **Region-aware "latest" pickers:**
   - bottom-right title block — current value sits below the label,
     bottom-most when stacked;
   - bottom-left revision history table — newest data row above the header;
   - top-left classic Exeeco ISSUE/ALTERATIONS table — newest data row at
     the bottom of the ISSUE column.
6. **Cross-region consolidation** using a strict-newer comparator (letters
   < numbers; within numbers, larger is newer; hyphenated forms compared
   lexicographically on both sides) so the revision table can correctly
   override a stale title-block value.

Every extraction emits a :class:`RevResult` with ``value``, ``confidence``,
``engine``, ``region`` and an extensive ``trace`` for downstream review.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import fitz  # PyMuPDF

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional
    def tqdm(it, **_):
        return it

LOG = logging.getLogger("new_site_rev_extractor")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)

# ---------------------------------------------------------------------------
# Optional dependencies (loaded lazily so the module imports without them)
# ---------------------------------------------------------------------------

try:
    import pytesseract  # type: ignore
    from PIL import Image, ImageOps  # type: ignore
    _OCR_AVAILABLE = True
except Exception:  # noqa: BLE001
    _OCR_AVAILABLE = False
    pytesseract = None  # type: ignore
    Image = None  # type: ignore
    ImageOps = None  # type: ignore

try:
    from openai import AzureOpenAI  # type: ignore
    _AZURE_AVAILABLE = True
except Exception:  # noqa: BLE001
    _AZURE_AVAILABLE = False
    AzureOpenAI = None  # type: ignore


# ===========================================================================
# 1. Patterns & constants
# ===========================================================================

# Marker tokens we treat as a revision/issue label.
# Accepts: REV, Rev, Rev., REVISION, Revision, ISS, Iss, Iss., ISSUE, Issue,
# "ISSUE NO", "Issue No.", and the trailing-dot variants.
MARKER_RE = re.compile(
    r"^(?:rev\.?|revision|iss\.?|issue|issue\s*no\.?)$",
    re.IGNORECASE,
)

# Strict value regex: every value type known on the new site.
#   X-Y    : 0-0, 1-0, 1-1, 0-3, 2-1, 5-0 etc.
#   NN     : 0 .. 99 (covers 1, 8, 10, 16 …)
#   0N     : leading-zero forms 01, 02, … 08
#   [A-Z]  : single letters
#
# We intentionally do NOT accept double letters here — they were a quirk of the
# previous site and would create grid-letter false positives on this site.
VALUE_RE = re.compile(r"^(?:\d{1,2}-\d{1,2}|0\d|\d{1,2}|[A-Z])$")

# A *narrower* sub-pattern used when we have to choose among many candidates,
# to score "obvious" values higher than `O` (could be the letter or the Ø symbol).
HIGH_TRUST_VALUE_RE = re.compile(r"^(?:\d{1,2}-\d{1,2}|0\d|\d{1,2}|[A-DF-Z])$")
# (Excludes 'E' and 'O' which are common drawing artefacts — Ø, grid 'E' etc.)

# Title-block anchor words help confirm we're in the right region.
TITLE_BLOCK_ANCHORS = {
    "DRAWING", "DWG", "DWG.", "SHEET", "SCALE", "SIZE", "TITLE",
    "DRAWN", "CHECKED", "APPROVED", "CREATED", "MATERIAL", "MASS",
    "ROTORK", "EXEECO", "PROJECT",
}

# Revision-table header words: presence of several of these in a tight cluster
# strongly suggests we're looking at a revision/issue history table.
REV_TABLE_HEADERS = {
    "REV", "REV.", "REVISION", "REVISIONS",
    "ISS", "ISS.", "ISSUE", "ISSUES",
    "EC", "ECN", "CN",
    "REVISED", "CHECKED", "APPROVED", "APPR", "APPD",
    "DATE", "DESCRIPTION", "CHANGE", "ALTERATIONS", "ALTERATION",
    "DRAWN", "DFT", "CHKD",
}

# Words/values that should never be returned as the answer — common false friends.
BLOCKLIST_VALUES = {
    "NO",          # column header "No"
    "OF",          # "Sheet 1 OF 1"
    "AT", "BY", "IN", "ON", "TO", "OR",
    "DO", "EC", "CN",
    "BS",          # "BS970", "BS EN" etc.
    "PA",
    "CT",          # CT5, CT9 grade letters next to a digit
    # Diameter symbol Ø is frequently exported as the bare letter "O" by PDF
    # tools. On engineering drawings the standalone letter O is overwhelmingly
    # this symbol, not a real revision value. The value set for this site
    # uses A-D, F-Z, never O, so we can safely exclude it.
    "O",
    # The letter I is similarly noisy (centreline / column annotation in many
    # templates) and is not in the documented value set for this site.
    "I",
}

# Tokens that look value-shaped but are likely the drawing's own grid
# coordinate markers (the A-F vertical column / 1-8 horizontal row).
GRID_LETTER_PATTERN = re.compile(r"^[A-H]$")
GRID_NUMBER_PATTERN = re.compile(r"^[1-8]$")

# Region keyword sets used for layout fingerprinting.
EXEECO_MARKERS = {"EXEECO"}
ROTORK_MARKERS = {"ROTORK", "WWW.ROTORK.COM"}


# ===========================================================================
# 2. Data classes
# ===========================================================================

@dataclass
class Token:
    """A single text token in *visual* (post-rotation) page coordinates."""
    text: str
    x: float          # centre x in visual coordinates
    y: float          # centre y in visual coordinates
    w: float
    h: float
    # Original mediabox coords retained for debugging only.
    mx: float = 0.0
    my: float = 0.0


@dataclass
class PageGeometry:
    width: float
    height: float
    rotation: int
    has_native_text: bool


@dataclass
class Candidate:
    value: str
    score: float
    region: str            # "br_title", "tl_issue", "bl_revtable", "br_marker_block"
    marker_text: str = ""
    x: float = 0.0
    y: float = 0.0
    rationale: str = ""


@dataclass
class RevResult:
    file: str
    page: int = 1
    value: str = ""
    engine: str = ""            # native | native+rotation | ocr | gpt_vision | failed
    confidence: str = "none"    # high | medium | low | none
    region: str = ""            # which region the value came from
    needs_review: bool = False
    review_reason: str = ""
    trace: List[str] = field(default_factory=list)


# Defaults tunable via CLI
DEFAULT_MAX_WORKERS = 4
DEFAULT_RENDER_DPI = 200       # for optional OCR


# ===========================================================================
# 3. Geometry helpers
# ===========================================================================

def _rotate_point(x: float, y: float, rotation: int,
                  mb_w: float, mb_h: float) -> Tuple[float, float, float, float]:
    """Map a (cx, cy) from mediabox space to visual space.

    Returns ``(visual_x, visual_y, visual_w, visual_h)`` where ``visual_w`` and
    ``visual_h`` are the page's *visual* width and height after applying the
    declared rotation. PyMuPDF reports word coordinates in mediabox space even
    when ``page.rotation`` is non-zero, so we transform them ourselves.
    """
    if rotation == 0:
        return x, y, mb_w, mb_h
    if rotation == 90:
        # 90° clockwise: (x, y) -> (mb_h - y, x)
        return mb_h - y, x, mb_h, mb_w
    if rotation == 180:
        return mb_w - x, mb_h - y, mb_w, mb_h
    if rotation == 270:
        return y, mb_w - x, mb_h, mb_w
    return x, y, mb_w, mb_h


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _norm_text(t: str) -> str:
    return re.sub(r"\s+", " ", t.replace("\u00A0", " ")).strip()


def _looks_like_grid_letter(t: Token, page_w: float, page_h: float) -> bool:
    """Heuristic: is this token the drawing-grid label, not a real value?

    Grid labels in this site sit in a column at the absolute left/right edge
    (commonly `A B C D E F` repeated vertically) or a row at the absolute
    top/bottom (`1 2 3 4 5 6 7 8`). The single-letter check uses an *absolute*
    margin in points rather than a fraction of the page, because the title
    block's first data column often sits just 40-50 px from the page edge —
    well inside any percentage-based margin we'd otherwise pick.
    """
    if GRID_LETTER_PATTERN.fullmatch(t.text):
        if t.x < 38.0 or t.x > page_w - 38.0:
            return True
    if GRID_NUMBER_PATTERN.fullmatch(t.text):
        if t.y < 32.0 or t.y > page_h - 32.0:
            return True
    return False


def _build_grid_label_xs(tokens: Sequence[Token], page_w: float) -> Tuple[float, float]:
    """Detect the grid-label column x-coordinates (left and right).

    A real grid has 4-6 letters tightly aligned at the same x. We:
      * collect single-letter tokens A..H near each edge,
      * cluster them by x (within ~6 px),
      * pick the cluster of size >= 3 with the *smallest* x (left side) /
        *largest* x (right side).

    Returns ``(left_x, right_x)`` or ``(-1, -1)`` if not detected. Using a
    median-based cluster pick avoids being skewed by stray A's that aren't
    grid labels but happen to sit close to the edge.
    """

    def _pick_cluster(xs: List[float], prefer: str) -> float:
        if not xs:
            return -1.0
        xs_sorted = sorted(xs)
        clusters: List[List[float]] = []
        for x in xs_sorted:
            if clusters and abs(x - clusters[-1][-1]) <= 6.0:
                clusters[-1].append(x)
            else:
                clusters.append([x])
        # Keep only clusters with >= 3 members (a real grid has 4-6).
        big = [c for c in clusters if len(c) >= 3]
        if not big:
            return -1.0
        if prefer == "left":
            chosen = min(big, key=lambda c: c[0])
        else:
            chosen = max(big, key=lambda c: c[-1])
        return sum(chosen) / len(chosen)

    left_xs = [t.x for t in tokens
               if GRID_LETTER_PATTERN.fullmatch(t.text) and t.x < page_w * 0.15]
    right_xs = [t.x for t in tokens
                if GRID_LETTER_PATTERN.fullmatch(t.text) and t.x > page_w * 0.85]
    return (_pick_cluster(left_xs, "left"),
            _pick_cluster(right_xs, "right"))


def _build_grid_label_ys(tokens: Sequence[Token], page_h: float) -> Tuple[float, float]:
    """As ``_build_grid_label_xs`` but for the top/bottom 1-8 row."""

    def _pick_cluster(ys: List[float], prefer: str) -> float:
        if not ys:
            return -1.0
        ys_sorted = sorted(ys)
        clusters: List[List[float]] = []
        for y in ys_sorted:
            if clusters and abs(y - clusters[-1][-1]) <= 6.0:
                clusters[-1].append(y)
            else:
                clusters.append([y])
        big = [c for c in clusters if len(c) >= 3]
        if not big:
            return -1.0
        if prefer == "top":
            chosen = min(big, key=lambda c: c[0])
        else:
            chosen = max(big, key=lambda c: c[-1])
        return sum(chosen) / len(chosen)

    top_ys = [t.y for t in tokens
              if GRID_NUMBER_PATTERN.fullmatch(t.text) and t.y < page_h * 0.15]
    bot_ys = [t.y for t in tokens
              if GRID_NUMBER_PATTERN.fullmatch(t.text) and t.y > page_h * 0.85]
    return (_pick_cluster(top_ys, "top"),
            _pick_cluster(bot_ys, "bottom"))


def _is_grid_label(
    t: Token,
    grid_xs: Tuple[float, float],
    grid_ys_top_bottom: Tuple[float, float],
) -> bool:
    """Stronger grid-label check that uses the *detected* grid coordinates.

    Only flags a token as a grid label when its position closely matches the
    detected grid coordinate (within 4 px).
    """
    lx, rx = grid_xs
    ty, by = grid_ys_top_bottom
    if GRID_LETTER_PATTERN.fullmatch(t.text):
        if lx > 0 and abs(t.x - lx) <= 4.0:
            return True
        if rx > 0 and abs(t.x - rx) <= 4.0:
            return True
    if GRID_NUMBER_PATTERN.fullmatch(t.text):
        if ty > 0 and abs(t.y - ty) <= 4.0:
            return True
        if by > 0 and abs(t.y - by) <= 4.0:
            return True
    return False


# ===========================================================================
# 4. Token extraction (rotation-aware)
# ===========================================================================

def _extract_tokens(page: fitz.Page) -> Tuple[List[Token], PageGeometry]:
    """Pull native tokens from a page, converting to visual coordinates."""
    mb = page.mediabox
    mb_w, mb_h = mb.width, mb.height
    rotation = page.rotation or 0
    raw = page.get_text("words")
    tokens: List[Token] = []
    page_w_v = mb_w
    page_h_v = mb_h
    for x0, y0, x1, y1, txt, *_ in raw:
        s = _norm_text(txt)
        if not s:
            continue
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        w = x1 - x0
        h = y1 - y0
        vx, vy, page_w_v, page_h_v = _rotate_point(cx, cy, rotation, mb_w, mb_h)
        tokens.append(Token(text=s, x=vx, y=vy, w=w, h=h, mx=cx, my=cy))
    geom = PageGeometry(
        width=page_w_v,
        height=page_h_v,
        rotation=rotation,
        has_native_text=bool(tokens),
    )
    return tokens, geom


# ===========================================================================
# 5. Layout fingerprinting
# ===========================================================================

def _fingerprint_layout(tokens: Sequence[Token]) -> str:
    """Return a coarse layout family: ``"modern"``, ``"classic"``, or ``"unknown"``.

    * ``modern``  — Rotork modern template (BR title block, BL history table).
    * ``classic`` — Exeeco-style template (TL issue table, BR drawing-number block).

    We do not *require* fingerprinting to be correct; downstream extraction
    tries multiple region hypotheses in any case. The fingerprint just biases
    scoring so the right hypothesis wins on tie-breaks.
    """
    text_upper = " ".join(t.text for t in tokens).upper()
    is_classic = any(m in text_upper for m in EXEECO_MARKERS)
    is_modern = any(m in text_upper for m in ROTORK_MARKERS)
    # Strong "classic" signal: the literal phrase "ISSUE ALTERATIONS" or
    # "ORIGINAL ISSUE" usually anchors the TL issue table.
    if "ORIGINAL ISSUE" in text_upper or "ISSUE ALTERATIONS" in text_upper:
        return "classic"
    if is_classic and not is_modern:
        return "classic"
    if is_modern and not is_classic:
        return "modern"
    return "unknown"


# ===========================================================================
# 6. Region predicates (all in visual coordinates)
# ===========================================================================

def _in_region(t: Token, page_w: float, page_h: float, region: str) -> bool:
    """True when token sits inside the given visual region."""
    x, y = t.x, t.y
    if region == "br":   # bottom-right
        return x > page_w * 0.60 and y > page_h * 0.60
    if region == "bl":   # bottom-left
        return x < page_w * 0.45 and y > page_h * 0.60
    if region == "tl":   # top-left
        return x < page_w * 0.45 and y < page_h * 0.40
    if region == "tr":   # top-right
        return x > page_w * 0.60 and y < page_h * 0.40
    return True


# ===========================================================================
# 7. Marker discovery
# ===========================================================================

def _find_markers(tokens: Sequence[Token]) -> List[Token]:
    return [t for t in tokens if MARKER_RE.match(t.text)]


def _is_template_stamp(
    marker: Token,
    other_markers: Sequence[Token],
    page_w: float,
    page_h: float,
) -> bool:
    """Detect the small "Rev NN" template-version stamp that some drawing
    templates place in an absolute page corner (typically bottom-left).

    Such stamps are noise — they describe the *template* version, not the
    drawing's revision. The give-aways are:

      * the marker sits inside the absolute corner of the page (<= 30 px from
        both adjacent edges), AND
      * there is another marker of an equivalent type within ~50 px (the
        genuine title-block or table-header marker).

    The genuine title-block / table-header marker is always slightly inset
    from the page edge to leave room for the box border, so the corner-pinned
    one is the stamp.
    """
    corner_margin = 30.0
    near_left = marker.x <= corner_margin
    near_right = marker.x >= page_w - corner_margin
    near_top = marker.y <= corner_margin
    near_bottom = marker.y >= page_h - corner_margin
    in_corner = (near_bottom and near_left) or (near_bottom and near_right) \
                or (near_top and near_left) or (near_top and near_right)
    if not in_corner:
        return False
    # Look for another non-corner marker within ~100 px (i.e. the real one).
    # Wider than strictly needed for the observed cases (which had the genuine
    # marker within ~25 px) but generous enough to catch template variations
    # we haven't seen yet.
    for other in other_markers:
        if other is marker:
            continue
        if _distance((other.x, other.y), (marker.x, marker.y)) > 100.0:
            continue
        # The "real" marker is the one further from each relevant edge.
        if min(other.x, page_w - other.x, other.y, page_h - other.y) > \
           min(marker.x, page_w - marker.x, marker.y, page_h - marker.y) + 5.0:
            return True
    return False


def _is_inline_rev_reference(marker: Token, tokens: Sequence[Token]) -> bool:
    """True when this REV/ISS marker is actually embedded inside running text
    rather than serving as a column header / label.

    Examples we want to skip:
      * "CONTRACT G/T12345 REV 8"  — body-text reference to a contract revision
      * "PROJECT XYZ — REV B"      — annotation, not the drawing's revision label

    The diagnostic: a real label has no text word *immediately to its left on
    the same baseline*. Title-block labels are isolated in a cell; body-text
    "REV" tokens are flanked by other words.
    """
    same_line = max(marker.h, 6.0) * 1.2
    for t in tokens:
        if t is marker:
            continue
        if abs(t.y - marker.y) > same_line:
            continue
        gap = marker.x - (t.x + t.w / 2.0)
        # Word ending within ~20 px to the left of the marker AND that word
        # is itself a "real" text word (length >= 3, alphabetic) suggests
        # the marker is inline.
        if 0 < gap <= 25.0 and len(t.text) >= 3:
            upper = t.text.upper()
            # Skip if the left-neighbour is itself a known label/anchor.
            if upper in TITLE_BLOCK_ANCHORS:
                continue
            if MARKER_RE.match(t.text):
                continue
            # Has at least one letter? Then it's body text, not a digit.
            if any(ch.isalpha() for ch in t.text):
                return True
    return False


def _classify_marker_region(m: Token, page_w: float, page_h: float) -> str:
    """Return the visual region label for a given marker token."""
    if _in_region(m, page_w, page_h, "br"):
        return "br"
    if _in_region(m, page_w, page_h, "bl"):
        return "bl"
    if _in_region(m, page_w, page_h, "tl"):
        return "tl"
    if _in_region(m, page_w, page_h, "tr"):
        return "tr"
    return "other"


def _classify_marker_region_with_content_bbox(
    m: Token, tokens: Sequence[Token], page_w: float, page_h: float,
) -> str:
    """Region classifier that uses the *content* bounding box rather than the
    raw page rectangle.

    On scanned and oddly-cropped PDFs the drawing content can sit anywhere
    within a much larger page. A marker at y=50% of the page might still be
    in the *visual* top-left of the drawing if the drawing only extends from
    y=10% to y=55%. Compute the content bbox from all tokens and treat THAT
    rectangle as the "page" for region classification.
    """
    primary = _classify_marker_region(m, page_w, page_h)
    if primary != "other":
        return primary
    # Compute content bbox from all tokens.
    if not tokens:
        return "other"
    xs = [t.x for t in tokens]
    ys = [t.y for t in tokens]
    bx0, bx1 = min(xs), max(xs)
    by0, by1 = min(ys), max(ys)
    bw = bx1 - bx0
    bh = by1 - by0
    if bw < 20 or bh < 20:
        return "other"
    # Normalised position inside the content bbox.
    nx = (m.x - bx0) / bw if bw > 0 else 0.5
    ny = (m.y - by0) / bh if bh > 0 else 0.5
    if nx < 0.45 and ny < 0.40:
        return "tl"
    if nx > 0.60 and ny < 0.40:
        return "tr"
    if nx < 0.45 and ny > 0.60:
        return "bl"
    if nx > 0.60 and ny > 0.60:
        return "br"
    return "other"


# ===========================================================================
# 8. Candidate extraction around a marker
# ===========================================================================

def _value_candidates_near_marker(
    marker: Token,
    tokens: Sequence[Token],
    page_w: float,
    page_h: float,
    *,
    radius: float = 80.0,
    grid_xs: Optional[Tuple[float, float]] = None,
    grid_ys: Optional[Tuple[float, float]] = None,
) -> List[Token]:
    """Tokens that look like values and sit within `radius` px of the marker.

    Grid-letter filter strategy: if precise grid coordinates were detected for
    this page, use them exclusively (an A at x=27 only counts as a grid letter
    if the detected grid column is also at ~27). If the precise detector
    couldn't find a grid (likely because this template doesn't have one), fall
    back to the conservative absolute-margin rule.
    """
    have_grid = (grid_xs is not None and (grid_xs[0] > 0 or grid_xs[1] > 0)) or \
                (grid_ys is not None and (grid_ys[0] > 0 or grid_ys[1] > 0))
    out: List[Token] = []
    for t in tokens:
        if t is marker:
            continue
        if not VALUE_RE.match(t.text):
            continue
        if t.text.upper() in BLOCKLIST_VALUES:
            continue
        # Grid filter
        if have_grid:
            if _is_grid_label(t,
                              grid_xs or (-1.0, -1.0),
                              grid_ys or (-1.0, -1.0)):
                continue
        else:
            if _looks_like_grid_letter(t, page_w, page_h):
                continue
        if _distance((t.x, t.y), (marker.x, marker.y)) <= radius:
            out.append(t)
    return out


def _stitch_hyphen_pair(left: Token, right: Token, line_tol: float = 1.2) -> Optional[str]:
    """If two adjacent value-tokens form ``X`` and ``-Y`` on the same baseline,
    stitch them into ``X-Y``. Some PDFs split hyphenated values into separate
    tokens (e.g. ``8`` then ``-0`` → ``8-0``)."""
    if abs(left.y - right.y) > max(left.h, right.h) * line_tol:
        return None
    if right.x - left.x > max(left.w, right.w) * 3.0:
        return None
    if right.x <= left.x:
        return None
    if not re.fullmatch(r"\d{1,2}", left.text):
        return None
    if not re.fullmatch(r"-\d{1,2}|-?\d{1,2}", right.text):
        return None
    rhs = right.text.lstrip("-")
    return f"{left.text}-{rhs}"


def _expand_hyphenated_values(tokens: Sequence[Token]) -> List[Token]:
    """Return synthetic value tokens for any ``N``/``-M`` pairs on the same line.

    This is rarely needed (PyMuPDF usually keeps `8-0` as one word) but it lets
    us catch edge cases where the hyphen got split as a separate token.
    """
    synth: List[Token] = []
    by_line: Dict[int, List[Token]] = {}
    for t in tokens:
        if not re.fullmatch(r"\d{1,2}|-\d{1,2}", t.text):
            continue
        key = int(round(t.y / 4.0))
        by_line.setdefault(key, []).append(t)
    for line in by_line.values():
        line.sort(key=lambda t: t.x)
        for i in range(len(line) - 1):
            stitched = _stitch_hyphen_pair(line[i], line[i + 1])
            if stitched and re.fullmatch(r"\d{1,2}-\d{1,2}", stitched):
                t = line[i]
                synth.append(Token(text=stitched, x=t.x, y=t.y,
                                   w=t.w + line[i + 1].w, h=t.h,
                                   mx=t.mx, my=t.my))
    return synth


# ===========================================================================
# 9. "Most recent" selection in title block and revision table
# ===========================================================================

def _pick_latest_in_title_block(
    candidates: Sequence[Token],
    marker: Token,
) -> Optional[Tuple[Token, str]]:
    """Pick the latest revision value in a *title-block* layout.

    Rule established for this site:
      - Marker label sits at the top of the column.
      - The current value sits directly *below* the marker (larger ``y``).
      - When multiple values are stacked, the **bottom-most** (largest ``y``)
        is the latest.

    We require the candidate to sit *inside* the marker's column (small
    ``|Δx|``) and *below* the marker (``y > marker.y``). The column tolerance
    is deliberately tight: title-block cells in this site are 20-40 px wide.
    """
    if not candidates:
        return None
    # Title-block column width tolerance. Marker height is a good proxy for
    # cell line height; multiply by 3 to allow modest horizontal drift.
    col_tol = max(marker.w * 1.2, marker.h * 3.0, 18.0)
    in_column: List[Token] = []
    for c in candidates:
        if c.y <= marker.y + max(c.h, marker.h) * 0.3:
            continue  # at or above the marker — not the "value below" cell
        if abs(c.x - marker.x) > col_tol:
            continue  # outside the marker's column
        # The value should also be reasonably close vertically (within a few
        # rows). Title-block cells are rarely taller than ~60 px.
        if c.y - marker.y > max(marker.h, c.h) * 8.0:
            continue
        in_column.append(c)
    if in_column:
        # Latest = bottom-most (largest y), then closest to marker x.
        best = max(in_column, key=lambda t: (t.y, -abs(t.x - marker.x)))
        return best, "below-marker"
    # Fallback: candidate sitting on the same line, immediately to the right.
    same_line = [
        c for c in candidates
        if abs(c.y - marker.y) <= max(c.h, marker.h) * 1.2
        and c.x > marker.x
        and (c.x - marker.x) <= max(marker.w, c.w) * 5.0
    ]
    if same_line:
        best = min(same_line, key=lambda t: t.x - marker.x)
        return best, "same-line"
    return None


def _pick_latest_in_revision_table(
    candidates: Sequence[Token],
    marker: Token,
    page_h: float,
) -> Optional[Tuple[Token, str]]:
    """Pick the latest revision value in a *revision-history table*.

    Rule established for this site:
      - Newer revisions are stacked **above** older ones (smaller ``y``).
      - The header (``Iss. | EC | Revised | …``) sits *below* the data rows.
      - The marker tends to be the header itself.

    So we want the candidate with the **smallest** ``y`` (top-most) that is
    *above* the marker header.
    """
    above_header: List[Token] = []
    for c in candidates:
        # Must be at least one row above the marker (account for row height).
        if c.y >= marker.y - max(c.h, marker.h) * 0.4:
            continue
        # Must be in the first column (roughly aligned with the marker x).
        if abs(c.x - marker.x) > max(c.w, marker.w) * 6.0 + 30.0:
            continue
        above_header.append(c)
    if not above_header:
        return None
    best = min(above_header, key=lambda t: (t.y, abs(t.x - marker.x)))
    return best, "above-header"


def _pick_latest_in_classic_issue_table(
    candidates: Sequence[Token],
    marker: Token,
    tokens: Sequence[Token],
    page_w: float,
    page_h: float,
) -> Optional[Tuple[Token, str]]:
    """Pick the latest issue value in an *Exeeco classic* top-left issue table.

    Classic-Exeeco layout (`8000A754`, `8C463`, `IB4D237-CERT31`):

        ISSUE | ALTERATIONS
        ------+----------------
        A     | ORIGINAL ISSUE
        1     | REF 7 ADDED  ...
        2     | ...
        ...
        5     | EC1414 ...

    Rule: header (`ISSUE`) sits at the top, values stack **downward**.
    Latest issue = the **bottom-most** value in the column directly below the
    `ISSUE` header.

    Column-alignment tolerance is deliberately tight: the ISSUE column is
    typically only 30-50 px wide and the value sits centred in it. We reject
    candidates more than ~22 px off the header x.
    """
    if not candidates:
        return None
    col_tol = max(marker.w * 1.4, marker.h * 2.4, 18.0)
    in_column: List[Token] = []
    for c in candidates:
        if c.y <= marker.y + max(marker.h, c.h) * 0.2:
            continue  # at or above marker = not a data row
        if abs(c.x - marker.x) > col_tol:
            continue  # outside the ISSUE column
        # Reject rows much further down than the issue table actually extends.
        # An issue table rarely runs more than 10-12 rows tall; values further
        # away than ~20× marker height are almost certainly something else.
        if c.y - marker.y > max(marker.h, c.h) * 20.0:
            continue
        in_column.append(c)
    if not in_column:
        return None
    # Latest = bottom-most.
    best = max(in_column, key=lambda t: (t.y, -abs(t.x - marker.x)))
    return best, "below-marker"


# ===========================================================================
# 10. Revision-table detection helpers
# ===========================================================================

def _density_of_table_headers(near: Sequence[Token]) -> int:
    return sum(1 for t in near if t.text.upper() in REV_TABLE_HEADERS)


def _is_revision_table_marker(
    marker: Token,
    tokens: Sequence[Token],
    radius: float = 220.0,
) -> bool:
    """Does this marker sit inside a revision-history table?

    We look for >=2 typical table-header words within a tight radius.
    """
    near = [t for t in tokens
            if _distance((t.x, t.y), (marker.x, marker.y)) <= radius]
    return _density_of_table_headers(near) >= 2


# ===========================================================================
# 11. Core marker-anchored extraction
# ===========================================================================

# Region weights for scoring (higher = preferred for the final pick).
REGION_WEIGHTS = {
    "br_title":      100.0,   # bottom-right title block (modern)
    "tl_issue":       95.0,   # top-left issue table (classic)
    "bl_revtable":    80.0,   # bottom-left revision history table
    "br_marker":      70.0,   # bottom-right but inside a small marker block
    "other":          20.0,
}


def _score_candidate(
    cand_value: str,
    base_region_weight: float,
    *,
    distance_to_marker: float,
    high_trust: bool,
    family_match: bool,
) -> float:
    score = base_region_weight
    score += 1000.0 / (distance_to_marker + 5.0)
    if high_trust:
        score += 8.0
    if family_match:
        score += 5.0
    return score


def _extract_from_marker(
    marker: Token,
    tokens: Sequence[Token],
    geom: PageGeometry,
    layout: str,
    *,
    grid_xs: Optional[Tuple[float, float]] = None,
    grid_ys: Optional[Tuple[float, float]] = None,
) -> Optional[Candidate]:
    """Try to extract a value associated with one marker.

    Returns a :class:`Candidate` or ``None``.
    """
    page_w, page_h = geom.width, geom.height
    region = _classify_marker_region_with_content_bbox(
        marker, tokens, page_w, page_h,
    )
    is_table_marker = _is_revision_table_marker(marker, tokens)

    # Collect raw candidates first within a generous radius.
    raw = _value_candidates_near_marker(
        marker, tokens, page_w, page_h, radius=120.0,
        grid_xs=grid_xs, grid_ys=grid_ys,
    )

    # Also include any stitched hyphenated values we synthesised from this neighbourhood.
    synth = _expand_hyphenated_values(
        [t for t in tokens if _distance((t.x, t.y), (marker.x, marker.y)) <= 120.0]
    )
    raw = raw + synth

    if not raw:
        return None

    chosen: Optional[Tuple[Token, str]] = None
    region_tag = "other"

    if region == "br":
        # The BR corner is the *title block*. Even if the surroundings contain
        # words that overlap with our REV_TABLE_HEADERS set (Drawn / Checked /
        # Approved / Date all appear in title blocks too), the value here sits
        # in a single cell directly under the marker. Use the title-block rule.
        chosen = _pick_latest_in_title_block(raw, marker)
        region_tag = "br_title"
    elif is_table_marker:
        # The chronology direction is REGION-DEPENDENT, not purely "table = up".
        #   * TOP-region tables (classic Exeeco "ISSUE / ALTERATIONS") have the
        #     header AT THE TOP and grow downward — latest = BOTTOM-most.
        #   * BOTTOM-region tables (modern Rotork revision history) have the
        #     header AT THE BOTTOM and grow upward — latest = TOP-most.
        if region in ("tl", "tr"):
            chosen = _pick_latest_in_classic_issue_table(raw, marker, tokens, page_w, page_h)
            region_tag = "tl_issue"
        elif region == "bl":
            chosen = _pick_latest_in_revision_table(raw, marker, page_h)
            region_tag = "bl_revtable"
        else:
            chosen = _pick_latest_in_revision_table(raw, marker, page_h)
            region_tag = "other"
    elif region == "tl":
        # Classic Exeeco issue table without other table-header words nearby
        # (single-row "A | ORIGINAL ISSUE" case) — still latest = bottom-most.
        chosen = _pick_latest_in_classic_issue_table(raw, marker, tokens, page_w, page_h)
        region_tag = "tl_issue"
    else:
        chosen = _pick_latest_in_title_block(raw, marker)
        region_tag = f"{region}_misc"

    if not chosen:
        return None
    tok, rationale = chosen

    family_match = (
        (layout == "modern" and region_tag == "br_title") or
        (layout == "classic" and region_tag in {"tl_issue", "bl_revtable"})
    )

    score = _score_candidate(
        tok.text,
        REGION_WEIGHTS.get(region_tag, REGION_WEIGHTS["other"]),
        distance_to_marker=_distance((tok.x, tok.y), (marker.x, marker.y)),
        high_trust=bool(HIGH_TRUST_VALUE_RE.fullmatch(tok.text)),
        family_match=family_match,
    )

    return Candidate(
        value=tok.text,
        score=score,
        region=region_tag,
        marker_text=marker.text,
        x=tok.x,
        y=tok.y,
        rationale=f"{rationale}; marker_at=({marker.x:.0f},{marker.y:.0f})",
    )


# ===========================================================================
# 12. Cross-region consolidation
# ===========================================================================

def _consolidate(
    candidates: Sequence[Candidate],
    layout: str,
) -> Optional[Candidate]:
    """Decide between competing candidates from different regions.

    Rules
    -----
    * Modern site: revision table can carry a NEWER value than the title block;
      if both regions have a value, pick the table's value when it is strictly
      newer (higher number / later letter). Otherwise prefer the title block.
    * Classic site: prefer the TL issue table; only fall back to other regions
      when the TL table is missing.
    * Blank title block (no candidate at all in BR): use whatever the revision
      table gave us.
    """
    if not candidates:
        return None

    # Group by region.
    by_region: Dict[str, Candidate] = {}
    for c in candidates:
        prev = by_region.get(c.region)
        if not prev or c.score > prev.score:
            by_region[c.region] = c

    tl = by_region.get("tl_issue")
    br = by_region.get("br_title")
    bl = by_region.get("bl_revtable")

    if layout == "classic":
        # Strongly prefer TL issue table.
        if tl:
            return tl
        return br or bl or max(by_region.values(), key=lambda c: c.score)

    # Modern / unknown:
    if br and bl:
        if _is_strictly_newer(bl.value, br.value):
            bl.rationale += "; chose table over BR (newer)"
            return bl
        return br
    if br:
        return br
    if bl:
        return bl
    if tl:
        return tl
    return max(by_region.values(), key=lambda c: c.score)


def _is_strictly_newer(a: str, b: str) -> bool:
    """Return True iff revision ``a`` is strictly newer than ``b``.

    Chronology rules (best-effort, deliberately conservative):
      * Letters then numbers: ``A < B < ... < Z`` and ``A < 1 < 2 < ...``
        — once a drawing gets a numeric revision after an alphabetic one,
        any number outranks any letter.
      * Hyphenated forms compare lexicographically on each side: ``1-0 < 1-1``,
        ``1-1 < 2-0``.
    """
    if not a or not b or a == b:
        return False

    def _key(v: str) -> Tuple[int, Tuple[int, ...]]:
        v = v.strip().upper()
        # Hyphenated → (2, (major, minor))
        m = re.fullmatch(r"(\d{1,2})-(\d{1,2})", v)
        if m:
            return (2, (int(m.group(1)), int(m.group(2))))
        # Pure number → (1, (n,))
        if re.fullmatch(r"\d{1,2}", v) or re.fullmatch(r"0\d", v):
            return (1, (int(v),))
        # Single letter → (0, (ord,))
        if re.fullmatch(r"[A-Z]", v):
            return (0, (ord(v) - ord("A"),))
        return (-1, (0,))

    return _key(a) > _key(b)


# ===========================================================================
# 13. Single-page extraction
# ===========================================================================

def _extract_page(
    page: fitz.Page,
    page_index: int,
    *,
    file_name: str,
) -> Tuple[Optional[Candidate], List[str], PageGeometry, List[Token]]:
    """Run native extraction on one page.

    Returns ``(best_candidate_or_None, trace, geometry, tokens)``.
    """
    trace: List[str] = []
    tokens, geom = _extract_tokens(page)
    layout = _fingerprint_layout(tokens)
    trace.append(f"page={page_index + 1} rot={geom.rotation} layout={layout} "
                 f"size=({geom.width:.0f}x{geom.height:.0f}) tokens={len(tokens)}")

    if not tokens:
        trace.append("no native tokens")
        return None, trace, geom, tokens

    markers = _find_markers(tokens)
    trace.append(f"markers={[m.text for m in markers]}")
    if not markers:
        trace.append("no markers found in native text")
        return None, trace, geom, tokens

    # Drop template-version stamps pinned to page corners when a genuine
    # marker exists nearby. Also drop body-text "REV"/"ISS" references that
    # are embedded inside running text (e.g. "CONTRACT G/T12345 REV 8").
    filtered_markers: List[Token] = []
    for m in markers:
        if _is_template_stamp(m, markers, geom.width, geom.height):
            trace.append(f"  skip template stamp '{m.text}'@({m.x:.0f},{m.y:.0f})")
            continue
        if _is_inline_rev_reference(m, tokens):
            trace.append(f"  skip inline ref     '{m.text}'@({m.x:.0f},{m.y:.0f})")
            continue
        filtered_markers.append(m)
    if not filtered_markers:
        # Every marker on the page was filtered. Rather than silently fall
        # back to the unfiltered list (which would return a value derived
        # from a known false friend), return None so the hybrid pipeline
        # can route this file to the GPT fallback.
        trace.append("all markers filtered out — no usable marker on this page")
        return None, trace, geom, tokens
    markers = filtered_markers

    # Detect the drawing-grid label coordinates once per page; these help
    # disambiguate "A is a grid letter" from "A is a revision value".
    grid_xs = _build_grid_label_xs(tokens, geom.width)
    grid_ys = _build_grid_label_ys(tokens, geom.height)
    if grid_xs[0] > 0 or grid_xs[1] > 0:
        trace.append(f"grid_xs={grid_xs} grid_ys={grid_ys}")

    candidates: List[Candidate] = []
    for m in markers:
        c = _extract_from_marker(m, tokens, geom, layout,
                                 grid_xs=grid_xs, grid_ys=grid_ys)
        if c:
            trace.append(f"  marker '{m.text}'@({m.x:.0f},{m.y:.0f}) "
                         f"=> {c.value} region={c.region} score={c.score:.1f}")
            candidates.append(c)
        else:
            trace.append(f"  marker '{m.text}'@({m.x:.0f},{m.y:.0f}) => no value")

    if not candidates:
        trace.append("no marker yielded a value")
        return None, trace, geom, tokens

    best = _consolidate(candidates, layout)
    if best:
        trace.append(f"consolidated => {best.value} region={best.region}")
    return best, trace, geom, tokens


# ===========================================================================
# 14. OCR fallback (for scanned PDFs)
# ===========================================================================

# Tesseract config used when OCR'ing whole pages at moderate DPI to locate
# markers. psm=11 (sparse text) handles scattered drawing labels well.
_OCR_FULL_CONFIG = "--psm 11"

# Tesseract config used inside a marker-anchored crop. We try both psm 6
# (uniform block — good for tables with multiple rows) and psm 11 (sparse —
# good for isolated values) and merge the results.
_OCR_CROP_CONFIGS = ["--psm 6", "--psm 11"]


def _ocr_image_to_tokens(img, *, config: str = _OCR_FULL_CONFIG) -> List[Token]:
    """Run Tesseract on a PIL image and return Token objects.

    Tokens come back in *image* pixel coordinates (origin top-left). The caller
    is responsible for translating them back to page coordinates if needed.
    """
    if not _OCR_AVAILABLE:
        return []
    try:
        data = pytesseract.image_to_data(
            img, output_type=pytesseract.Output.DICT, config=config,
        )
    except Exception as e:  # noqa: BLE001
        LOG.debug(f"OCR error: {e}")
        return []
    tokens: List[Token] = []
    n = len(data.get("text", []))
    for i in range(n):
        s = _norm_text(data["text"][i])
        if not s:
            continue
        # Strip stray punctuation Tesseract sometimes attaches to single tokens.
        s_clean = s.strip("|.,;:'\"()[]{}<>_")
        if not s_clean:
            continue
        x = float(data["left"][i]) + float(data["width"][i]) / 2.0
        y = float(data["top"][i]) + float(data["height"][i]) / 2.0
        w = float(data["width"][i])
        h = float(data["height"][i])
        # Uppercase-fold tokens that look like letters — Tesseract sometimes
        # OCR's a printed "A" as lowercase "a". The new-site value set only
        # contains uppercase letters.
        if re.fullmatch(r"[A-Za-z]{1,2}", s_clean):
            s_clean = s_clean.upper()
        tokens.append(Token(text=s_clean, x=x, y=y, w=w, h=h))
    return tokens


def _ocr_page_to_tokens(page: fitz.Page, dpi: int, rotate_deg: int = 0) -> Tuple[List[Token], "Image.Image"]:
    """Render and OCR a whole page (optionally pre-rotated)."""
    if not _OCR_AVAILABLE:
        return [], None  # type: ignore
    pix = page.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    if rotate_deg:
        img = img.rotate(-rotate_deg, expand=True)
    tokens = _ocr_image_to_tokens(img, config=_OCR_FULL_CONFIG)
    return tokens, img


def _ocr_marker_crop(
    img: "Image.Image",
    marker: Token,
    *,
    pad_x_mult: float = 3.0,
    pad_below_mult: float = 16.0,
    pad_above_mult: float = 2.0,
) -> List[Token]:
    """Second-stage OCR: crop tightly around a marker, upscale, and re-OCR.

    Returns Token objects whose ``x``/``y`` are translated back into the
    ORIGINAL image's coordinate system.

    Crop padding choices:
      * ``pad_x_mult=3`` — horizontal padding generous enough to capture
        sibling cells in a table row AND value cells in narrower columns
        below the marker (the value cell is sometimes narrower than the
        header cell and centred at a slightly different x).
      * ``pad_below_mult=16`` — classic issue tables grow up to ~10 rows;
        we add slack on top.
      * ``pad_above_mult=2`` — captures values that sit ABOVE the marker
        (BL revision history tables).
    """
    if not _OCR_AVAILABLE:
        return []
    pad_x = max(marker.w * pad_x_mult, marker.h * 6.0, 60.0)
    pad_below = max(marker.h * pad_below_mult, 120.0)
    pad_above = max(marker.h * pad_above_mult, 30.0)
    x0 = max(0, int(marker.x - marker.w / 2.0 - pad_x))
    x1 = min(img.width, int(marker.x + marker.w / 2.0 + pad_x))
    y0 = max(0, int(marker.y - marker.h / 2.0 - pad_above))
    y1 = min(img.height, int(marker.y + marker.h / 2.0 + pad_below))
    if x1 <= x0 + 8 or y1 <= y0 + 8:
        return []
    crop = img.crop((x0, y0, x1, y1))
    if crop.mode != "L":
        crop = ImageOps.grayscale(crop)
    crop = ImageOps.autocontrast(crop, cutoff=2)
    crop = crop.resize((crop.width * 3, crop.height * 3), Image.LANCZOS)
    seen: Dict[Tuple[str, int, int], Token] = {}
    for cfg in _OCR_CROP_CONFIGS:
        toks = _ocr_image_to_tokens(crop, config=cfg)
        for t in toks:
            cx = x0 + t.x / 3.0
            cy = y0 + t.y / 3.0
            cw = t.w / 3.0
            ch = t.h / 3.0
            key = (t.text, int(cx / 8), int(cy / 8))
            if key in seen:
                continue
            seen[key] = Token(text=t.text, x=cx, y=cy, w=cw, h=ch)
    return list(seen.values())


def _extract_via_ocr(
    page: fitz.Page,
    *,
    coarse_dpi: int = 200,
    fine_dpi: int = 300,
) -> Tuple[Optional[Candidate], List[str], int]:
    """OCR-based fallback. Two-stage:

    1. Render whole page at ``coarse_dpi``, OCR to find marker positions.
    2. For each marker, crop tightly around it on an image rendered at
       ``fine_dpi`` and re-OCR with PSM 6 + 11.

    Then route the (marker, candidate-tokens) tuples through the same
    :func:`_extract_from_marker` engine used for native text.

    Tries all four image rotations (most drawings are angle 0; the rest is
    insurance for content-rotated scans). The fine-DPI render is computed
    once and rotated in-memory to avoid four expensive pixmap renderings.
    We short-circuit out of the rotation loop as soon as one angle yields a
    high-score (>=100) candidate, since the others are very unlikely to
    beat it.
    """
    trace: List[str] = []
    if not _OCR_AVAILABLE:
        trace.append("OCR not available")
        return None, trace, 0

    # Render the fine image once. We'll rotate it in-memory per angle.
    pix_fine = page.get_pixmap(dpi=fine_dpi)
    fine_base = Image.open(io.BytesIO(pix_fine.tobytes("png")))
    scale = fine_dpi / coarse_dpi

    best: Optional[Candidate] = None
    best_angle = 0
    best_score = -1.0

    for angle in (0, 90, 180, 270):
        # Coarse pass — full page at moderate DPI for marker discovery.
        coarse_toks, _coarse_img = _ocr_page_to_tokens(
            page, dpi=coarse_dpi, rotate_deg=angle,
        )
        if not coarse_toks:
            continue
        markers = _find_markers(coarse_toks)
        if not markers:
            continue

        # Rotate the pre-rendered fine image to match this angle.
        fine_img = fine_base if angle == 0 else fine_base.rotate(-angle, expand=True)
        markers_fine = [
            Token(text=m.text, x=m.x * scale, y=m.y * scale,
                  w=m.w * scale, h=m.h * scale)
            for m in markers
        ]

        # Re-OCR each marker's neighbourhood, in the fine image.
        all_tokens: List[Token] = list(markers_fine)
        for m in markers_fine:
            all_tokens.extend(_ocr_marker_crop(fine_img, m))

        geom = PageGeometry(
            width=float(fine_img.width),
            height=float(fine_img.height),
            rotation=0,
            has_native_text=False,
        )
        layout = _fingerprint_layout(all_tokens)
        grid_xs = _build_grid_label_xs(all_tokens, geom.width)
        grid_ys = _build_grid_label_ys(all_tokens, geom.height)

        # Dedupe markers in fine token list before extraction.
        unique_markers: List[Token] = []
        seen_keys = set()
        for m in markers_fine:
            key = (m.text.upper(), int(m.x / 12), int(m.y / 12))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            unique_markers.append(m)

        cands: List[Candidate] = []
        for m in unique_markers:
            if _is_template_stamp(m, unique_markers, geom.width, geom.height):
                continue
            if _is_inline_rev_reference(m, all_tokens):
                continue
            c = _extract_from_marker(
                m, all_tokens, geom, layout,
                grid_xs=grid_xs, grid_ys=grid_ys,
            )
            if c:
                cands.append(c)
        if not cands:
            continue
        cand = _consolidate(cands, layout)
        if cand and cand.score > best_score:
            best = cand
            best_score = cand.score
            best_angle = angle
            trace.append(
                f"OCR angle={angle} layout={layout} markers={len(unique_markers)} "
                f"=> {cand.value} ({cand.region}) score={cand.score:.1f}"
            )
            # Strong result → skip remaining rotations. The base case
            # (correctly-oriented drawing) almost always wins at angle 0
            # so this short-circuit saves three OCR passes on the typical
            # scanned PDF.
            if cand.score >= 100.0:
                trace.append(f"OCR short-circuited at angle={angle} (strong score)")
                break

    if best is None:
        trace.append("OCR yielded no candidates at any rotation")
    return best, trace, best_angle


# ===========================================================================
# 16. Azure GPT-4 Vision pipeline
# ===========================================================================
#
# Design notes for the new-site GPT pipeline
# ------------------------------------------
# The GPT pipeline must work as a *standalone* extractor (gpt-only mode) AND
# as a fallback after PyMuPDF (default mode). It must therefore be production
# grade by itself — accurate, well-instructed, JSON-safe, cost-bounded, and
# resilient to ambiguity.
#
# Key design decisions:
#
#   1. **Full-page rendering** at a deliberate DPI. The legacy script cropped
#      to the bottom-right quarter, which silently destroyed accuracy for
#      classic-Exeeco top-left issue tables and bottom-left revision history
#      tables. We always send the whole page first. A second targeted region
#      pass is used only when the model returns low confidence.
#
#   2. **Schema-validated JSON output.** We force `response_format=json_object`
#      where supported, validate every returned field against the documented
#      value set, and retry once with a stricter user nudge if the first
#      response is unparseable or schematically wrong.
#
#   3. **Detail-tier control.** `image_url.detail` is set to ``"high"`` for
#      the primary pass. Revision values are small (4-8 pt printed text in a
#      cramped cell); ``"low"`` detail throws away the resolution that makes
#      the difference between reading `8-0` correctly versus mistaking it for
#      `8 0` or `B 0`.
#
#   4. **System prompt is exhaustive about the new site.** It enumerates the
#      exact value set, the three region families, the chronology rule for
#      each region, the conflict-resolution rule, and a long list of
#      known false friends (Sheet 1 of 1, contract refs, grid letters, the
#      diameter symbol Ø rendered as "O", template version stamps).
#
#   5. **Few-shot examples in the system prompt.** Three carefully chosen
#      anchor examples cover the three distinct decisions: title-block
#      bottom-most, BL table top-most, and TL classic-Exeeco bottom-most.
#
#   6. **Multi-page handling.** Drawings with multiple pages typically share
#      the same revision; we extract from page 1 and only consult further
#      pages if page 1 returns low confidence or blank.
#
#   7. **Per-call cost accounting.** Each call records the model's reasoning
#      so a CSV review can be done without re-running the pipeline.

GPT_SYSTEM_PROMPT = """\
You are an expert technical reader of UK engineering drawings produced at a Rotork manufacturing site. Your single job is to extract the **current** revision number (also called issue number) of the drawing shown in the image.

# Vocabulary

The label that marks the revision/issue field on this site uses any of these wordings (treat them as synonyms; spelling, capitalisation and trailing dot are all variant):
- REV, Rev, Rev.
- REVISION, Revision
- ISS, Iss, Iss.
- ISSUE, Issue
- ISSUE NO, Issue No, Issue Number

# Valid value set (exhaustive)

A returned `rev_value` must match exactly one of these forms:
- Hyphenated:  0-0, 1-0, 1-1, 0-1, 2-0, 2-1, 3-0, 4-0, 5-0, 7-0, 8-0, 0-2, 0-3, 1-3, 1-5  (general form: N-M where N and M are 0..99)
- Leading-zero single number: 00, 01, 02, 03, 04, 05, 06, 07, 08, 09
- Plain integer: 0, 1, 2, 3, ... up to 99
- Single letter: A, B, C, D (and any other A-Z if you see it clearly — but NEVER `O` or `I` — see false friends)
- Blank (the title block field is empty AND no revision table value can be read): return `rev_value = ""` with `region = "blank"`.

Reject anything else. If the only candidate you can see is something like `O` (diameter symbol), `I`, `E` (grid letter), a year (`2014`, `19`), a sheet number (`Sheet 1 of 1`), or a contract reference (`CONTRACT G/T12345 REV 8`), return `rev_value = ""` with low confidence.

# Where to look

There are three regions to consider, in priority order:

1. **Bottom-right title block** — modern Rotork template. Has a small cell labelled `Rev` or `Iss.`; the current value sits directly **below** the label, inside that same cell.

2. **Bottom-left revision history table** — modern Rotork template. A multi-row table whose header row sits at the BOTTOM, with column headers like `Iss. | EC | Revised | Checked | Approved | Date | Change Description`. Data rows are stacked ABOVE the header; the most recent revision is the **top-most** row (smallest y in image coordinates, i.e. visually nearest the top of the table block).

3. **Top-left ISSUE/ALTERATIONS table** — classic Exeeco template. A two-column table whose header row sits at the TOP (`ISSUE | ALTERATIONS`); data rows are stacked BELOW the header; the most recent revision is the **bottom-most** value in the `ISSUE` column.

# Chronology rule per region

- **Title block (bottom-right):** if multiple values are stacked in the cell (e.g. an `A` on one line and a `1` below it), the **bottom-most** is the current value. The label is on top, the value is directly below.
- **Bottom-left revision history table:** the header sits at the bottom; data grows upward; the **top-most** data row (visually highest above the header) is the current value.
- **Top-left ISSUE/ALTERATIONS table:** the header sits at the top; data grows downward; the **bottom-most** value in the ISSUE column is the current value.

# Conflict resolution (very important)

The revision history table can carry a NEWER value than the title-block field. When both regions show a value:

- If the bottom-left revision history table contains a value that is *strictly newer* than the title-block field, use the **table** value. Chronology: letters < numbers (A < B < ... < Z, then 1 > Z; once a number appears the letters are obsolete); within numbers, larger is newer; within hyphenated forms, compare lexicographically on each side (`1-0 < 1-1 < 2-0`).
- If the title-block field is blank but the table has values, use the table.
- Otherwise prefer the title-block field.

For classic-Exeeco drawings, only the top-left ISSUE/ALTERATIONS table is authoritative; there is no separate title-block revision field to conflict with.

# False friends to ignore

Do **not** return any of these even if they look like values:

- `O` standing alone — it is the diameter symbol Ø, never a revision value on this site.
- `I` standing alone — centreline annotation.
- Single A-H letters in a vertical column at the absolute left or right edge of the page — drawing-grid labels.
- Single 1-8 digits in a horizontal row at the absolute top or bottom edge of the page — drawing-grid labels.
- `Sheet 1 of 1` style annotations — the `1` here is the sheet number.
- `Rev 01` printed near a page corner with no other revision-table context — a template-version stamp; ignore.
- `CONTRACT G/T... REV 8` and similar — body-text contract references; ignore the `8`.
- Years rendered next to `Rev`/`Iss.` cells (e.g. `2014`, `19`) — also ignore.
- The letter inside `REF 'A'` boxes in the parts list of an Exeeco drawing — that's a part reference, not a revision.

# Output

Return ONLY a JSON object with exactly these keys:

{
  "rev_value":   "<value from the set above, or empty string>",
  "region":      "br_title" | "bl_revtable" | "tl_issue" | "blank" | "other",
  "confidence":  "high" | "medium" | "low",
  "evidence":    "<<= 25 word description of where you saw it and which neighbouring tokens confirm it>"
}

`confidence` calibration:
- `high` — you can read the label and value clearly; the value matches the documented set; no ambiguity.
- `medium` — you can read the value but the label is partly occluded, OR you had to pick between two candidates.
- `low` — image is unclear, value is ambiguous, or you cannot fully confirm the region.

If you genuinely cannot determine the value, return `rev_value = ""` with `confidence = "low"` and `region = "other"`. Never invent a value.
"""

# Few-shot anchor examples appended to the system prompt. We keep them short
# and surface only the discriminating pattern in each.
GPT_FEWSHOT_EXAMPLES = """
# Worked examples

Example A — modern Rotork drawing with stale title block and newer table value:
  - Bottom-right title block: `Iss. A`
  - Bottom-left revision table (newest-on-top): `1` above `A` above header
  - Correct answer: rev_value=1, region=bl_revtable (table is strictly newer)

Example B — modern Rotork drawing with simple title block:
  - Bottom-right title block: `Rev 8-0`
  - Bottom-left table: only `A` entry above header (older)
  - Correct answer: rev_value=8-0, region=br_title

Example C — classic Exeeco drawing with multi-row ISSUE table:
  - Top-left table: header `ISSUE | ALTERATIONS`; rows `A`, `1`, `2`, `3`, `4`, `5` reading top-to-bottom
  - Correct answer: rev_value=5, region=tl_issue (bottom-most in ISSUE column)
"""


# JSON schema we expect back. Validation here is strict on rev_value because
# the model occasionally returns plausible-looking values that aren't in the
# documented set (e.g. "5-40", "DE").
def _validate_gpt_payload(data: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(data, dict):
        return False, "not a dict"
    if "rev_value" not in data:
        return False, "missing rev_value"
    value = data["rev_value"]
    if not isinstance(value, str):
        return False, "rev_value must be a string"
    # Empty string is a legal value (meaning "blank/unknown").
    if value != "" and not VALUE_RE.match(value):
        return False, f"rev_value {value!r} not in documented value set"
    region = data.get("region", "other")
    if region not in {"br_title", "bl_revtable", "tl_issue", "blank", "other"}:
        return False, f"region {region!r} not in allowed set"
    conf = (data.get("confidence") or "medium").lower()
    if conf not in {"high", "medium", "low"}:
        return False, f"confidence {conf!r} not in allowed set"
    return True, ""


class AzureVisionExtractor:
    """Production-grade Azure GPT-4 Vision extractor for the new site.

    Parameters
    ----------
    endpoint, api_key
        Azure OpenAI credentials. Public surface accepts the full endpoint URL
        with or without the ``/openai/deployments/...`` suffix.
    deployment
        Azure deployment name. Defaults to ``"gpt-4.1"`` — GPT-4.1 / GPT-4o
        class models handle the small-text reading in revision cells reliably.
        Smaller models (e.g. ``gpt-4o-mini``) work but lose ~5-10 pp on hard
        cases.
    primary_dpi
        DPI for the first full-page pass. 140 is a good balance: most modern
        drawings have ~8 pt text in revision cells, which becomes ~16 px at
        this DPI — readable by the model without driving up cost. Scans with
        smaller text may benefit from 180.
    api_version
        Azure OpenAI API version. ``"2024-08-01-preview"`` supports
        ``response_format=json_object``.
    max_retries
        Number of retries on a malformed JSON response. One retry is almost
        always enough.
    enable_region_zoom
        When the primary pass returns `confidence != high`, render a second,
        higher-DPI image of the implicated region and ask the model again.
        Improves accuracy on faint scans at the cost of one extra call.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        *,
        deployment: str = "gpt-4.1",
        primary_dpi: int = 140,
        api_version: str = "2024-08-01-preview",
        max_retries: int = 1,
        enable_region_zoom: bool = True,
    ) -> None:
        if not _AZURE_AVAILABLE:
            raise RuntimeError("openai package not installed — run "
                               "`pip install openai>=1.0`")
        endpoint = endpoint.rstrip("/")
        if "/openai/deployments" in endpoint:
            endpoint = endpoint.split("/openai/deployments")[0]
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
        )
        self.deployment = deployment
        self.primary_dpi = primary_dpi
        self.max_retries = max_retries
        self.enable_region_zoom = enable_region_zoom
        # Cumulative cost counter (informational).
        # Guarded by a lock because a single AzureVisionExtractor instance
        # is shared across all parallel workers in run_pipeline.
        self.calls_made = 0
        self._call_lock = threading.Lock()

    # ---- rendering helpers ------------------------------------------------

    def _render_page_b64(self, page: fitz.Page, dpi: int) -> str:
        import base64
        pix = page.get_pixmap(dpi=dpi)
        return base64.b64encode(pix.tobytes("png")).decode("ascii")

    def _render_region_b64(
        self,
        page: fitz.Page,
        region: str,
        *,
        dpi: int = 220,
    ) -> Optional[str]:
        """Render a quadrant of the page for the targeted zoom-in pass.

        ``region`` is one of ``"br_title"``, ``"tl_issue"``, ``"bl_revtable"``.
        Coordinates are taken from the *visual* page (we apply the rotation
        matrix so the crop ends up looking the right way up regardless of
        ``page.rotation``).
        """
        import base64
        # Use page.rect, which is already rotated to visual orientation.
        r = page.rect
        if region == "br_title":
            clip = fitz.Rect(r.x0 + r.width * 0.55, r.y0 + r.height * 0.60,
                             r.x1, r.y1)
        elif region == "bl_revtable":
            clip = fitz.Rect(r.x0, r.y0 + r.height * 0.60,
                             r.x0 + r.width * 0.55, r.y1)
        elif region == "tl_issue":
            clip = fitz.Rect(r.x0, r.y0,
                             r.x0 + r.width * 0.55, r.y0 + r.height * 0.45)
        else:
            return None
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
        return base64.b64encode(pix.tobytes("png")).decode("ascii")

    # ---- single-call helpers ---------------------------------------------

    def _call_gpt(
        self,
        b64_images: Sequence[str],
        user_text: str,
        *,
        max_tokens: int = 400,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """Make one GPT call with one or more images attached.

        Returns ``(parsed_dict_or_None, raw_text)``. Forces JSON object output
        when the deployment supports it; falls back to free-form on the rare
        deployment that doesn't.
        """
        content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
        for b64 in b64_images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64}",
                    "detail": "high",
                },
            })
        kwargs: Dict[str, Any] = {
            "model": self.deployment,
            "messages": [
                {"role": "system", "content": GPT_SYSTEM_PROMPT + GPT_FEWSHOT_EXAMPLES},
                {"role": "user", "content": content},
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        # Newer deployments support strict JSON object output. Try it; if the
        # deployment rejects the parameter, fall back to free-form and parse.
        try:
            resp = self.client.chat.completions.create(
                **kwargs, response_format={"type": "json_object"},
            )
        except Exception:  # noqa: BLE001
            resp = self.client.chat.completions.create(**kwargs)
        with self._call_lock:
            self.calls_made += 1
        raw = resp.choices[0].message.content or ""
        return _parse_gpt_json(raw), raw

    # ---- public entry points ---------------------------------------------

    def extract_page(
        self,
        page: fitz.Page,
    ) -> Tuple[Optional[Candidate], List[str]]:
        """Extract from a single PDF page using the GPT pipeline.

        Order of operations:
          1. Primary pass: full page at ``self.primary_dpi``.
          2. If primary returns ``confidence != high`` or invalid JSON, retry
             primary once with a stricter user nudge.
          3. If still not ``high`` and ``enable_region_zoom`` is set, render
             the most-likely region (per primary's `region` hint, or BR by
             default) at higher DPI and re-ask.
          4. Reconcile, return the highest-confidence valid result.
        """
        trace: List[str] = []
        # Primary
        b64 = self._render_page_b64(page, self.primary_dpi)
        primary_user = (
            "Extract the current revision (or issue) value from the engineering "
            "drawing shown. Follow every rule in the system instructions. "
            "Return ONLY the JSON object."
        )
        data, raw = self._call_gpt([b64], primary_user)
        if not data:
            trace.append(f"primary: unparseable JSON: {raw[:160]}")
        else:
            ok, why = _validate_gpt_payload(data)
            if not ok:
                trace.append(f"primary: schema violation ({why}): {data}")
                data = None
            else:
                trace.append(f"primary: {data}")

        # Single retry with explicit nudge if primary failed.
        if data is None and self.max_retries > 0:
            retry_user = (
                "Your previous response did not conform to the required JSON "
                "schema or was unreadable. Re-read the drawing and return ONLY "
                "a JSON object with keys rev_value, region, confidence, "
                "evidence. rev_value must be empty string or one of: A-Z "
                "(but never `O` or `I` — those are diameter symbol / centreline), "
                "0-99, 0N (leading zero), or N-M (hyphenated, both 0-99). "
                "Do not invent values; if uncertain, return empty string with "
                "confidence=low."
            )
            data, raw = self._call_gpt([b64], retry_user)
            if data:
                ok, why = _validate_gpt_payload(data)
                if not ok:
                    trace.append(f"retry: schema violation ({why}): {data}")
                    data = None
                else:
                    trace.append(f"retry: {data}")
            else:
                trace.append(f"retry: unparseable JSON: {raw[:160]}")

        # Region zoom for non-high-confidence cases.
        if (data is not None
                and self.enable_region_zoom
                and data.get("confidence") != "high"):
            region_hint = data.get("region") or "br_title"
            if region_hint == "blank":
                region_hint = "bl_revtable"  # blank title block → check table
            elif region_hint == "other":
                region_hint = "br_title"
            zoom_b64 = self._render_region_b64(page, region_hint)
            if zoom_b64:
                zoom_user = (
                    f"Here is a higher-resolution crop of the {region_hint} region "
                    f"of the same drawing. Confirm or correct your previous "
                    f"answer ({data.get('rev_value')!r} with confidence "
                    f"{data.get('confidence')!r}). Same JSON schema, same rules."
                )
                data2, raw2 = self._call_gpt([zoom_b64], zoom_user)
                if data2:
                    ok, why = _validate_gpt_payload(data2)
                    if ok:
                        trace.append(f"zoom: {data2}")
                        # Prefer the zoom result if it returns higher confidence
                        # OR if the primary was low. Be conservative: if zoom
                        # disagrees on value but is no more confident than
                        # primary, stick with primary.
                        if (data2.get("confidence") == "high"
                                or data.get("confidence") == "low"):
                            data = data2
                    else:
                        trace.append(f"zoom: schema violation ({why}): {data2}")
                else:
                    trace.append(f"zoom: unparseable JSON: {raw2[:120]}")

        if data is None:
            return None, trace

        value = data.get("rev_value", "")
        region = data.get("region", "other")
        confidence = (data.get("confidence") or "medium").lower()
        evidence = (data.get("evidence") or "")[:200]
        if value == "":
            # Blank is a legitimate answer when the title block is genuinely
            # empty and the revision table is also empty.
            return Candidate(
                value="", region="blank",
                score=REGION_WEIGHTS.get("other", 20.0),
                rationale=f"gpt blank: {evidence}",
            ), trace
        score = REGION_WEIGHTS.get(region, REGION_WEIGHTS["other"])
        if confidence == "high":
            score += 30
        elif confidence == "medium":
            score += 10
        return Candidate(
            value=value, score=score, region=region,
            rationale=f"gpt: {evidence}",
        ), trace

    def extract(
        self,
        pdf_path: Path,
    ) -> Tuple[Optional[Candidate], List[str], int]:
        """Multi-page extraction. Most drawings are single page; multi-page
        drawings share their revision across pages, so we look at page 1
        first and only consult further pages if page 1 returned low
        confidence or empty value.
        """
        trace: List[str] = []
        best: Optional[Candidate] = None
        best_page = 1
        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc):
                cand, page_trace = self.extract_page(page)
                trace.append(f"page {i+1}:")
                trace.extend(f"  {t}" for t in page_trace)
                if cand is None:
                    continue
                if best is None or cand.score > best.score:
                    best = cand
                    best_page = i + 1
                # Stop early if first page is high-confidence and non-blank.
                if cand.score >= REGION_WEIGHTS.get(cand.region, 0) + 25 \
                   and cand.value:
                    break
        return best, trace, best_page


def _parse_gpt_json(raw: str) -> Optional[Dict[str, Any]]:
    """Tolerant JSON extraction. Handles fenced code blocks and bare objects."""
    if not raw:
        return None
    # Strip code fences if present.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1)
    else:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        candidate = m.group(0) if m else None
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # One more pass: try to fix trailing commas.
        cleaned = re.sub(r",(\s*[\]}])", r"\1", candidate)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None


# Back-compat alias
_AzureVisionExtractor = AzureVisionExtractor


# ===========================================================================
# 17. File-level extraction (multi-page, with fallbacks)
# ===========================================================================

PipelineMode = str  # one of "pymupdf_only", "gpt_only", "pymupdf_with_gpt_fallback"

_PIPELINE_MODES = ("pymupdf_only", "gpt_only", "pymupdf_with_gpt_fallback")


def extract_rev(
    pdf_path: Path,
    *,
    mode: PipelineMode = "pymupdf_with_gpt_fallback",
    gpt_extractor: Optional[AzureVisionExtractor] = None,
    enable_ocr: bool = False,
    ocr_dpi: int = DEFAULT_RENDER_DPI,
) -> RevResult:
    """Extract the current revision/issue from a single PDF.

    Parameters
    ----------
    pdf_path
        Path to the input PDF.
    mode
        Pipeline mode:

        * ``"pymupdf_only"`` — native PyMuPDF extraction only. Fastest, free,
          handles ~95% of the digitally-generated drawings on this site.
          Returns ``failed`` on scanned PDFs.
        * ``"gpt_only"`` — Azure GPT-4 Vision only. Most accurate, handles
          scans and unusual layouts. ~1-3¢ per call depending on model.
          Use this when cost is not a constraint and you want one engine.
        * ``"pymupdf_with_gpt_fallback"`` (default) — try PyMuPDF first; fall
          back to GPT only when native extraction fails or returns low
          confidence. Optimal cost/accuracy balance.

    gpt_extractor
        Required for ``"gpt_only"`` and ``"pymupdf_with_gpt_fallback"`` modes.
        Pass a pre-initialised :class:`AzureVisionExtractor` to avoid per-file
        Azure client setup.

    enable_ocr
        Optional Tesseract fallback inserted between native and GPT for cost
        sensitivity. Off by default — GPT is more accurate and the cost
        difference is negligible per the site requirements.
    """
    if mode not in _PIPELINE_MODES:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {_PIPELINE_MODES}")
    if mode in ("gpt_only", "pymupdf_with_gpt_fallback") and gpt_extractor is None:
        raise ValueError(f"mode={mode!r} requires gpt_extractor")

    trace: List[str] = [f"file={pdf_path.name} mode={mode}"]
    try:
        # ----- Mode: gpt_only --------------------------------------------
        if mode == "gpt_only":
            cand, gpt_trace, page_no = gpt_extractor.extract(pdf_path)
            trace.extend(gpt_trace)
            return _candidate_to_result(
                cand, pdf_path, page=page_no, engine="gpt_vision", trace=trace,
            )

        # ----- Native pass (mode = pymupdf_only or pymupdf_with_gpt_fallback)
        native_best: Optional[Candidate] = None
        native_page = 1
        with fitz.open(pdf_path) as doc:
            page_count = len(doc)
            for i in range(page_count):
                page = doc[i]
                cand, page_trace, _geom, _toks = _extract_page(
                    page, i, file_name=pdf_path.name,
                )
                trace.extend(page_trace)
                if cand and (native_best is None or cand.score > native_best.score):
                    native_best = cand
                    native_page = i + 1

            # Optional OCR retry (off by default — kept for cost-sensitive use).
            if enable_ocr and _OCR_AVAILABLE and (
                native_best is None or _below_threshold(native_best)
            ):
                trace.append("native low/empty — invoking optional OCR")
                for i in range(page_count):
                    page = doc[i]
                    cand, ocr_trace, _angle = _extract_via_ocr(
                        page, coarse_dpi=200, fine_dpi=ocr_dpi,
                    )
                    trace.extend(ocr_trace)
                    if cand and (native_best is None or cand.score > native_best.score):
                        native_best = cand
                        native_page = i + 1

        # ----- Mode: pymupdf_only ----------------------------------------
        if mode == "pymupdf_only":
            return _candidate_to_result(
                native_best, pdf_path, page=native_page, engine="native",
                trace=trace,
            )

        # ----- Mode: pymupdf_with_gpt_fallback ---------------------------
        # Use GPT when native produced no value, low-confidence value, or
        # came back blank.
        if native_best is not None and not _below_threshold(native_best) \
                and native_best.value:
            return _candidate_to_result(
                native_best, pdf_path, page=native_page, engine="native",
                trace=trace,
            )
        trace.append("native unsatisfactory — invoking GPT fallback")
        gpt_cand, gpt_trace, page_no = gpt_extractor.extract(pdf_path)
        trace.extend(gpt_trace)
        # If native had something but lower confidence than GPT, prefer GPT
        # only when GPT is at least medium confidence. Otherwise return
        # whichever has a value.
        if gpt_cand is None or not gpt_cand.value:
            if native_best is not None and native_best.value:
                return _candidate_to_result(
                    native_best, pdf_path, page=native_page, engine="native",
                    trace=trace,
                )
            return _candidate_to_result(
                gpt_cand, pdf_path, page=page_no, engine="gpt_vision",
                trace=trace,
            )
        return _candidate_to_result(
            gpt_cand, pdf_path, page=page_no, engine="gpt_vision", trace=trace,
        )

    except Exception as e:  # noqa: BLE001
        return RevResult(
            file=pdf_path.name,
            value="",
            engine="error",
            confidence="none",
            needs_review=True,
            review_reason=f"error: {e}",
            trace=trace + [f"exception: {e}"],
        )


def _candidate_to_result(
    cand: Optional[Candidate],
    pdf_path: Path,
    *,
    page: int,
    engine: str,
    trace: List[str],
) -> RevResult:
    if cand is None:
        return RevResult(
            file=pdf_path.name,
            page=page,
            value="",
            engine="failed" if engine == "native" else f"{engine}_failed",
            confidence="none",
            region="",
            needs_review=True,
            review_reason="no_extraction",
            trace=trace,
        )
    conf = _confidence_label(cand)
    return RevResult(
        file=pdf_path.name,
        page=page,
        value=cand.value,
        engine=engine,
        confidence=conf,
        region=cand.region,
        needs_review=(conf == "low" or cand.value == ""),
        review_reason=("low_confidence" if conf == "low"
                       else ("blank" if cand.value == "" else "")),
        trace=trace,
    )


def _below_threshold(c: Candidate) -> bool:
    """A native result is "below threshold" — and so eligible for GPT fallback
    in hybrid mode — when its score corresponds to the `low` confidence label.
    Keep this aligned with :func:`_confidence_label`.
    """
    return c.score < 70.0


def _confidence_label(c: Candidate) -> str:
    # Note: a GPT candidate marked "high" gets +30 → typically >= 100; a
    # native title-block candidate at min distance gets ~120-180; an OCR
    # candidate sits in the 50-90 range. Keep the "low" boundary aligned
    # with :func:`_below_threshold`.
    if c.score >= 110:
        return "high"
    if c.score >= 70:
        return "medium"
    return "low"


# ===========================================================================
# 18. Batch pipeline
# ===========================================================================

def run_pipeline(
    input_folder: Path,
    output_csv: Path,
    *,
    mode: PipelineMode = "pymupdf_with_gpt_fallback",
    azure_endpoint: str = "",
    azure_key: str = "",
    azure_deployment: str = "gpt-4.1",
    azure_primary_dpi: int = 140,
    azure_enable_region_zoom: bool = True,
    enable_ocr: bool = False,
    max_workers: int = DEFAULT_MAX_WORKERS,
    write_trace: bool = False,
) -> List[Dict[str, Any]]:
    """Run extraction over every PDF in ``input_folder`` and write a CSV.

    Parameters
    ----------
    mode
        See :func:`extract_rev` for mode semantics. Defaults to the recommended
        hybrid mode.
    azure_endpoint, azure_key, azure_deployment
        Azure OpenAI credentials. Required for any mode involving GPT.
    azure_primary_dpi
        DPI for the primary GPT image pass. Higher = better small-text reading
        at the cost of token volume. 140 is a balanced default.
    azure_enable_region_zoom
        Add a targeted second pass on low-confidence primary answers.
    enable_ocr
        Optional Tesseract fallback between native and GPT (off by default).
    max_workers
        Threads for parallel processing.
    write_trace
        Include per-file extraction trace in the CSV.

    Returns
    -------
    list[dict]
        The result rows that were written to the CSV.
    """
    pdfs = sorted(p for p in input_folder.iterdir()
                  if p.is_file() and p.suffix.lower() == ".pdf")
    if not pdfs:
        LOG.warning(f"No PDFs in {input_folder}")
        return []

    # Construct GPT client once if needed.
    gpt: Optional[AzureVisionExtractor] = None
    if mode in ("gpt_only", "pymupdf_with_gpt_fallback"):
        if not (_AZURE_AVAILABLE and azure_endpoint and azure_key):
            raise RuntimeError(
                f"mode={mode!r} requires Azure credentials and the openai "
                f"package. Pass --azure-endpoint and --azure-key, or use "
                f"--mode pymupdf_only."
            )
        gpt = AzureVisionExtractor(
            azure_endpoint, azure_key,
            deployment=azure_deployment,
            primary_dpi=azure_primary_dpi,
            enable_region_zoom=azure_enable_region_zoom,
        )

    def _process(pdf: Path) -> RevResult:
        return extract_rev(
            pdf,
            mode=mode,
            gpt_extractor=gpt,
            enable_ocr=enable_ocr,
        )

    rows: List[RevResult] = []
    if max_workers <= 1:
        for p in tqdm(pdfs, desc="Extracting"):
            rows.append(_process(p))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_process, p): p for p in pdfs}
            for fut in tqdm(as_completed(futures), total=len(pdfs), desc="Extracting"):
                rows.append(fut.result())

    rows.sort(key=lambda r: r.file)

    # Write CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["file", "page", "value", "engine", "confidence", "region",
                  "needs_review", "review_reason"]
    if write_trace:
        fieldnames.append("trace")
    with output_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            row = asdict(r)
            if not write_trace:
                row.pop("trace", None)
            else:
                row["trace"] = " | ".join(r.trace)
            row["needs_review"] = "yes" if r.needs_review else "no"
            w.writerow(row)

    # Summary
    total = len(rows)
    native = sum(1 for r in rows if r.engine == "native")
    gpt_used = sum(1 for r in rows if r.engine == "gpt_vision")
    failed = sum(1 for r in rows
                 if r.engine in {"failed", "error", "gpt_vision_failed"})
    review = sum(1 for r in rows if r.needs_review)
    LOG.info("=" * 60)
    LOG.info(f"Mode:            {mode}")
    LOG.info(f"Processed:       {total}")
    LOG.info(f"  native:        {native}")
    LOG.info(f"  gpt:           {gpt_used}")
    LOG.info(f"  failed:        {failed}")
    LOG.info(f"  needs review:  {review}")
    if gpt is not None:
        LOG.info(f"  GPT calls:     {gpt.calls_made}")
    LOG.info(f"Output:          {output_csv.resolve()}")
    LOG.info("=" * 60)

    return [asdict(r) for r in rows]


# ===========================================================================
# 19. CLI
# ===========================================================================

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Production revision/issue extractor for new-site engineering drawings.",
    )
    p.add_argument("input_folder", type=Path,
                   help="Folder containing PDFs to scan.")
    p.add_argument("-o", "--output", type=Path,
                   default=Path("rev_results.csv"),
                   help="Output CSV path.")
    p.add_argument("--mode", choices=_PIPELINE_MODES,
                   default="pymupdf_with_gpt_fallback",
                   help=("Pipeline mode. "
                         "pymupdf_only = fastest, free, native only; "
                         "gpt_only = highest accuracy, GPT for every file; "
                         "pymupdf_with_gpt_fallback (default) = native first, "
                         "GPT for hard cases."))
    p.add_argument("--azure-endpoint", default=os.getenv("AZURE_OPENAI_ENDPOINT"),
                   help="Azure OpenAI endpoint (or set AZURE_OPENAI_ENDPOINT).")
    p.add_argument("--azure-key", default=os.getenv("AZURE_OPENAI_KEY"),
                   help="Azure OpenAI API key (or set AZURE_OPENAI_KEY).")
    p.add_argument("--azure-deployment", default="gpt-4.1",
                   help="Azure deployment name (default: gpt-4.1).")
    p.add_argument("--azure-primary-dpi", type=int, default=140,
                   help="DPI for the primary GPT image (default: 140).")
    p.add_argument("--azure-no-region-zoom", action="store_true",
                   help="Disable the targeted region-zoom retry pass.")
    p.add_argument("--enable-ocr", action="store_true",
                   help="Insert an optional Tesseract OCR pass between native "
                        "and GPT (off by default).")
    p.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
                   help=f"Thread count (default: {DEFAULT_MAX_WORKERS}).")
    p.add_argument("--trace", action="store_true",
                   help="Include per-file extraction trace in the CSV.")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    t0 = time.time()
    run_pipeline(
        args.input_folder,
        args.output,
        mode=args.mode,
        azure_endpoint=args.azure_endpoint or "",
        azure_key=args.azure_key or "",
        azure_deployment=args.azure_deployment,
        azure_primary_dpi=args.azure_primary_dpi,
        azure_enable_region_zoom=not args.azure_no_region_zoom,
        enable_ocr=args.enable_ocr,
        max_workers=args.max_workers,
        write_trace=args.trace,
    )
    LOG.info(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
