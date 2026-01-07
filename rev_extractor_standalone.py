#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REV Extractor — Improved Robust Version
Fixed bugs, enhanced dot+letter detection, better validation
"""

from __future__ import annotations
import argparse, base64, csv, logging, os, re, json, time, math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import fitz  # PyMuPDF
from tqdm import tqdm

try:
    from openai import AzureOpenAI
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

LOG = logging.getLogger("rev_extractor_improved")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ----------------------------- VALIDATION FRAMEWORK ----------------------------

class ReviewFlag(Enum):
    NONE = "none"
    INVALID_NUMERIC_FORMAT = "invalid_numeric_format"
    INVALID_DOUBLE_LETTER = "invalid_double_letter"
    SINGLE_NUMERIC_ONLY = "single_numeric_only"
    BOTH_ENGINES_DISAGREE = "both_engines_disagree"
    BOTH_ENGINES_INVALID = "both_engines_invalid"

# Special valid REV markers
SPECIAL_VALID_REVS = {
    "-", ".-", "_", "._", "__", "___", "----", "–", "—",
    ". .", ". -", "- .", ". _", "_ ."
}

# Pattern for dot+letter combinations (like .E, .F, .A)
DOT_LETTER_PATTERN = re.compile(r"^\.+[A-Z]{1,2}$", re.IGNORECASE)

# Pattern for letter+dot combinations (like E., F., AA.)
LETTER_DOT_PATTERN = re.compile(r"^[A-Z]{1,2}\.+$", re.IGNORECASE)

def normalize_special_rev(value: str) -> str:
    """Normalize special REV for comparison."""
    if not value:
        return value
    v = value.strip().replace(" ", "")
    if all(c in "-._–— " for c in value):
        return value
    return value

def is_special_valid_rev(value: str) -> bool:
    """Check if value is a special valid REV marker."""
    if not value:
        return False
    # Pure special characters (-, _, etc.)
    if all(c in "-._–— " for c in value):
        return True
    # Dot+letter combinations (.E, .F, .AA)
    if DOT_LETTER_PATTERN.match(value):
        return True
    # Letter+dot combinations (E., F., AA.)
    if LETTER_DOT_PATTERN.match(value):
        return True
    return normalize_special_rev(value) in SPECIAL_VALID_REVS

def is_valid_numeric_rev(value: str) -> Tuple[bool, Optional[str]]:
    """Validate numeric REV: X-Y where Y should be 0."""
    pattern = re.compile(r"^(\d{1,3})-(\d{1,2})$")
    match = pattern.fullmatch(value)
    if not match:
        return True, None
    major, minor = match.groups()
    if int(minor) != 0:
        return False, f"Numeric REV {value} has non-zero minor version (should be X-0)"
    return True, None

def is_valid_double_letter_rev(value: str) -> Tuple[bool, Optional[str]]:
    """
    Validate double letter REV: first letter should be A/B/C.
    Exception: Dot+letter and letter+dot combinations are exempt from this rule.
    """
    if not value or len(value) != 2:
        return True, None
    
    # Skip validation for dot+letter combinations
    if DOT_LETTER_PATTERN.match(value) or LETTER_DOT_PATTERN.match(value):
        return True, None
    
    if not value.isalpha():
        return True, None
    
    first_letter = value[0].upper()
    if first_letter not in ('A', 'B', 'C'):
        return False, f"Double letter REV {value} starts with {first_letter} (should start with A/B/C)"
    
    return True, None

def is_single_numeric_only(value: str) -> bool:
    """Check if value is only digits without hyphen."""
    return bool(re.fullmatch(r"\d+", value))

def validate_rev_value(value: str, engine: str) -> Tuple[bool, Optional[ReviewFlag], Optional[str]]:
    """Comprehensive REV validation."""
    if not value or value in {"NO_REV", "EMPTY"}:
        return True, None, None
    
    if is_special_valid_rev(value):
        return True, None, None
    
    if is_single_numeric_only(value):
        return False, ReviewFlag.SINGLE_NUMERIC_ONLY, f"Single numeric {value} without hyphen"
    
    valid_numeric, numeric_reason = is_valid_numeric_rev(value)
    if not valid_numeric:
        return False, ReviewFlag.INVALID_NUMERIC_FORMAT, numeric_reason
    
    valid_double, double_reason = is_valid_double_letter_rev(value)
    if not valid_double:
        return False, ReviewFlag.INVALID_DOUBLE_LETTER, double_reason
    
    return True, None, None

# ----------------------------- DATA STRUCTURES ---------------------------------

@dataclass
class RevResult:
    file: str
    value: str
    engine: str
    confidence: str = "unknown"
    notes: str = ""
    review_flag: ReviewFlag = ReviewFlag.NONE
    validation_note: str = ""

@dataclass
class Token:
    text: str
    x: float
    y: float
    w: float
    h: float

# ----------------------------- NATIVE EXTRACTION -------------------------------

# Enhanced REV pattern - comprehensive coverage of all formats
REV_VALUE_RE = re.compile(
    r"^(?:"
    r"[A-Z]{1,2}|"                    # Single/double letters: A, B, AA, AB
    r"\d{1,3}-\d{1,2}|"               # Numeric hyphenated: 1-0, 2-0, 3-0
    r"\.+[A-Z]{1,2}|"                 # Dot+letter(s): .E, .F, ..A, .AA
    r"\.+\d{1,3}-\d{1,2}|"            # Dot+numeric: .1-0, .2-0
    r"[A-Z]{1,2}\.+|"                 # Letter(s)+dot: A., E., AA.
    r"\d{1,3}-\d{1,2}\.+|"            # Numeric+dot: 1-0., 2-0.
    r"[-._–—]{1,4}"                   # Pure special markers: -, _, .-, etc.
    r")$",
    re.IGNORECASE
)

REV_TOKEN_RE = re.compile(r"^rev\.?$", re.IGNORECASE)
SIZE_TOKEN_RE = re.compile(r"^size\.?$", re.IGNORECASE)

TITLE_ANCHORS = {"DWG", "DWG.", "DWGNO", "SHEET", "SCALE", "WEIGHT", "SIZE", "DRAWN", "CHECKED"}
REV_TABLE_HEADERS = {
    "REVISIONS", "REVISION", "DESCRIPTION", "DESCRIPTIONS",
    "EC", "DFT", "APPR", "APPD", "DATE", "CHKD", "DRAWN",
    "CHECKED", "APPROVED", "DRAWING", "CHANGE", "ECN"
}
DEFAULT_BLOCKLIST = {"EC", "DF", "DT", "AP", "ID", "NO", "IN", "ON", "BY"}

def norm_val(v: Any) -> str:
    """Normalize token text."""
    if v is None:
        return ""
    s = str(v).replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def get_native_tokens(pdf_path: Path, page_idx: int, corner: str = "br") -> Tuple[List[Token], float, float]:
    """Extract tokens from specific corner of PDF."""
    tokens: List[Token] = []
    
    with fitz.open(pdf_path) as doc:
        page = doc[page_idx]
        rect = page.rect
        page_w, page_h = rect.width, rect.height
        
        # Define corner crop region
        if corner == "br":
            crop = fitz.Rect(page_w * 0.5, page_h * 0.5, page_w, page_h)
        elif corner == "bl":
            crop = fitz.Rect(0, page_h * 0.5, page_w * 0.5, page_h)
        elif corner == "tl":
            crop = fitz.Rect(0, 0, page_w * 0.5, page_h * 0.5)
        elif corner == "tr":
            crop = fitz.Rect(page_w * 0.5, 0, page_w, page_h * 0.5)
        else:
            crop = rect
        
        # Extract words from crop region
        words = page.get_text("words", clip=crop)
        
        for x0, y0, x1, y1, txt, *_ in words:
            txt_clean = txt.strip()
            if not txt_clean:
                continue
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            tokens.append(Token(text=txt_clean, x=cx, y=cy, w=(x1-x0), h=(y1-y0)))
    
    return tokens, page_w, page_h

def is_in_revision_table(token: Token, all_tokens: List[Token], page_w: float, page_h: float) -> bool:
    """Detect if token is in revision table (top area with table headers)."""
    # If in bottom half, less likely to be in rev table
    if token.y > page_h * 0.5:
        return False
    
    # Count revision table headers nearby
    nearby = [t for t in all_tokens if distance((t.x, t.y), (token.x, token.y)) <= 350]
    table_header_count = sum(1 for t in nearby if norm_val(t.text).upper() in REV_TABLE_HEADERS)
    
    return table_header_count >= 2

def is_likely_dimension(value: str, tokens: List[Token], token: Token) -> bool:
    """
    Check if a numeric value is likely a dimension rather than a REV.
    Dimensions often have nearby measurement context.
    """
    if not re.match(r"^\d+-\d+$", value):
        return False
    
    # Check for nearby dimension-like tokens
    nearby = [t for t in tokens if distance((t.x, t.y), (token.x, token.y)) <= 100]
    dimension_indicators = {"HOLE", "DIA", "DEPTH", "LENGTH", "WIDTH", "HEIGHT", "THK", "THICK"}
    
    for t in nearby:
        if any(ind in norm_val(t.text).upper() for ind in dimension_indicators):
            return True
    
    return False

def extract_native_pymupdf_corner(
    pdf_path: Path,
    corner: str = "br",
    page_idx: int = 0
) -> Optional[RevResult]:
    """
    Extract REV from specific corner using PyMuPDF.
    Enhanced with better dot+letter recognition and SIZE field distinction.
    """
    try:
        tokens, page_w, page_h = get_native_tokens(pdf_path, page_idx, corner)
        
        if not tokens:
            return None
        
        # Find REV and SIZE labels
        rev_labels = [t for t in tokens if REV_TOKEN_RE.match(norm_val(t.text))]
        size_labels = [t for t in tokens if SIZE_TOKEN_RE.match(norm_val(t.text))]
        
        # Look for anchor words (title block indicators)
        anchors = [t for t in tokens if norm_val(t.text).upper() in TITLE_ANCHORS]
        has_anchors = len(anchors) > 0
        
        # If no anchors found, might not be valid title block
        if not has_anchors and corner != "br":
            return None
        
        candidates = []
        
        # Strategy 1: Look near REV labels (highest priority)
        if rev_labels:
            for rev_label in rev_labels:
                # Find tokens near REV label
                nearby = [t for t in tokens 
                         if distance((t.x, t.y), (rev_label.x, rev_label.y)) <= 200]
                
                for t in nearby:
                    v = norm_val(t.text)
                    
                    # Check if it matches REV pattern
                    if not REV_VALUE_RE.match(v):
                        continue
                    
                    # Skip if in revision table
                    if is_in_revision_table(t, tokens, page_w, page_h):
                        continue
                    
                    # Skip if near SIZE label (but not if it's dot+letter format)
                    if not (DOT_LETTER_PATTERN.match(v) or LETTER_DOT_PATTERN.match(v)):
                        near_size = any(distance((t.x, t.y), (s.x, s.y)) <= 80 
                                      for s in size_labels)
                        if near_size:
                            continue
                    
                    # Skip if likely a dimension
                    if is_likely_dimension(v, tokens, t):
                        continue
                    
                    # Skip blocklist (but NOT special markers or dot+letter)
                    if not is_special_valid_rev(v) and v.upper() in DEFAULT_BLOCKLIST:
                        continue
                    
                    # Calculate score
                    dist = distance((t.x, t.y), (rev_label.x, rev_label.y))
                    score = 150.0 / (dist + 1.0)  # Higher base score for REV label proximity
                    
                    # MAJOR BOOST: Dot+letter combinations (very common REV format)
                    if DOT_LETTER_PATTERN.match(v):
                        score += 100.0  # Strong boost for .E, .F, etc.
                        LOG.debug(f"  Dot+letter REV candidate: {v} (score: {score:.1f})")
                    
                    # BOOST: Letter+dot combinations
                    if LETTER_DOT_PATTERN.match(v):
                        score += 80.0
                    
                    # Bonus for same line as REV label
                    if abs(t.y - rev_label.y) <= max(t.h, rev_label.h) * 0.8:
                        score += 30.0
                    
                    # Bonus for being to the right of REV label
                    if t.x > rev_label.x:
                        score += 15.0
                    
                    # Bonus for anchor words nearby
                    anchor_count = sum(1 for a in anchors 
                                     if distance((t.x, t.y), (a.x, a.y)) <= 200)
                    score += anchor_count * 10.0
                    
                    # Penalty for being near SIZE label (unless dot+letter)
                    if not (DOT_LETTER_PATTERN.match(v) or LETTER_DOT_PATTERN.match(v)):
                        min_size_dist = min((distance((t.x, t.y), (s.x, s.y)) for s in size_labels), default=999)
                        if min_size_dist < 100:
                            score -= 50.0
                    
                    candidates.append((score, v, f"{corner.upper()}: Near REV label, {anchor_count} anchors"))
        
        # Strategy 2: Look for standalone values near anchors (no explicit REV label)
        else:
            for t in tokens:
                v = norm_val(t.text)
                
                if not REV_VALUE_RE.match(v):
                    continue
                
                if is_in_revision_table(t, tokens, page_w, page_h):
                    continue
                
                # Skip if near SIZE label (but not if it's dot+letter format)
                if not (DOT_LETTER_PATTERN.match(v) or LETTER_DOT_PATTERN.match(v)):
                    near_size = any(distance((t.x, t.y), (s.x, s.y)) <= 80 
                                  for s in size_labels)
                    if near_size:
                        continue
                
                # Skip if likely a dimension
                if is_likely_dimension(v, tokens, t):
                    continue
                
                if not is_special_valid_rev(v) and v.upper() in DEFAULT_BLOCKLIST:
                    continue
                
                # Score based on anchor proximity
                anchor_count = sum(1 for a in anchors 
                                 if distance((t.x, t.y), (a.x, a.y)) <= 200)
                
                if anchor_count > 0:
                    score = anchor_count * 15.0
                    
                    # MAJOR BOOST: Dot+letter combinations
                    if DOT_LETTER_PATTERN.match(v):
                        score += 100.0
                        LOG.debug(f"  Dot+letter REV candidate (no label): {v} (score: {score:.1f})")
                    
                    # BOOST: Letter+dot combinations
                    if LETTER_DOT_PATTERN.match(v):
                        score += 80.0
                    
                    # Penalty for being near SIZE label
                    min_size_dist = min((distance((t.x, t.y), (s.x, s.y)) for s in size_labels), default=999)
                    if min_size_dist < 100:
                        score -= 40.0
                    
                    candidates.append((score, v, f"{corner.upper()}: Near {anchor_count} anchors"))
        
        if not candidates:
            return None
        
        # Choose best candidate
        best_score, best_value, notes = max(candidates, key=lambda x: x[0])
        
        # Log if we found a dot+letter value
        if DOT_LETTER_PATTERN.match(best_value):
            LOG.info(f"  ✓ Dot+letter REV found: {best_value} in {pdf_path.name}")
        
        return RevResult(
            file=pdf_path.name,
            value=best_value,
            engine=f"pymupdf_{corner}",
            confidence="high" if best_score > 80 else ("medium" if best_score > 40 else "low"),
            notes=notes
        )
        
    except Exception as e:
        LOG.debug(f"Native extraction failed for {pdf_path.name} ({corner}): {e}")
        return None

def extract_native_multi_corner(pdf_path: Path) -> Optional[RevResult]:
    """Try extraction from multiple corners."""
    for corner in ["br", "bl", "tl", "tr"]:
        result = extract_native_pymupdf_corner(pdf_path, corner=corner)
        
        if result and result.value and result.value != "NO_REV":
            # Validate the result
            is_valid, flag, reason = validate_rev_value(result.value, "pymupdf")
            
            if is_valid:
                if corner != "br":
                    LOG.info(f"  Found valid REV in {corner.upper()} for {pdf_path.name}")
                return result
    
    return None

# ----------------------------- GPT EXTRACTION ----------------------------------

GPT_SYSTEM_PROMPT = """You are an expert at analyzing engineering drawings and extracting revision information.

YOUR TASK:
Extract the REV (revision) value from the title block of this engineering drawing.

CRITICAL: REV VALUE FORMATS

1. SPECIAL MARKERS (VALID):
   - Single or multiple dashes: -, --, ---
   - Underscores: _, __, ___
   - Combinations: .-, ._, -., _.
   - These are VALID REV markers meaning "no revision yet" or "initial"
   → If you see these in the REV field, return them EXACTLY as shown

2. LETTER REVISIONS (COMMON):
   - Single: A, B, C, D, E, F, ... Z
   - Double: AA, AE, BA, BC, BP, CE
   - **Dot+letter: .A, .B, .E, .F, .Z** ← VERY IMPORTANT! Common format!
   - **Letter+dot: A., E., F.** ← Also valid!
   - Examples: ".E" means revision E, ".F" means revision F
   → The dot is PART of the REV value - include it!

3. NUMERIC REVISIONS:
   - Format: [number]-[number]
   - Common: 1-0, 2-0, 3-0, 9-0, 12-0
   - With dots: .1-0, .2-0, 1-0., 2-0.
   - The second number is USUALLY 0
   - If you see 5-40, 18-8, 8-32, these might be dimensions (not REV)
   → Only return numeric values that are clearly in the REV field

4. NO REV:
   - Use ONLY if NO REV field exists OR field is truly empty
   - Don't use if there's a special marker (-, _, etc.)
   - Don't use if there's a dot+letter (.E, .F)

CRITICAL: DISTINGUISH REV FROM SIZE
Many drawings have both SIZE and REV fields:
- SIZE: Usually a single letter (A, B, C, D) indicating drawing size
- REV: The revision value (could be .E, .F, etc.)
→ Look for the field labeled "REV" or "REV:" - NOT the SIZE field!

TITLE BLOCK IDENTIFICATION:
- Check BOTTOM-RIGHT, BOTTOM-LEFT, TOP-RIGHT, TOP-LEFT
- Contains: DWG NO, SHEET, SCALE, SIZE, **REV**, company logo
- REV field is separate from SIZE field

AVOID:
- SIZE values (A, B, C, D near "SIZE:" label)
- Revision history tables (with dates/descriptions)
- Grid letters on edges
- Section markers (SECTION C-C)
- Dimensions (5-40, 18-8 are likely NOT REVs)

RESPONSE FORMAT:
{
  "rev_value": ".E",
  "confidence": "high",
  "location": "bottom-right title block, REV field",
  "notes": "Dot+letter REV .E, distinct from SIZE value"
}

EXAMPLES:

Example 1 - Dot+letter (VERY COMMON):
"SIZE: B | DWG NO: EA-13855 | REV: .E | SHEET 1"
✅ {"rev_value": ".E", "notes": "Dot+letter REV .E (not SIZE B)"}
❌ Wrong: {"rev_value": "B"} ← This is the SIZE, not REV!

Example 2 - Another dot+letter:
"SIZE: A | DWG NO: EA-10840 | REV: .F"
✅ {"rev_value": ".F", "notes": "Dot+letter REV .F (not SIZE A)"}
❌ Wrong: {"rev_value": "A"} ← This is the SIZE, not REV!

Example 3 - Letter+dot:
"REV: E. | SCALE 1:1"
✅ {"rev_value": "E.", "notes": "Letter+dot REV E."}

Example 4 - Plain letter:
"REV: E | SHEET 1"
✅ {"rev_value": "E", "notes": "Plain letter REV E"}

Example 5 - Numeric with dot:
"REV: .2-0"
✅ {"rev_value": ".2-0", "notes": "Dot+numeric REV .2-0"}

Example 6 - Special marker:
"DWG NO: 22468 | REV: - | SHEET 1"
✅ {"rev_value": "-", "notes": "Special REV marker (dash)"}

REMEMBER:
- Dot+letter (.E, .F, .A, .Z) is a VALID and COMMON REV format!
- Letter+dot (E., F., A.) is also VALID!
- Always check the "REV:" label - NOT the "SIZE:" label!
- Include the dot in your response if it's part of the REV value
"""

class AzureGPTExtractor:
    def __init__(self, endpoint: str, api_key: str, deployment_name: str = "gpt-4.1"):
        if not AZURE_AVAILABLE:
            raise ImportError("openai not installed")
        
        endpoint = endpoint.rstrip('/')
        if '/openai/deployments' in endpoint:
            endpoint = endpoint.split('/openai/deployments')[0]
        
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version="2024-02-15-preview",
            azure_endpoint=endpoint
        )
        self.deployment_name = deployment_name
        LOG.info("✓ GPT-4.1 ready")
    
    def pdf_to_base64(self, pdf_path: Path, corner: str = "br", dpi: int = 150) -> str:
        """Convert PDF corner to base64."""
        with fitz.open(pdf_path) as doc:
            page = doc[0]
            rect = page.rect
            
            if corner == "br":
                crop = fitz.Rect(rect.width * 0.5, rect.height * 0.5, rect.x1, rect.y1)
            elif corner == "bl":
                crop = fitz.Rect(rect.x0, rect.height * 0.5, rect.width * 0.5, rect.y1)
            elif corner == "tl":
                crop = fitz.Rect(rect.x0, rect.y0, rect.width * 0.5, rect.height * 0.5)
            elif corner == "tr":
                crop = fitz.Rect(rect.width * 0.5, rect.y0, rect.x1, rect.height * 0.5)
            else:
                crop = rect
            
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False, clip=crop)
            return base64.b64encode(pix.tobytes("png")).decode('utf-8')
    
    def extract_rev(self, pdf_path: Path, corner: str = "br") -> RevResult:
        """Extract REV using GPT-4.1."""
        try:
            img_b64 = self.pdf_to_base64(pdf_path, corner, dpi=150)
            
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": GPT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Extract REV from {corner.upper()} corner. CRITICAL: Look for 'REV:' label (NOT 'SIZE:' label). Dot+letter (.E, .F, .Z) is VERY COMMON - include the dot! Special markers (-, _) are also VALID."
                            },
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                        ]
                    }
                ],
                max_tokens=400,
                temperature=0
            )
            
            result_text = response.choices[0].message.content or ""
            
            # Parse JSON
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', result_text, re.DOTALL | re.IGNORECASE)
            if json_match:
                result_text = json_match.group(1)
            
            result_data = json.loads(result_text.strip())
            
            value = (result_data.get("rev_value") or "").strip()
            notes = (result_data.get("notes") or "").strip()
            confidence = (result_data.get("confidence") or "unknown").lower()
            
            if value.upper() in {"NO_REV", "EMPTY", "N/A", "NA"}:
                value = "NO_REV"
            
            # Log if GPT found a dot+letter value
            if DOT_LETTER_PATTERN.match(value):
                LOG.info(f"  ✓ GPT found dot+letter REV: {value} in {pdf_path.name}")
            
            return RevResult(
                file=pdf_path.name,
                value=value,
                engine=f"gpt_{corner}",
                confidence=confidence,
                notes=notes[:150]
            )
            
        except Exception as e:
            LOG.error(f"GPT failed {pdf_path.name} ({corner}): {e}")
            return RevResult(pdf_path.name, "", "gpt_failed", "none", str(e)[:100])

# ----------------------------- SMART PIPELINE ----------------------------------

def smart_extraction(pdf_path: Path, gpt: AzureGPTExtractor) -> RevResult:
    """Smart extraction with validation and rerun logic."""
    
    # Step 1: Try native multi-corner
    native_result = extract_native_multi_corner(pdf_path)
    
    if native_result and native_result.value and native_result.value != "NO_REV":
        # Validate
        is_valid, flag, reason = validate_rev_value(native_result.value, "pymupdf")
        
        if is_valid:
            return native_result
        else:
            # Suspicious - rerun with GPT
            LOG.info(f"  Native returned suspicious '{native_result.value}': {reason}")
            LOG.info(f"  Rerunning with GPT...")
            
            corner = "br"
            if "_" in native_result.engine:
                corner = native_result.engine.split("_")[-1]
            
            gpt_result = gpt.extract_rev(pdf_path, corner=corner)
            gpt_valid, gpt_flag, gpt_reason = validate_rev_value(gpt_result.value, "gpt")
            
            if gpt_valid:
                gpt_result.notes = f"Native={native_result.value} ({reason}), GPT corrected to {gpt_result.value}"
                return gpt_result
            elif native_result.value == gpt_result.value:
                # Both agree on suspicious value
                native_result.review_flag = flag or ReviewFlag.BOTH_ENGINES_DISAGREE
                native_result.validation_note = f"REVIEW: {reason}"
                return native_result
            else:
                # Both different invalid
                return RevResult(
                    file=pdf_path.name,
                    value=native_result.value,
                    engine=f"{native_result.engine}+{gpt_result.engine}",
                    confidence="low",
                    notes=f"Native={native_result.value}, GPT={gpt_result.value}",
                    review_flag=ReviewFlag.BOTH_ENGINES_INVALID,
                    validation_note=f"REVIEW: {reason} | GPT: {gpt_reason}"
                )
    
    # Step 2: Native failed - use GPT multi-corner
    LOG.info(f"→ GPT: {pdf_path.name}")
    
    for corner in ["br", "bl", "tl", "tr"]:
        gpt_result = gpt.extract_rev(pdf_path, corner=corner)
        
        if gpt_result.value and gpt_result.value != "NO_REV":
            is_valid, flag, reason = validate_rev_value(gpt_result.value, "gpt")
            
            if is_valid:
                if corner != "br":
                    gpt_result.notes = f"[{corner.upper()}] {gpt_result.notes}"
                return gpt_result
            elif corner == "tr":
                gpt_result.review_flag = flag or ReviewFlag.NONE
                gpt_result.validation_note = f"REVIEW: {reason}" if reason else ""
                return gpt_result
    
    return RevResult(pdf_path.name, "NO_REV", "all_failed", "low", "All attempts returned NO_REV")

# ----------------------------- MAIN PIPELINE -----------------------------------

def run_pipeline(
    input_folder: Path,
    output_csv: Path,
    azure_endpoint: str,
    azure_key: str,
    deployment_name: str = "gpt-4.1"
) -> List[Dict[str, Any]]:
    """Fully standalone robust pipeline."""
    rows: List[Dict[str, Any]] = []
    
    LOG.info("Initializing GPT...")
    gpt = AzureGPTExtractor(azure_endpoint, azure_key, deployment_name)
    
    pdfs = sorted(input_folder.glob("*.pdf"))
    if not pdfs:
        LOG.warning(f"No PDFs in {input_folder}")
        return rows
    
    LOG.info(f"Processing {len(pdfs)} PDFs")
    
    stats = {
        "native": 0, 
        "gpt": 0, 
        "failed": 0, 
        "flagged": 0,
        "dot_letter": 0  # Track dot+letter detections
    }
    
    for pdf_path in tqdm(pdfs, desc="Processing"):
        try:
            result = smart_extraction(pdf_path, gpt)
            
            if "pymupdf" in result.engine:
                stats["native"] += 1
            elif "gpt" in result.engine:
                stats["gpt"] += 1
            if result.review_flag != ReviewFlag.NONE:
                stats["flagged"] += 1
            
            # Track dot+letter format detections
            if DOT_LETTER_PATTERN.match(result.value) or LETTER_DOT_PATTERN.match(result.value):
                stats["dot_letter"] += 1
            
            rows.append({
                "file": result.file,
                "value": result.value,
                "engine": result.engine,
                "confidence": result.confidence,
                "review_flag": result.review_flag.value,
                "validation_note": result.validation_note,
                "notes": result.notes[:100]
            })
            
        except Exception as e:
            LOG.error(f"Failed {pdf_path.name}: {e}")
            stats["failed"] += 1
            rows.append({
                "file": pdf_path.name,
                "value": "",
                "engine": "error",
                "confidence": "none",
                "review_flag": "error",
                "validation_note": str(e)[:50],
                "notes": str(e)[:100]
            })
    
    # Write CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'file', 'value', 'engine', 'confidence', 'review_flag', 'validation_note', 'notes'
        ])
        writer.writeheader()
        writer.writerows(rows)
    
    LOG.info(f"\n{'='*60}")
    LOG.info(f"Results: {output_csv}")
    LOG.info(f"Total: {len(rows)} | Native: {stats['native']} | GPT: {stats['gpt']}")
    LOG.info(f"Dot+letter formats: {stats['dot_letter']} | Flagged: {stats['flagged']} | Failed: {stats['failed']}")
    LOG.info(f"Cost: ${stats['gpt'] * 0.01:.2f}")
    LOG.info(f"{'='*60}\n")
    
    return rows

def main():
    start = time.time()
    
    p = argparse.ArgumentParser(description="Improved Robust REV Extractor")
    p.add_argument("input_folder", type=Path)
    p.add_argument("-o", "--output", type=Path, default=Path("rev_results_improved.csv"))
    p.add_argument("--azure-endpoint", default=os.getenv("AZURE_OPENAI_ENDPOINT"))
    p.add_argument("--azure-key", default=os.getenv("AZURE_OPENAI_KEY"))
    p.add_argument("--deployment-name", default="gpt-4.1")
    args = p.parse_args()
    
    if not args.azure_endpoint or not args.azure_key:
        LOG.error("Azure credentials required")
        return []
    
    results = run_pipeline(
        args.input_folder,
        args.output,
        args.azure_endpoint,
        args.azure_key,
        args.deployment_name
    )
    
    elapsed = time.time() - start
    LOG.info(f"Completed in {int(elapsed//60)}m {int(elapsed%60)}s")
    
    return results

if __name__ == "__main__":
    main()
