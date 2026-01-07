#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REV Extractor — Standalone with Enhanced SIZE/REV Distinction
CRITICAL FIX: Prevents SIZE field values (A, B, C, D) from being mistaken for REV values
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

LOG = logging.getLogger("rev_extractor_v2")
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
    ". .", ". -", "- .", ". _", "_ .", ".E", ".F", ".A", ".B"
}

# Dot+letter pattern (common REV format)
DOT_LETTER_PATTERN = re.compile(r"^\.+[A-Z]{1,2}$", re.IGNORECASE)
DOT_NUMERIC_PATTERN = re.compile(r"^\.+\d{1,2}-\d{1,2}$", re.IGNORECASE)

def is_special_valid_rev(value: str) -> bool:
    """Check if value is a special valid REV marker."""
    if not value:
        return False
    if all(c in "-._–— " for c in value):
        return True
    if DOT_LETTER_PATTERN.match(value):
        return True
    if DOT_NUMERIC_PATTERN.match(value):
        return True
    return value in SPECIAL_VALID_REVS

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
    """Validate double letter REV: first letter should be A/B/C."""
    if not value or len(value) != 2:
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

def is_likely_size_value(value: str) -> bool:
    """
    Check if value is likely a SIZE field value (A, B, C, D, E).
    SIZE values are typically single letters A-E.
    """
    return bool(re.fullmatch(r"[A-E]", value, re.IGNORECASE))

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

# Enhanced REV pattern to include all observed formats
REV_VALUE_RE = re.compile(
    r"^(?:"
    r"[A-Z]{1,2}|"                    # Letters: A, AB
    r"\d{1,2}-\d{1,2}|"               # Numeric: 1-0, 2-0
    r"\.+[A-Z]{1,2}|"                 # Dot+letter: .E, .F, ..A
    r"\.+\d{1,2}-\d{1,2}|"            # Dot+numeric: .1-0
    r"[A-Z]{1,2}\.+|"                 # Letter+dot: E., F.
    r"[-._]{1,4}"                     # Special: -, _, .-, ._
    r")$",
    re.IGNORECASE
)

REV_TOKEN_RE = re.compile(r"^rev\.?$", re.IGNORECASE)
SIZE_TOKEN_RE = re.compile(r"^size\.?$", re.IGNORECASE)

TITLE_ANCHORS = {"DWG", "DWG.", "DWGNO", "SHEET", "SCALE", "WEIGHT", "DRAWN", "CHECKED"}
REV_TABLE_HEADERS = {
    "REVISIONS", "REVISION", "DESCRIPTION", "DESCRIPTIONS",
    "EC", "DFT", "APPR", "APPD", "DATE", "CHKD",
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
    return math.hypot(a[0] - b[0], a[1] - b[1])

def get_native_tokens(pdf_path: Path, page_idx: int, corner: str = "br") -> Tuple[List[Token], float, float]:
    """Extract tokens from specific corner."""
    tokens: List[Token] = []
    
    with fitz.open(pdf_path) as doc:
        page = doc[page_idx]
        rect = page.rect
        page_w, page_h = rect.width, rect.height
        
        # Define corner crop
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
    """Detect if token is in revision table."""
    if token.y > page_h * 0.5:
        return False
    nearby = [t for t in all_tokens if distance((t.x, t.y), (token.x, token.y)) <= 350]
    table_header_count = sum(1 for t in nearby if norm_val(t.text).upper() in REV_TABLE_HEADERS)
    return table_header_count >= 2

def extract_native_pymupdf_corner(
    pdf_path: Path,
    corner: str = "br",
    page_idx: int = 0
) -> Optional[RevResult]:
    """
    Extract REV with AGGRESSIVE SIZE field filtering.
    """
    try:
        tokens, page_w, page_h = get_native_tokens(pdf_path, page_idx, corner)
        
        if not tokens:
            return None
        
        # Find REV labels
        rev_labels = [t for t in tokens if REV_TOKEN_RE.match(norm_val(t.text))]
        
        # Find SIZE labels (CRITICAL: to avoid confusion)
        size_labels = [t for t in tokens if SIZE_TOKEN_RE.match(norm_val(t.text))]
        
        # Find anchors
        anchors = [t for t in tokens if norm_val(t.text).upper() in TITLE_ANCHORS]
        has_anchors = len(anchors) > 0
        
        if not has_anchors and corner != "br":
            return None
        
        candidates = []
        
        # Strategy 1: Look near REV labels
        if rev_labels:
            for rev_label in rev_labels:
                nearby = [t for t in tokens 
                         if distance((t.x, t.y), (rev_label.x, rev_label.y)) <= 200]
                
                for t in nearby:
                    v = norm_val(t.text)
                    
                    if not REV_VALUE_RE.match(v):
                        continue
                    
                    if is_in_revision_table(t, tokens, page_w, page_h):
                        continue
                    
                    # CRITICAL: Skip if near SIZE label
                    if size_labels:
                        near_size = any(distance((t.x, t.y), (s.x, s.y)) <= 150 
                                       for s in size_labels)
                        if near_size:
                            LOG.debug(f"Skipping '{v}' - within 150px of SIZE label")
                            continue
                    
                    # CRITICAL: Extra check for single letters A-E (likely SIZE values)
                    if is_likely_size_value(v) and size_labels:
                        # Check if this token is actually closer to SIZE than REV
                        closest_to_size = min((distance((t.x, t.y), (s.x, s.y)) for s in size_labels), default=999)
                        closest_to_rev = distance((t.x, t.y), (rev_label.x, rev_label.y))
                        
                        if closest_to_size < closest_to_rev:
                            LOG.debug(f"Skipping '{v}' - closer to SIZE ({closest_to_size:.0f}px) than REV ({closest_to_rev:.0f}px)")
                            continue
                    
                    if not is_special_valid_rev(v) and not DOT_LETTER_PATTERN.match(v) and v.upper() in DEFAULT_BLOCKLIST:
                        continue
                    
                    # Score
                    dist = distance((t.x, t.y), (rev_label.x, rev_label.y))
                    score = 100.0 / (dist + 1.0)
                    
                    # MASSIVE BOOST for dot+letter (very common, should beat SIZE)
                    if DOT_LETTER_PATTERN.match(v):
                        score += 100.0  # Increased from 50.0
                        LOG.debug(f"Found dot+letter REV '{v}' - applying +100 boost")
                    
                    # Bonus for same line
                    if abs(t.y - rev_label.y) <= max(t.h, rev_label.h) * 0.8:
                        score += 20.0
                    
                    # Bonus for right of label
                    if t.x > rev_label.x:
                        score += 10.0
                    
                    # Bonus for anchors nearby
                    anchor_count = sum(1 for a in anchors 
                                     if distance((t.x, t.y), (a.x, a.y)) <= 200)
                    score += anchor_count * 5.0
                    
                    candidates.append((score, v, f"{corner.upper()}: Near REV label, {anchor_count} anchors"))
        
        # Strategy 2: Look standalone near anchors (but AVOID SIZE area)
        else:
            for t in tokens:
                v = norm_val(t.text)
                
                if not REV_VALUE_RE.match(v):
                    continue
                
                if is_in_revision_table(t, tokens, page_w, page_h):
                    continue
                
                # CRITICAL: Aggressive SIZE filtering
                if size_labels:
                    near_size = any(distance((t.x, t.y), (s.x, s.y)) <= 150 
                                   for s in size_labels)
                    if near_size:
                        LOG.debug(f"Skipping '{v}' - near SIZE label in strategy 2")
                        continue
                
                # CRITICAL: Reject single letters A-E if SIZE label exists
                if is_likely_size_value(v) and size_labels:
                    LOG.debug(f"Skipping '{v}' - likely SIZE value (single letter A-E)")
                    continue
                
                if not is_special_valid_rev(v) and not DOT_LETTER_PATTERN.match(v) and v.upper() in DEFAULT_BLOCKLIST:
                    continue
                
                # Score based on anchors
                anchor_count = sum(1 for a in anchors 
                                 if distance((t.x, t.y), (a.x, a.y)) <= 200)
                
                if anchor_count > 0:
                    score = anchor_count * 10.0
                    
                    # MASSIVE BOOST for dot+letter
                    if DOT_LETTER_PATTERN.match(v):
                        score += 100.0
                        LOG.debug(f"Found standalone dot+letter REV '{v}' - applying +100 boost")
                    
                    candidates.append((score, v, f"{corner.upper()}: Near {anchor_count} anchors"))
        
        if not candidates:
            return None
        
        best_score, best_value, notes = max(candidates, key=lambda x: x[0])
        
        LOG.info(f"  {pdf_path.name} ({corner.upper()}): Found '{best_value}' (score={best_score:.1f})")
        
        return RevResult(
            file=pdf_path.name,
            value=best_value,
            engine=f"pymupdf_{corner}",
            confidence="high" if best_score > 50 else "medium",
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

⚠️ CRITICAL: DISTINGUISH REV FROM SIZE ⚠️

Many drawings have BOTH SIZE and REV fields:
- SIZE: Drawing size (single letter A, B, C, D, E near "SIZE:" label)
- REV: Revision number (near "REV:" label) - THIS IS WHAT YOU WANT!

**You MUST look for the "REV:" or "REV" label specifically - NOT the "SIZE:" label!**

VALID REV FORMATS:

1. DOT+LETTER (VERY COMMON):
   - .E, .F, .A, .B, .C (dot followed by letter)
   - ..A, ...E (multiple dots + letter)
   - INCLUDE THE DOT in your response!

2. NUMERIC REVISIONS:
   - 1-0, 2-0, 3-0, 9-0 (hyphenated, second number usually 0)
   - .1-0, .2-0 (dot + numeric)

3. LETTER REVISIONS:
   - A, B, C, D, E, F (standalone letters)
   - AA, AB, BA, BC (double letters, first usually A/B/C)

4. SPECIAL MARKERS:
   - -, _, .-, ._ (dashes, underscores, combinations)

5. NO REV:
   - Only if NO REV field exists or field is empty

TITLE BLOCK LOCATION:
- Check BOTTOM-RIGHT, BOTTOM-LEFT, TOP-RIGHT, TOP-LEFT
- Contains: DWG NO, SHEET, SCALE, SIZE, **REV** (separate fields!)

CRITICAL EXAMPLES:

Example 1 - DOT+LETTER (NOT SIZE):
"SIZE: B | DWG NO: EA-13855 | REV: .E | SHEET 1"
✅ Correct: {"rev_value": ".E", "notes": "Dot+letter REV .E (not SIZE B)"}
❌ WRONG: {"rev_value": "B"} ← This is SIZE, not REV!
❌ WRONG: {"rev_value": "E"} ← Missing the dot!

Example 2 - ANOTHER DOT+LETTER:
"SIZE: A | DWG NO: EA-10840 | REV: .F | SCALE 5:1"
✅ Correct: {"rev_value": ".F", "notes": "Dot+letter REV .F (not SIZE A)"}
❌ WRONG: {"rev_value": "A"} ← This is SIZE, not REV!
❌ WRONG: {"rev_value": "F"} ← Missing the dot!

Example 3 - REGULAR LETTER:
"SIZE: C | REV: D | SHEET 1"
✅ Correct: {"rev_value": "D", "notes": "Letter REV D (not SIZE C)"}

Example 4 - NUMERIC:
"DWG NO: 21620 | REV: 2-0 | SIZE: A"
✅ Correct: {"rev_value": "2-0", "notes": "Numeric REV 2-0"}

RESPONSE FORMAT:
{
  "rev_value": ".E",
  "confidence": "high",
  "location": "bottom-right, REV field",
  "notes": "Dot+letter REV .E in REV field (not SIZE value)"
}

REMEMBER:
- Look for "REV:" label, NOT "SIZE:" label!
- Dot+letter (.E, .F) is VERY COMMON - include the dot!
- Single letters A-E near "SIZE:" are SIZE values, not REV!
- If unsure between SIZE and REV, always choose the one near "REV:" label
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
                                "text": f"Extract REV from {corner.upper()} corner. CRITICAL: Look for 'REV:' label (NOT 'SIZE:' label). Dot+letter (.E, .F) is VERY COMMON - include the dot!"
                            },
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                        ]
                    }
                ],
                max_tokens=400,
                temperature=0
            )
            
            result_text = response.choices[0].message.content or ""
            
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', result_text, re.DOTALL | re.IGNORECASE)
            if json_match:
                result_text = json_match.group(1)
            
            result_data = json.loads(result_text.strip())
            
            value = (result_data.get("rev_value") or "").strip()
            notes = (result_data.get("notes") or "").strip()
            confidence = (result_data.get("confidence") or "unknown").lower()
            
            if value.upper() in {"NO_REV", "EMPTY", "N/A", "NA"}:
                value = "NO_REV"
            
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
        is_valid, flag, reason = validate_rev_value(native_result.value, "pymupdf")
        
        if is_valid:
            return native_result
        else:
            LOG.info(f"  Native returned suspicious '{native_result.value}': {reason}")
            LOG.info(f"  Rerunning with GPT...")
            
            corner = "br"
            if "_" in native_result.engine:
                corner = native_result.engine.split("_")[-1]
            
            gpt_result = gpt.extract_rev(pdf_path, corner=corner)
            gpt_valid, gpt_flag, gpt_reason = validate_rev_value(gpt_result.value, "gpt")
            
            if gpt_valid:
                gpt_result.notes = f"Native={native_result.value} ({reason}), GPT corrected"
                return gpt_result
            elif native_result.value == gpt_result.value:
                native_result.review_flag = flag or ReviewFlag.BOTH_ENGINES_DISAGREE
                native_result.validation_note = f"REVIEW: {reason}"
                return native_result
            else:
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
    
    stats = {"native": 0, "gpt": 0, "failed": 0, "flagged": 0}
    
    for pdf_path in tqdm(pdfs, desc="Processing"):
        try:
            result = smart_extraction(pdf_path, gpt)
            
            if "pymupdf" in result.engine:
                stats["native"] += 1
            elif "gpt" in result.engine:
                stats["gpt"] += 1
            if result.review_flag != ReviewFlag.NONE:
                stats["flagged"] += 1
            
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
    LOG.info(f"Flagged: {stats['flagged']} | Failed: {stats['failed']}")
    LOG.info(f"Cost: ${stats['gpt'] * 0.01:.2f}")
    LOG.info(f"{'='*60}\n")
    
    return rows

def main():
    start = time.time()
    
    p = argparse.ArgumentParser(description="Enhanced REV Extractor with SIZE/REV Distinction")
    p.add_argument("input_folder", type=Path)
    p.add_argument("-o", "--output", type=Path, default=Path("rev_results_v2.csv"))
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
