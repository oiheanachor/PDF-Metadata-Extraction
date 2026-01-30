#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REV Extractor — Dual-Extraction Enhanced (v3.0)
- Extracts from BOTH title block AND revision table
- Handles all 7 validated edge cases
- Optimized for speed with 2 workers max
- Updated validation logic (.A, .B are valid)
"""

from __future__ import annotations
import argparse, base64, csv, logging, os, re, json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz  # PyMuPDF
from tqdm import tqdm

# Import validation from enhanced fixed script
try:
    from rev_extractor_fixed_v2 import (
        process_pdf_native, DEFAULT_BR_X, DEFAULT_BR_Y, 
        DEFAULT_EDGE_MARGIN, DEFAULT_REV_2L_BLOCKLIST
    )
    NATIVE_AVAILABLE = True
except ImportError:
    NATIVE_AVAILABLE = False

# Azure OpenAI SDK
try:
    from openai import AzureOpenAI
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

LOG = logging.getLogger("rev_extractor_v3")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ---------------------------------------------------------------------
# ENHANCED VALIDATION (Updated for .A, .B patterns)
# ---------------------------------------------------------------------

def is_special_char(s: str) -> bool:
    """Check if value is special character indicating no revision."""
    s_upper = s.upper() if s else ""
    # Direct special characters
    if re.fullmatch(r"[-_.]+|\.[-_]+", s):
        return True
    # Empty indicators
    if s_upper in {"EMPTY", "NO_REV", "N/A", "NA"}:
        return True
    return False

def canonicalize_rev_value(v: str) -> str:
    """Canonicalize REV values."""
    s = norm_val(v)
    if s in {"", "NO_REV", "NONE", "N/A"}:
        return "NO_REV"
    # Normalize special characters
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
    """
    Enhanced domain validation.
    Returns True if valid, False if suspicious.
    
    NEW: .A, .B, .AB are VALID (common notation for revisions)
    """
    s = canonicalize_rev_value(v)
    
    if s == "NO_REV":
        return True
    
    # NEW: Special character prefixes with letters are VALID
    # .A, .B, .AB, .C, etc.
    if re.fullmatch(r"\.[A-Z]{1,2}", s):
        return True
    
    # Pure special characters need verification but are plausible
    if s in {"-", "_", ".-", "._"}:
        return True  # Changed from False - these are valid "no revision"
    
    # Single/double letters
    if re.fullmatch(r"[A-Z]{1,2}", s):
        if len(s) == 2:
            # Double letters: first letter should be A, B, or C
            return s[0] in {"A", "B", "C"}
        return True
    
    # Numeric hyphenated: should end with -0
    m = re.fullmatch(r"(\d{1,3})-(\d{1,3})", s)
    if m:
        return m.group(2) == "0"  # Must be X-0 format
    
    return False

def is_suspicious_rev_value(v: str) -> bool:
    """
    Enhanced suspicion detection.
    
    NEW: Catches page sizes (A4, A3) and single digits
    NEW: .A, .B, .AB are NOT suspicious
    """
    s = norm_val(v)
    
    # NEW: Page/paper sizes are HIGHLY suspicious (Case 7)
    # A0-A5, B, C, D, E (when standalone - could be SIZE field)
    if re.fullmatch(r"A[0-5]|[BCDE]", s.upper()):
        return True
    
    # Single numeric (Case 5)
    if re.fullmatch(r"\d", s):
        return True
    
    # Rotation tokens
    if s.upper() in {"LTR", "RTL"}:
        return True
    
    # NEW: Special char prefixes with letters are VALID - NOT suspicious
    if re.fullmatch(r"\.[A-Z]{1,2}", s.upper()):
        return False
    
    # Pure special chars are NOT suspicious anymore (valid "no revision")
    if is_special_char(s):
        return False
    
    s2 = canonicalize_rev_value(s)
    return bool(not is_plausible_rev_value(s2))

def norm_val(v: Any) -> str:
    """Normalize value to string."""
    if v is None:
        return ""
    s = str(v).replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _normalize_output_value(v: str) -> str:
    """
    Normalize output value.
    
    NEW: Strip leading special characters from valid letter patterns
    .A → A, .AB → AB, .B → B
    """
    vu = norm_val(v).upper()
    
    # Handle "OF" special case
    if vu == "OF":
        return "EMPTY"
    
    # NEW: Strip leading dots/dashes from valid letter patterns
    # .A → A, .AB → AB, .- → -
    if re.match(r"^[._-]+([A-Z]{1,2})$", vu):
        # Extract just the letters
        letters = re.sub(r"^[._-]+", "", vu)
        return letters
    
    # Keep special characters as-is (-, _, etc.)
    if is_special_char(vu):
        return canonicalize_rev_value(vu)
    
    return norm_val(v)

# ---------------------------------------------------------------------
# DUAL EXTRACTION GPT PROMPT
# ---------------------------------------------------------------------
GPT_SYSTEM_PROMPT = """You are an expert at analyzing engineering drawings and extracting revision information.

YOUR TASK:
Extract the REV (revision) value from TWO locations in this engineering drawing.

CRITICAL: Extract from BOTH locations:

**Location 1: Title Block (bottom-right corner)**
- Near DWG NO, SHEET, SCALE, SIZE fields
- Look for "REV:" or "REV" label
- DO NOT confuse with SIZE field (single letters D, B, C, E near "SIZE:")
- DO NOT confuse with page format (A4, A3)

**Location 2: Revision Table (top-right corner)**
- Table with "REVISIONS" or "REVISION HISTORY" header
- Columns typically: REV | DESCRIPTION | EC | DFT | DATE | APPR
- **EXTRACT BOTTOM ROW ONLY** (most recent = current revision)
- If table has multiple rows, bottom row is the current REV
- If no table exists, return null

VALID REV FORMATS:

1. **Numeric**: X-0 only (1-0, 2-0, 12-0)
   - NEVER X-Y where Y is not zero

2. **Letters**: Single (A-Z) or double (AA, AB, BA, BC)
   - Double letters MUST start with A, B, or C
   - May have leading dot: .A, .B, .AB (valid notation)

3. **Special characters** (valid for "no revision"):
   - Single: -, _
   - With dot: .-, ._
   - Multiple: __, ___

INVALID PATTERNS TO AVOID:
- Single digits: 1, 2, 9 (likely sheet/page numbers)
- Page sizes: A4, A3, B, C, D, E (when near SIZE field)
- Non-zero hyphenated: 5-40, 18-8
- Double letters starting D+: DE, DF

RESPONSE FORMAT (always return both extractions):
{
  "title_block_rev": "value or null",
  "revision_table_rev": "value or null",
  "revision_table_exists": true/false,
  "confidence": "high/medium/low",
  "notes": "explanation of what you found"
}

EXAMPLES:
{
  "title_block_rev": "BC",
  "revision_table_rev": "BC",
  "revision_table_exists": true,
  "confidence": "high",
  "notes": "Both sources agree on BC"
}

{
  "title_block_rev": "D",
  "revision_table_rev": null,
  "revision_table_exists": false,
  "confidence": "medium",
  "notes": "Only title block available, D found near REV label"
}

{
  "title_block_rev": "0",
  "revision_table_rev": "D",
  "revision_table_exists": true,
  "confidence": "low",
  "notes": "Title block unclear (looks like 0), table shows D"
}"""

# ---------------------------------------------------------------------
# DATA CLASSES
# ---------------------------------------------------------------------
@dataclass
class RevResult:
    file: str
    value: str
    engine: str
    confidence: str = "unknown"
    notes: str = ""
    human_review: bool = False
    review_reason: str = ""

# ---------------------------------------------------------------------
# NATIVE EXTRACTION (with updated validation)
# ---------------------------------------------------------------------
def extract_native_pymupdf(pdf_path: Path) -> Optional[RevResult]:
    """Try native PyMuPDF extraction with updated validation."""
    if not NATIVE_AVAILABLE:
        return None
        
    try:
        best = process_pdf_native(
            pdf_path,
            brx=DEFAULT_BR_X,
            bry=DEFAULT_BR_Y,
            blocklist=DEFAULT_REV_2L_BLOCKLIST,
            edge_margin=DEFAULT_EDGE_MARGIN
        )
        
        if best and best.value:
            value = _normalize_output_value(best.value)
            
            # Check if suspicious with updated logic
            is_susp = is_suspicious_rev_value(value)
            is_plaus = is_plausible_rev_value(value)
            
            return RevResult(
                file=pdf_path.name,
                value=value,
                engine=f"pymupdf_{best.engine}",
                confidence="high" if best.score > 100 and is_plaus else "medium",
                notes=best.context_snippet,
                human_review=is_susp,
                review_reason="suspicious_value" if is_susp else ""
            )
        return None
    except Exception as e:
        LOG.debug(f"Native extraction failed for {pdf_path.name}: {e}")
        return None

# ---------------------------------------------------------------------
# DUAL EXTRACTION RESOLUTION
# ---------------------------------------------------------------------
def resolve_dual_extraction(gpt_result: dict, pdf_path: Path) -> RevResult:
    """
    Resolve REV value from dual extraction results.
    Handles all 7 edge cases.
    """
    tb_rev = gpt_result.get("title_block_rev")
    rt_rev = gpt_result.get("revision_table_rev")
    table_exists = gpt_result.get("revision_table_exists", False)
    gpt_confidence = gpt_result.get("confidence", "unknown").lower()
    gpt_notes = gpt_result.get("notes", "")
    
    # Normalize values
    tb_rev = _normalize_output_value(tb_rev) if tb_rev else ""
    rt_rev = _normalize_output_value(rt_rev) if rt_rev else ""
    
    # CASE 1: Both exist and agree → BEST outcome
    if tb_rev and rt_rev and tb_rev == rt_rev:
        return RevResult(
            file=pdf_path.name,
            value=tb_rev,
            engine="gpt_dual_agree",
            confidence="high",
            notes=f"Both sources confirm: {tb_rev}",
            human_review=False
        )
    
    # CASE 2: Both exist but disagree
    if tb_rev and rt_rev and tb_rev != rt_rev:
        # Check plausibility of both
        tb_plausible = is_plausible_rev_value(tb_rev)
        rt_plausible = is_plausible_rev_value(rt_rev)
        
        tb_suspicious = is_suspicious_rev_value(tb_rev)
        rt_suspicious = is_suspicious_rev_value(rt_rev)
        
        # Prefer revision table (cleaner, handles Cases 1, 3, 4, 6)
        if rt_plausible and not rt_suspicious:
            return RevResult(
                file=pdf_path.name,
                value=rt_rev,
                engine="gpt_dual_table_preferred",
                confidence="high" if not tb_suspicious else "medium",
                notes=f"Table={rt_rev}, Title={tb_rev}. Using table (cleaner format).",
                human_review=False
            )
        
        # Fallback to title block if table suspicious
        if tb_plausible and not tb_suspicious:
            return RevResult(
                file=pdf_path.name,
                value=tb_rev,
                engine="gpt_dual_title_preferred",
                confidence="medium",
                notes=f"Table suspicious, using title block: {tb_rev}",
                human_review=True,
                review_reason="sources_disagree"
            )
        
        # Both suspicious → flag both
        return RevResult(
            file=pdf_path.name,
            value=rt_rev if rt_plausible else tb_rev,
            engine="gpt_dual_both_suspicious",
            confidence="low",
            notes=f"Both suspicious: Table={rt_rev}, Title={tb_rev}",
            human_review=True,
            review_reason="both_suspicious"
        )
    
    # CASE 3: Only revision table exists (Case 2 - rotated drawings)
    if rt_rev and not tb_rev:
        is_susp = is_suspicious_rev_value(rt_rev)
        return RevResult(
            file=pdf_path.name,
            value=rt_rev,
            engine="gpt_table_only",
            confidence="high" if not is_susp else "medium",
            notes=f"Table only: {rt_rev}. {gpt_notes}",
            human_review=is_susp,
            review_reason="suspicious_value" if is_susp else ""
        )
    
    # CASE 4: Only title block exists (standard case)
    if tb_rev and not rt_rev:
        is_susp = is_suspicious_rev_value(tb_rev)
        return RevResult(
            file=pdf_path.name,
            value=tb_rev,
            engine="gpt_title_only",
            confidence="medium" if not is_susp else "low",
            notes=f"Title only: {tb_rev}. No revision table found.",
            human_review=is_susp,
            review_reason="suspicious_value" if is_susp else ""
        )
    
    # CASE 5: Both empty (Case 5 - truly no revision)
    if not tb_rev and not rt_rev:
        if table_exists:
            # Table exists but is empty → confirmed no revision
            return RevResult(
                file=pdf_path.name,
                value="EMPTY",
                engine="gpt_both_empty_confirmed",
                confidence="high",
                notes="Revision table exists but is empty (no revision assigned)",
                human_review=False
            )
        else:
            # No table, no title block value → flag
            return RevResult(
                file=pdf_path.name,
                value="",
                engine="gpt_no_sources",
                confidence="none",
                notes="No REV found in either location",
                human_review=True,
                review_reason="no_rev_found"
            )

# ---------------------------------------------------------------------
# GPT EXTRACTOR (Dual Extraction)
# ---------------------------------------------------------------------
class AzureGPTExtractor:
    def __init__(self, endpoint: str, api_key: str, deployment_name: str = "gpt-4.1", dpi: int = 100):
        if not AZURE_AVAILABLE:
            raise ImportError("openai not installed. Run: pip install openai")
        
        endpoint = endpoint.rstrip('/')
        if '/openai/deployments' in endpoint:
            endpoint = endpoint.split('/openai/deployments')[0]
        
        LOG.info(f"Initializing GPT client with endpoint: {endpoint}")
        
        try:
            self.client = AzureOpenAI(
                api_key=api_key,
                api_version="2024-02-15-preview",
                azure_endpoint=endpoint,
                timeout=60.0,
                max_retries=3  # SDK handles retries
            )
            self.deployment_name = deployment_name
            self.dpi = dpi  # Configurable DPI for speed optimization
            LOG.info(f"✓ GPT client initialized (DPI: {dpi}, SDK retries enabled)")
        except Exception as e:
            LOG.error(f"Failed to initialize GPT client: {e}")
            raise
    
    def pdf_to_base64_image(self, pdf_path: Path, page_idx: int = 0) -> str:
        """
        Convert PDF to base64 image.
        
        OPTIMIZATION: Captures FULL PAGE now (not just bottom-right)
        since we need BOTH title block AND revision table.
        """
        with fitz.open(pdf_path) as doc:
            page = doc[page_idx]
            
            # OPTIMIZATION: Use full page at lower DPI for speed
            zoom = self.dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            png_bytes = pix.tobytes("png")
            return base64.b64encode(png_bytes).decode('utf-8')

    def extract_rev(self, pdf_path: Path) -> RevResult:
        """Extract REV using dual extraction with GPT-4 Vision."""
        try:
            LOG.debug(f"Converting {pdf_path.name} to image (DPI: {self.dpi})...")
            img_base64 = self.pdf_to_base64_image(pdf_path)
            
            LOG.debug(f"Sending to GPT API (dual extraction)...")
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": GPT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Extract REV from BOTH title block (bottom-right) "
                                    "AND revision table (top-right, bottom row). "
                                    "Return JSON with both values."
                                )
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                            }
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0
            )
            
            result_text = response.choices[0].message.content or ""
            LOG.debug(f"GPT response: {result_text[:200]}...")
            
            # Parse JSON
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', result_text, re.DOTALL | re.IGNORECASE)
            if json_match:
                result_text = json_match.group(1)
            
            result_data = json.loads(result_text.strip())
            
            # Resolve dual extraction
            return resolve_dual_extraction(result_data, pdf_path)
            
        except json.JSONDecodeError as e:
            LOG.error(f"JSON parse error for {pdf_path.name}: {e}")
            LOG.error(f"Raw response: {result_text[:500]}")
            return RevResult(
                file=pdf_path.name,
                value="",
                engine="gpt_json_error",
                confidence="none",
                notes=f"JSON parse failed: {str(e)[:100]}",
                human_review=True,
                review_reason="json_error"
            )
        except Exception as e:
            LOG.error(f"GPT extraction failed for {pdf_path.name}: {e}")
            return RevResult(
                file=pdf_path.name,
                value="",
                engine="gpt_failed",
                confidence="none",
                notes=str(e)[:100],
                human_review=True,
                review_reason="gpt_error"
            )

# ---------------------------------------------------------------------
# COMPARISON LOGIC (updated for dual extraction)
# ---------------------------------------------------------------------
def compare_and_decide(
    native_result: Optional[RevResult],
    gpt_result: RevResult,
    pdf_path: Path
) -> RevResult:
    """
    Compare PyMuPDF and GPT results.
    GPT now does dual extraction, so it's more reliable.
    """
    
    # No native result → use GPT
    if not native_result:
        return gpt_result
    
    native_val = native_result.value
    gpt_val = gpt_result.value
    
    native_plausible = is_plausible_rev_value(native_val)
    gpt_plausible = is_plausible_rev_value(gpt_val)
    
    # Both agree
    if native_val == gpt_val:
        return RevResult(
            file=pdf_path.name,
            value=native_val,
            engine="pymupdf+gpt_agree",
            confidence="high",
            notes=f"Both engines agree: {native_val}",
            human_review=False,
            review_reason=""
        )
    
    # Both valid - prefer GPT (has dual extraction advantage)
    if native_plausible and gpt_plausible:
        return RevResult(
            file=pdf_path.name,
            value=gpt_val,
            engine="gpt+pymupdf_differ",
            confidence="medium",
            notes=f"Chose GPT {gpt_val} over PyMuPDF {native_val} (dual extraction)",
            human_review=True,
            review_reason="engines_disagree"
        )
    
    # One valid
    if gpt_plausible and not native_plausible:
        return RevResult(
            file=pdf_path.name,
            value=gpt_val,
            engine="gpt_valid",
            confidence="high",
            notes=f"GPT validated: {gpt_val} (PyMuPDF suspicious: {native_val})",
            human_review=False,
            review_reason=""
        )
    
    if native_plausible and not gpt_plausible:
        return RevResult(
            file=pdf_path.name,
            value=native_val,
            engine="pymupdf_valid",
            confidence="medium",
            notes=f"PyMuPDF validated: {native_val} (GPT suspicious: {gpt_val})",
            human_review=False,
            review_reason=""
        )
    
    # Both invalid
    return RevResult(
        file=pdf_path.name,
        value=gpt_val,  # Default to GPT (has more context)
        engine="both_invalid",
        confidence="low",
        notes=f"Both suspicious: PyMuPDF={native_val}, GPT={gpt_val}",
        human_review=True,
        review_reason="both_invalid"
    )

# ---------------------------------------------------------------------
# HYBRID PIPELINE (optimized)
# ---------------------------------------------------------------------
def run_hybrid_pipeline(
    input_folder: Path,
    output_csv: Path,
    azure_endpoint: str,
    azure_key: str,
    deployment_name: str = "gpt-4.1",
    disable_gpt: bool = False,
    max_workers: int = 2,  # Default 2 for safety
    dpi: int = 100  # NEW: Configurable DPI
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Enhanced hybrid pipeline with dual extraction and speed optimization."""
    rows: List[Dict[str, Any]] = []
    
    if not disable_gpt:
        LOG.info("Initializing Azure GPT client with dual extraction...")
        gpt = AzureGPTExtractor(azure_endpoint, azure_key, deployment_name, dpi)
    else:
        gpt = None
        LOG.info("GPT disabled - using PyMuPDF only")
        max_workers = 1
    
    pdfs = list(input_folder.glob("*.pdf"))
    if not pdfs:
        LOG.warning(f"No PDFs found in {input_folder}")
        return rows, {}
    
    stats = {
        "total": len(pdfs),
        "native_only": 0,
        "native_suspicious": 0,
        "gpt_used": 0,
        "human_review": 0,
        "both_agree": 0,
        "both_differ": 0,
        "dual_agree": 0,  # NEW: Dual extraction agreement
        "table_used": 0   # NEW: Revision table used
    }
    
    def process_single_pdf(pdf_path: Path) -> Tuple[Dict[str, Any], Dict[str, int]]:
        """Process a single PDF."""
        local_stats = {
            "native_only": 0,
            "native_suspicious": 0,
            "gpt_used": 0,
            "human_review": 0,
            "both_agree": 0,
            "both_differ": 0,
            "dual_agree": 0,
            "table_used": 0
        }
        
        try:
            # Step 1: Try PyMuPDF native
            native_result = extract_native_pymupdf(pdf_path)
            
            # Step 2: Decide if GPT needed
            needs_gpt = False
            if not native_result or not native_result.value:
                needs_gpt = True
            elif native_result.human_review:
                needs_gpt = True
                local_stats["native_suspicious"] += 1
            
            # Step 3: Use GPT if needed
            if needs_gpt and not disable_gpt:
                local_stats["gpt_used"] += 1
                gpt_result = gpt.extract_rev(pdf_path)
                
                # Track dual extraction stats
                if "dual_agree" in gpt_result.engine:
                    local_stats["dual_agree"] += 1
                if "table" in gpt_result.engine:
                    local_stats["table_used"] += 1
                
                final_result = compare_and_decide(native_result, gpt_result, pdf_path)
            elif native_result:
                final_result = native_result
                local_stats["native_only"] += 1
            else:
                final_result = RevResult(
                    file=pdf_path.name,
                    value="",
                    engine="failed",
                    confidence="none",
                    notes="No extraction succeeded",
                    human_review=True,
                    review_reason="extraction_failed"
                )
            
            # Track stats
            if final_result.human_review:
                local_stats["human_review"] += 1
            if "agree" in final_result.engine:
                local_stats["both_agree"] += 1
            elif "differ" in final_result.engine:
                local_stats["both_differ"] += 1
            
            row = {
                "file": final_result.file,
                "value": final_result.value,
                "engine": final_result.engine,
                "confidence": final_result.confidence,
                "human_review": "yes" if final_result.human_review else "no",
                "review_reason": final_result.review_reason,
                "notes": final_result.notes
            }
            
            return (row, local_stats)
            
        except Exception as e:
            LOG.error(f"Failed {pdf_path.name}: {e}")
            row = {
                "file": pdf_path.name,
                "value": "",
                "engine": "error",
                "confidence": "none",
                "human_review": "yes",
                "review_reason": "processing_error",
                "notes": str(e)[:100]
            }
            local_stats["human_review"] += 1
            return (row, local_stats)
    
    # Process PDFs
    LOG.info(f"Processing {len(pdfs)} PDFs with {max_workers} workers, DPI={dpi}...")
    
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single_pdf, pdf): pdf for pdf in pdfs}
            
            for future in tqdm(as_completed(futures), total=len(pdfs), desc="Processing"):
                try:
                    row, local_stats = future.result()
                    rows.append(row)
                    for key in local_stats:
                        stats[key] += local_stats[key]
                except Exception as e:
                    pdf = futures[future]
                    LOG.error(f"Future failed for {pdf.name}: {e}")
    else:
        for pdf_path in tqdm(pdfs, desc="Processing"):
            row, local_stats = process_single_pdf(pdf_path)
            rows.append(row)
            for key in local_stats:
                stats[key] += local_stats[key]
    
    # Write CSV
    try:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, 'w', newline='', encoding='utf-8-sig') as outf:
            writer = csv.DictWriter(outf, fieldnames=[
                'file', 'value', 'engine', 'confidence', 'human_review', 'review_reason', 'notes'
            ])
            writer.writeheader()
            writer.writerows(rows)
        
        LOG.info(f"\n{'='*60}")
        LOG.info(f"Results: {output_csv.resolve()}")
        LOG.info(f"{'='*60}")
        LOG.info(f"Total PDFs: {stats['total']}")
        LOG.info(f"Native only: {stats['native_only']} ({stats['native_only']/stats['total']*100:.1f}%)")
        LOG.info(f"GPT used: {stats['gpt_used']} ({stats['gpt_used']/stats['total']*100:.1f}%)")
        LOG.info(f"  - Dual sources agree: {stats['dual_agree']}")
        LOG.info(f"  - Revision table used: {stats['table_used']}")
        LOG.info(f"Both engines agree: {stats['both_agree']}")
        LOG.info(f"Human review needed: {stats['human_review']} ({stats['human_review']/stats['total']*100:.1f}%)")
        if stats['gpt_used'] > 0:
            LOG.info(f"Estimated cost: ${stats['gpt_used'] * 0.010:.2f}")
        LOG.info(f"{'='*60}\n")
        
    except Exception as e:
        LOG.error(f"Failed to write CSV: {e}")
    
    return rows, stats

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args(argv=None):
    a = argparse.ArgumentParser(description="Dual-extraction REV extractor (v3.0)")
    a.add_argument("input_folder", type=Path)
    a.add_argument("-o", "--output", type=Path, default=Path("rev_results_v3.csv"))
    a.add_argument("--azure-endpoint", type=str, default=os.getenv("AZURE_OPENAI_ENDPOINT"))
    a.add_argument("--azure-key", type=str, default=os.getenv("AZURE_OPENAI_KEY"))
    a.add_argument("--deployment-name", type=str, default="gpt-4.1")
    a.add_argument("--disable-gpt", action="store_true", help="Use PyMuPDF only")
    a.add_argument("--gpt-only", action="store_true", help="Force GPT-only mode")
    a.add_argument("--max-workers", type=int, default=2, 
                   help="Parallel workers (default: 2 to avoid 429 errors)")
    a.add_argument("--dpi", type=int, default=100,
                   help="Image DPI (default: 100, lower=faster)")
    return a.parse_args(argv)

def main(argv=None):
    start_time = time.time()
    args = parse_args(argv)
    
    if getattr(args, 'gpt_only', False):
        global NATIVE_AVAILABLE
        NATIVE_AVAILABLE = False
        args.disable_gpt = False
        LOG.info("GPT-only mode enabled")
    
    if not args.disable_gpt and (not args.azure_endpoint or not args.azure_key):
        LOG.error("❌ Azure credentials required (or use --disable-gpt)!")
        return []
    
    results, stats = run_hybrid_pipeline(
        args.input_folder,
        args.output,
        args.azure_endpoint or "",
        args.azure_key or "",
        args.deployment_name,
        args.disable_gpt,
        args.max_workers,
        args.dpi  # NEW
    )

    end_time = time.time()
    total_seconds = end_time - start_time
    mins = int(total_seconds // 60)
    secs = int(total_seconds % 60)
    LOG.info(f"✓ Completed in {mins}m {secs}s")

    return results

if __name__ == "__main__":
    main()
