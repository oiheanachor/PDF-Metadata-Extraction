#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REV Extractor — Enhanced PyMuPDF + Azure GPT-4 with Validation
Handles edge cases: invalid numerics, double letters, single numerics, special chars
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
        process_pdf_native, _normalize_output_value,
        is_plausible_rev_value, is_suspicious_rev_value, canonicalize_rev_value,
        is_special_char,  # NEW: For special character detection
        DEFAULT_BR_X, DEFAULT_BR_Y, DEFAULT_EDGE_MARGIN, DEFAULT_REV_2L_BLOCKLIST
    )
    NATIVE_AVAILABLE = True
except ImportError:
    NATIVE_AVAILABLE = False
    LOG.warning("Could not import rev_extractor_fixed - native extraction disabled")

# Azure OpenAI SDK
try:
    from openai import AzureOpenAI
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

LOG = logging.getLogger("rev_extractor_enhanced")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ---------------------------------------------------------------------
# ENHANCED GPT SYSTEM PROMPT (with special character support)
# ---------------------------------------------------------------------
GPT_SYSTEM_PROMPT = """You are an expert at analyzing engineering drawings and extracting revision information.

YOUR TASK:
Extract the REV (revision) value from the title block of this engineering drawing.

CRITICAL RULES:

1. Title Block Location:
   - The REV value is in the TITLE BLOCK, typically in the BOTTOM-RIGHT corner
   - Title blocks contain: DWG NO, SHEET, SCALE, DRAWN BY, CHECKED BY, APPROVED BY
   - Usually has company logo/name (ROTORK, FAIRCHILD, etc.)

2. Avoid These Common Mistakes:
   - DO NOT extract from REVISION TABLES (top-right, shows history: REV A | DATE | DESCRIPTION)
   - DO NOT extract grid reference letters (A, B, C along edges)
   - DO NOT extract part numbers or item callouts
   - DO NOT extract section markers (e.g., "SECTION C-C")

3. REV Value Formats (in STRICT priority order):

   NUMERIC REVISIONS:
   - Format: X-0 only (e.g., 1-0, 2-0, 3-0, 12-0)
   - NEVER return X-Y where Y is not zero (5-40, 18-8 are INVALID)

   LETTER REVISIONS:
   - Single letter: A, B, C, ... Z
   - Double letters MUST start with A, B, or C: AA, AB, BA, BC, CE
   - NEVER return double letters starting with D or higher (DE, DF, FF are INVALID)

   SPECIAL CHARACTERS (VALID for empty/no revision):
   - Single dash: -
   - Single underscore: _
   - Dash with dot: .-
   - Underscore with dot: ._
   - Multiple underscores: __, ___
   - These indicate "no revision" or "not applicable"

   INVALID FORMATS TO AVOID:
   - Single/multiple bare numbers (1, 2, 202) - INVALID
   - Non-zero hyphenated (5-40, 18-8) - INVALID
   - Double letters D+ (DE, DF, FF) - INVALID

4. Validation Checklist:
   - Is it in the title block (bottom corner)?
   - Is there a "REV:" or "REV" label nearby?
   - Does it follow a VALID format?
   - If numeric, does it end with -0?
   - If double letter, does it start with A, B, or C?

RESPONSE FORMAT:
Return ONLY a JSON object like:
{
  "rev_value": "2-0",
  "confidence": "high",
  "location": "bottom-right title block",
  "notes": "Clear numeric REV 2-0 ending in zero"
}

OR for special characters:
{
  "rev_value": "-",
  "confidence": "high",
  "location": "bottom-right title block",
  "notes": "Dash indicates no revision"
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
# NATIVE EXTRACTION (Enhanced with validation)
# ---------------------------------------------------------------------
def extract_native_pymupdf(pdf_path: Path) -> Optional[RevResult]:
    """Try native PyMuPDF extraction."""
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
            
            # Check if suspicious
            is_suspicious = is_suspicious_rev_value(value)
            is_plausible = is_plausible_rev_value(value)
            
            return RevResult(
                file=pdf_path.name,
                value=value,
                engine=f"pymupdf_{best.engine}",
                confidence="high" if best.score > 100 and is_plausible else "medium",
                notes=best.context_snippet,
                human_review=is_suspicious or not is_plausible,
                review_reason="suspicious_value" if is_suspicious else ("implausible_value" if not is_plausible else "")
            )
        return None
    except Exception as e:
        LOG.debug(f"Native extraction failed for {pdf_path.name}: {e}")
        return None

# ---------------------------------------------------------------------
# GPT EXTRACTOR (Enhanced)
# ---------------------------------------------------------------------
class AzureGPTExtractor:
    def __init__(self, endpoint: str, api_key: str, deployment_name: str = "gpt-4.1"):
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
                azure_endpoint=endpoint
            )
            self.deployment_name = deployment_name
            LOG.info("✓ GPT client initialized successfully")
        except Exception as e:
            LOG.error(f"Failed to initialize GPT client: {e}")
            raise
    
    def pdf_to_base64_image(self, pdf_path: Path, page_idx: int = 0, dpi: int = 100) -> str:
        """
        Convert PDF page to base64-encoded PNG (focused on title block).
        
        Uses 100 DPI for balance between quality and speed.
        Higher DPI = better quality but slower processing and higher cost.
        """
        with fitz.open(pdf_path) as doc:
            page = doc[page_idx]
            rect = page.rect
            # Crop to bottom-right quarter
            crop = fitz.Rect(
                rect.x0 + rect.width * 0.5,
                rect.y0 + rect.height * 0.5,
                rect.x1,
                rect.y1,
            )
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False, clip=crop)
            png_bytes = pix.tobytes("png")
            return base64.b64encode(png_bytes).decode('utf-8')

    def extract_rev(self, pdf_path: Path) -> RevResult:
        """Extract REV using GPT-4 Vision with validation."""
        try:
            LOG.debug(f"Converting {pdf_path.name} to image...")
            img_base64 = self.pdf_to_base64_image(pdf_path)
            
            LOG.debug(f"Sending to GPT API...")
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
                                    "Extract the REV value from this engineering drawing. "
                                    "Focus on the TITLE BLOCK. Follow validation rules strictly. "
                                    "Return ONLY the JSON object."
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
            
            raw_value = result_data.get("rev_value", "")
            value = canonicalize_rev_value(raw_value)
            confidence = (result_data.get("confidence") or "unknown").lower()
            notes = (result_data.get("notes") or "").strip()
            
            # Normalize NO_REV -> EMPTY
            if value == "NO_REV":
                value = "EMPTY"
            
            # Validate result
            is_plausible = is_plausible_rev_value(value)
            is_suspicious = is_suspicious_rev_value(value)
            
            return RevResult(
                file=pdf_path.name,
                value=value,
                engine="gpt_vision",
                confidence=confidence,
                notes=notes[:100],
                human_review=is_suspicious or not is_plausible,
                review_reason="suspicious_value" if is_suspicious else ("implausible_value" if not is_plausible else "")
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
# COMPARISON LOGIC
# ---------------------------------------------------------------------
def compare_and_decide(
    native_result: Optional[RevResult],
    gpt_result: RevResult,
    pdf_path: Path
) -> RevResult:
    """
    Compare PyMuPDF and GPT results, choose the best one.
    
    Priority:
    1. Both valid and agree → Use PyMuPDF (higher confidence)
    2. One valid, one invalid → Use valid one
    3. Both valid but disagree → Use higher confidence
    4. Both invalid → Flag for human review
    """
    
    # No native result → use GPT
    if not native_result:
        return gpt_result
    
    native_val = native_result.value
    gpt_val = gpt_result.value
    
    native_plausible = is_plausible_rev_value(native_val)
    gpt_plausible = is_plausible_rev_value(gpt_val)
    
    # Log special character scenarios
    if is_special_char(native_val):
        LOG.info(f"  Native returned special char '{native_val}', GPT returned '{gpt_val}'")
    
    # Both agree - ALWAYS valid, even for special characters!
    if native_val == gpt_val:
        # Special case: both agree on special char
        if is_special_char(native_val):
            return RevResult(
                file=pdf_path.name,
                value=native_val,
                engine="pymupdf+gpt_agree",
                confidence="high",
                notes=f"Both engines agree: '{native_val}' (special char confirmed)",
                human_review=False,  # ✓ No review needed when both agree!
                review_reason=""
            )
        # Normal case: both agree
        return RevResult(
            file=pdf_path.name,
            value=native_val,
            engine="pymupdf+gpt_agree",
            confidence="high",
            notes=f"Both engines agree: {native_val}",
            human_review=False,
            review_reason=""
        )
    
    # Both valid
    if native_plausible and gpt_plausible:
        # Choose higher confidence
        if native_result.confidence == "high":
            return RevResult(
                file=pdf_path.name,
                value=native_val,
                engine="pymupdf+gpt_differ",
                confidence="medium",
                notes=f"Chose PyMuPDF {native_val} over GPT {gpt_val} (higher conf)",
                human_review=True,
                review_reason="engines_disagree"
            )
        else:
            return RevResult(
                file=pdf_path.name,
                value=gpt_val,
                engine="gpt+pymupdf_differ",
                confidence="medium",
                notes=f"Chose GPT {gpt_val} over PyMuPDF {native_val} (higher conf)",
                human_review=True,
                review_reason="engines_disagree"
            )
    
    # One valid, one invalid
    if native_plausible and not gpt_plausible:
        return RevResult(
            file=pdf_path.name,
            value=native_val,
            engine="pymupdf_valid",
            confidence="medium",
            notes=f"Chose PyMuPDF {native_val} (GPT {gpt_val} invalid)",
            human_review=False,
            review_reason=""
        )
    
    if gpt_plausible and not native_plausible:
        # Special case: GPT returned special char with high confidence
        # This means GPT visually confirmed it's truly empty/no revision
        if is_special_char(gpt_val) and gpt_result.confidence == "high":
            return RevResult(
                file=pdf_path.name,
                value=gpt_val,
                engine="gpt_valid",
                confidence="high",
                notes=f"GPT confirmed special char '{gpt_val}' (no revision)",
                human_review=False,  # ✓ No review needed for high-confidence GPT
                review_reason=""
            )
        # Normal case: GPT valid, native invalid
        return RevResult(
            file=pdf_path.name,
            value=gpt_val,
            engine="gpt_valid",
            confidence="medium",
            notes=f"Chose GPT '{gpt_val}' (PyMuPDF '{native_val}' invalid/suspicious)",
            human_review=False,
            review_reason=""
        )
    
    # Both invalid → flag for review
    # UNLESS both are special characters (different forms of "empty/no revision")
    if is_special_char(native_val) and is_special_char(gpt_val):
        # Both engines agree it's empty/no revision, just different representations
        return RevResult(
            file=pdf_path.name,
            value=gpt_val,  # Prefer GPT's visual interpretation
            engine="both_special_char",
            confidence="medium",
            notes=f"Both engines indicate no revision: PyMuPDF='{native_val}', GPT='{gpt_val}'",
            human_review=False,  # ✓ No review needed
            review_reason=""
        )
    
    # Truly both invalid
    return RevResult(
        file=pdf_path.name,
        value=gpt_val,  # Default to GPT
        engine="both_invalid",
        confidence="low",
        notes=f"Both invalid: PyMuPDF={native_val}, GPT={gpt_val}",
        human_review=True,
        review_reason="both_invalid"
    )

# ---------------------------------------------------------------------
# HYBRID PIPELINE
# ---------------------------------------------------------------------
def run_hybrid_pipeline(
    input_folder: Path,
    output_csv: Path,
    azure_endpoint: str,
    azure_key: str,
    deployment_name: str = "gpt-4.1",
    disable_gpt: bool = False,
    max_workers: int = 4  # NEW: Parallel processing
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Enhanced hybrid pipeline with validation and parallel processing."""
    rows: List[Dict[str, Any]] = []
    
    if not disable_gpt:
        LOG.info("Initializing Azure GPT client...")
        gpt = AzureGPTExtractor(azure_endpoint, azure_key, deployment_name)
    else:
        gpt = None
        LOG.info("GPT disabled - using PyMuPDF only")
        max_workers = 1  # No parallelization without GPT
    
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
        "both_differ": 0
    }
    
    def process_single_pdf(pdf_path: Path) -> Tuple[Dict[str, Any], Dict[str, int]]:
        """Process a single PDF (for parallel execution)."""
        local_stats = {
            "native_only": 0,
            "native_suspicious": 0,
            "gpt_used": 0,
            "human_review": 0,
            "both_agree": 0,
            "both_differ": 0
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
                LOG.debug(f"→ Suspicious value {native_result.value} in {pdf_path.name}, retrying with GPT")
            
            # Step 3: Use GPT if needed
            if needs_gpt and not disable_gpt:
                local_stats["gpt_used"] += 1
                gpt_result = gpt.extract_rev(pdf_path)
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
    
    # Process PDFs in parallel
    LOG.info(f"Processing {len(pdfs)} PDFs with {max_workers} workers...")
    
    if max_workers > 1:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single_pdf, pdf): pdf for pdf in pdfs}
            
            for future in tqdm(as_completed(futures), total=len(pdfs), desc="Processing PDFs"):
                try:
                    row, local_stats = future.result()
                    rows.append(row)
                    # Aggregate stats
                    for key in local_stats:
                        stats[key] += local_stats[key]
                except Exception as e:
                    pdf = futures[future]
                    LOG.error(f"Future failed for {pdf.name}: {e}")
    else:
        # Sequential processing (for debugging)
        for pdf_path in tqdm(pdfs, desc="Processing PDFs"):
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
        LOG.info(f"Native suspicious (GPT retry): {stats['native_suspicious']}")
        LOG.info(f"GPT used: {stats['gpt_used']} ({stats['gpt_used']/stats['total']*100:.1f}%)")
        LOG.info(f"Both engines agree: {stats['both_agree']}")
        LOG.info(f"Engines differ: {stats['both_differ']}")
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
    a = argparse.ArgumentParser(description="Enhanced PyMuPDF + GPT with validation")
    a.add_argument("input_folder", type=Path)
    a.add_argument("-o", "--output", type=Path, default=Path("rev_results_enhanced.csv"))
    a.add_argument("--azure-endpoint", type=str, default=os.getenv("AZURE_OPENAI_ENDPOINT"))
    a.add_argument("--azure-key", type=str, default=os.getenv("AZURE_OPENAI_KEY"))
    a.add_argument("--deployment-name", type=str, default="gpt-4.1")
    a.add_argument("--disable-gpt", action="store_true", help="Use PyMuPDF only")
    a.add_argument("--gpt-only", action="store_true", help="Force GPT-only (disable native extractor)")
    a.add_argument("--max-workers", type=int, default=2, 
                   help="Number of parallel workers (default: 2, use 1 for sequential)")
    return a.parse_args(argv)

def main(argv=None):
    start_time = time.time()
    args = parse_args(argv)
    # Support a runtime "gpt-only" override: disable native extraction even if import succeeded
    if getattr(args, 'gpt_only', False):
        global NATIVE_AVAILABLE
        NATIVE_AVAILABLE = False
        # Ensure pipeline knows GPT is enabled
        args.disable_gpt = False
        LOG.info("GPT-only mode enabled: native extraction disabled")
    
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
        args.max_workers  # NEW
    )

    end_time = time.time()
    total_seconds = end_time - start_time
    mins = int(total_seconds // 60)
    secs = int(total_seconds % 60)
    LOG.info(f"Completed in {mins}m {secs}s")

    return results

if __name__ == "__main__":
    main()
