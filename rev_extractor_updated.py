#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REV Extractor — PyMuPDF + Azure GPT-4 Fallback (FIXED)
Fixed: GPT now confidently extracts numeric REVs (2-0, 3-0) instead of returning NO_REV
"""

from __future__ import annotations
import argparse, base64, csv, logging, os, re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
from tqdm import tqdm

# Azure OpenAI SDK
try:
    from openai import AzureOpenAI
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

LOG = logging.getLogger("rev_extractor_gpt_fixed")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# FIXED SYSTEM PROMPT - Be confident about numeric REVs
GPT_SYSTEM_PROMPT = """You are an expert at analyzing engineering drawings and extracting revision information.

YOUR TASK:
Extract the REV (revision) value from the title block of this engineering drawing.

TITLE BLOCK LOCATION:
- Usually in the BOTTOM-RIGHT corner
- Contains: DWG NO, SHEET, SCALE, DRAWN BY, CHECKED BY, APPROVED BY, DATE
- Often includes company name/logo (ROTORK, FAIRCHILD, etc.)

VALID REV FORMATS (ALL EQUALLY COMMON AND VALID):

NUMERIC REVISIONS - THESE ARE STANDARD AND COMMON:
- Hyphenated numbers: 1-0, 2-0, 3-0, 4-0, 5-1, 12-01, 15-02, etc.
- Format: [major]-[minor] version (e.g., 2-0 means version 2.0)
- If you see "REV: 2-0" or "REV: 3-0" → confidently return that value
- These are NOT drawing numbers (drawing numbers are longer like 055-IPI-057)
- DO NOT confuse hyphenated REVs with grid coordinates or other numbers

LETTER REVISIONS - ALSO STANDARD:
- Single letter: A, B, C, D, E, F, ... Z
- Double letters: AA, AB, AC, etc.

NO REV - ONLY USE WHEN TRULY ABSENT:
- Use ONLY if NO REV field exists in the title block
- Or if the REV field exists but is clearly empty/blank
- Don't return NO_REV just because the format looks unfamiliar to you

CRITICAL RULES:

1. NUMERIC REVs (like 2-0, 3-0, 1-0) ARE VALID AND COMMON
   - If you see a hyphenated number in or near the REV field → return it confidently
   - Don't second-guess yourself - these are standard revision formats
   - Example: "DWG NO: 21620 | REV: 2-0" → return "2-0" with high confidence

2. CHECK TITLE BLOCK ONLY
   - Ignore revision history tables (usually top-right, with dates and descriptions)
   - Ignore grid reference letters/numbers on drawing edges
   - Ignore section markers (SECTION C-C, VIEW A-A)
   - Ignore drawing numbers (they're longer: 055-IPI-057, not REVs)

3. BE CONFIDENT WITH CLEAR VALUES
   - If you clearly see "REV: 2-0" → return "2-0" (don't return NO_REV!)
   - If you clearly see "REV: A" → return "A"
   - Only return NO_REV when the field genuinely doesn't exist or is empty

RESPONSE FORMAT:
{
  "rev_value": "2-0",
  "confidence": "high",
  "location": "bottom-right title block",
  "notes": "Clear hyphenated numeric REV 2-0 in title block"
}

EXAMPLES:

Example 1 - Numeric REV (COMMON AND VALID):
Title block: "DWG NO: 21620 | REV: 2-0 | SHEET 1"
✅ Correct: {"rev_value": "2-0", "confidence": "high", "notes": "Numeric REV 2-0 in title block"}
❌ Wrong: {"rev_value": "NO_REV"} ← Don't do this! The value is clearly visible!

Example 2 - Another Numeric REV:
Title block: "DWG NO: 22416 | REV: 3-0"
✅ Correct: {"rev_value": "3-0", "confidence": "high"}
❌ Wrong: {"rev_value": "NO_REV"} ← Wrong! 3-0 is a valid REV!

Example 3 - Numeric REV 1-0:
Title block: "DWG NO: 12345 | REV: 1-0"
✅ Correct: {"rev_value": "1-0", "confidence": "high"}

Example 4 - Letter REV:
Title block: "DWG NO: 032-IPI-008 | REV: E"
✅ Correct: {"rev_value": "E", "confidence": "high"}

Example 5 - Letter REV:
Title block: "REV: A | SHEET 1"
✅ Correct: {"rev_value": "A", "confidence": "high"}

Example 6 - NO REV (truly absent):
Title block has DWG NO, SHEET, SCALE but NO REV field at all
✅ Correct: {"rev_value": "NO_REV", "confidence": "high", "notes": "No REV field present"}

Example 7 - NO REV (empty field):
Title block shows "REV: ____" (blank/empty)
✅ Correct: {"rev_value": "NO_REV", "confidence": "high", "notes": "REV field empty"}

REMEMBER:
- Hyphenated numbers like 2-0, 3-0, 1-0 are STANDARD REV formats → return them confidently
- Only use NO_REV when the field truly doesn't exist or is empty
- Trust what you see in the title block
"""

@dataclass
class RevResult:
    file: str
    value: str
    engine: str
    confidence: str = "unknown"
    notes: str = ""

# Native extraction function
def extract_native_pymupdf(pdf_path: Path) -> Optional[RevResult]:
    """Try native PyMuPDF extraction using the fixed logic."""
    try:
        from rev_extractor_fixed import (
            process_pdf_native, _normalize_output_value,
            DEFAULT_BR_X, DEFAULT_BR_Y, DEFAULT_EDGE_MARGIN, DEFAULT_REV_2L_BLOCKLIST
        )
        
        best = process_pdf_native(
            pdf_path,
            brx=DEFAULT_BR_X,
            bry=DEFAULT_BR_Y,
            blocklist=DEFAULT_REV_2L_BLOCKLIST,
            edge_margin=DEFAULT_EDGE_MARGIN
        )
        
        if best and best.value:
            value = _normalize_output_value(best.value)
            return RevResult(
                file=pdf_path.name,
                value=value,
                engine=f"pymupdf_{best.engine}",
                confidence="high" if best.score > 100 else "medium",
                notes=best.context_snippet
            )
        return None
    except Exception as e:
        LOG.debug(f"Native extraction failed for {pdf_path.name}: {e}")
        return None

REV_NUMERIC_PATTERN = re.compile(r"\b\d{1,3}-\d{1,2}\b")

def _normalize_rev_value(raw: Any) -> str:
    """Normalise GPT rev_value into the formats we expect."""
    if raw is None:
        return ""
    v = str(raw).strip().upper()

    if v in {"", "NONE", "N/A", "NA"}:
        return "NO_REV"

    if v in {"NO_REV", "EMPTY"}:
        return v

    # Convert 2.0 → 2-0, 3.00 → 3-0, etc.
    m = re.fullmatch(r"(\d+)\.(\d+)", v)
    if m:
        major, minor = m.groups()
        return f"{int(major)}-{int(minor)}"

    # Remove spaces, e.g. "2 - 0" → "2-0"
    v = v.replace(" ", "")
    return v

def _validate_gpt_result(pdf_path: Path, result_data: Dict[str, Any]) -> RevResult:
    """
    Validate and correct GPT output using the real PDF text.
    
    This validation is LIGHT - we now trust GPT more since the prompt is clearer.
    We only correct obvious hallucinations.
    """
    raw_value = result_data.get("rev_value", "")
    value = _normalize_rev_value(raw_value)
    confidence = (result_data.get("confidence") or "unknown").lower()
    notes = (result_data.get("notes") or "").strip()
    engine = "gpt_vision"

    # Read page text once
    try:
        with fitz.open(pdf_path) as doc:
            page_text = (doc[0].get_text("text") or "")
    except Exception:
        page_text = ""
    text_upper = page_text.upper()

    # Collect numeric candidates up-front
    numeric_candidates = set(REV_NUMERIC_PATTERN.findall(text_upper))

    # ------------------------------
    # If GPT returned a numeric REV, validate it exists on page
    # ------------------------------
    if REV_NUMERIC_PATTERN.fullmatch(value or ""):
        if value in numeric_candidates:
            # Good: supported by page text
            if confidence in {"unknown", ""}:
                confidence = "high"
        elif len(numeric_candidates) == 1:
            # GPT got wrong number but there's only one candidate - snap to it
            real_value = next(iter(numeric_candidates))
            if notes:
                notes += " | "
            notes += f"GPT suggested {value}, corrected to {real_value} based on page text"
            value = real_value
            confidence = "high"
        else:
            # Multiple or no candidates - can't validate, but TRUST GPT since prompt is better now
            # Only downgrade confidence, don't force NO_REV
            if confidence == "high":
                confidence = "medium"
            if notes:
                notes += " | "
            notes += "Could not validate numeric value in page text"

        return RevResult(
            file=pdf_path.name,
            value=value,
            engine=engine,
            confidence=confidence or "unknown",
            notes=notes,
        )

    # ------------------------------
    # If GPT returned NO_REV/EMPTY
    # ------------------------------
    if value in {"NO_REV", "EMPTY"}:
        # Light salvage: if there's exactly ONE numeric candidate near REV label, override
        if page_text and len(numeric_candidates) == 1:
            candidate = next(iter(numeric_candidates))
            # More relaxed pattern - within 20 chars of REV
            near_pattern = re.compile(
                rf"REV[^A-Z0-9]{{0,20}}{re.escape(candidate)}\b",
                re.IGNORECASE,
            )
            if near_pattern.search(page_text):
                if notes:
                    notes += " | "
                notes += f"GPT returned {value} but found {candidate} near REV label; overriding"
                return RevResult(
                    file=pdf_path.name,
                    value=candidate,
                    engine=engine,
                    confidence="medium",
                    notes=notes,
                )

        # Keep GPT's NO_REV
        if not notes:
            notes = "No REV field found"
        return RevResult(
            file=pdf_path.name,
            value=value,
            engine=engine,
            confidence=confidence,
            notes=notes,
        )

    # ------------------------------
    # Letter or other value - basic validation
    # ------------------------------
    if value and page_text:
        near_rev_pattern = re.compile(
            rf"REV[^A-Z0-9]{{0,8}}{re.escape(value)}\b",
            re.IGNORECASE,
        )
        if not near_rev_pattern.search(page_text):
            if confidence == "high":
                confidence = "medium"
            if notes:
                notes += " | "
            notes += "Value not found near REV label in text"

    return RevResult(
        file=pdf_path.name,
        value=value,
        engine=engine,
        confidence=confidence or "unknown",
        notes=notes,
    )


class AzureGPTExtractor:
    def __init__(self, endpoint: str, api_key: str, deployment_name: str = "gpt-4.1"):
        if not AZURE_AVAILABLE:
            raise ImportError("openai not installed. Run: pip install openai")
        
        endpoint = endpoint.rstrip("/")
        if "/openai/deployments" in endpoint:
            endpoint = endpoint.split("/openai/deployments")[0]
        
        LOG.info(f"Initializing GPT client with endpoint: {endpoint}")
        LOG.info(f"Using deployment: {deployment_name}")
        
        try:
            self.client = AzureOpenAI(
                api_key=api_key,
                api_version="2024-02-01",
                azure_endpoint=endpoint,
            )
            self.deployment_name = deployment_name
            LOG.info("✓ GPT client initialized")
        except Exception as e:
            LOG.error(f"Failed to initialize GPT client: {e}")
            raise

    def pdf_to_base64_image(self, pdf_path: Path, page_idx: int = 0, dpi: int = 150) -> str:
        """Convert a PDF page to base64-encoded PNG for vision."""
        with fitz.open(pdf_path) as doc:
            page = doc[page_idx]
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            png_bytes = pix.tobytes("png")
        return base64.b64encode(png_bytes).decode("utf-8")

    def extract_rev(self, pdf_path: Path) -> RevResult:
        """Extract REV using Azure GPT-4 vision with updated confident prompt."""
        try:
            LOG.debug(f"Converting {pdf_path.name} to image...")
            img_base64 = self.pdf_to_base64_image(pdf_path, page_idx=0, dpi=150)

            LOG.debug(f"Sending {pdf_path.name} to GPT API...")
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {
                        "role": "system",
                        "content": GPT_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Extract the REV value from this engineering drawing. "
                                    "Find the title block (bottom-right, with DWG NO, SHEET, SCALE). "
                                    "Numeric formats like 2-0, 3-0, 1-0 are VALID and COMMON - return them confidently. "
                                    "Only return NO_REV if there's truly no REV field or it's empty."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                },
                            },
                        ],
                    },
                ],
                max_tokens=400,
                temperature=0.0,
            )

            result_text = response.choices[0].message.content or ""
            LOG.debug(f"Raw GPT response: {result_text[:200]}")

            # Strip code fences if present
            import json
            json_match = re.search(r"```json\s*(\{.*?\})\s*```", result_text, re.DOTALL | re.IGNORECASE)
            if json_match:
                result_text = json_match.group(1)
            elif "```" in result_text:
                result_text = re.sub(r"```.*?```", "", result_text, flags=re.DOTALL)

            result_data = json.loads(result_text.strip())

            # Light validation
            return _validate_gpt_result(pdf_path, result_data)

        except Exception as e:
            LOG.error(f"GPT extraction failed for {pdf_path.name}: {e}")
            return RevResult(
                file=pdf_path.name,
                value="NO_REV",
                engine="gpt_failed",
                confidence="low",
                notes=str(e)[:100],
            )

def run_hybrid_pipeline(
    input_folder: Path,
    output_csv: Path,
    azure_endpoint: str,
    azure_key: str,
    deployment_name: str = "gpt-4.1"
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    
    if not azure_endpoint or not azure_key:
        LOG.error("Azure endpoint or key not provided")
        return rows
    
    LOG.info("Initializing Azure GPT client...")
    gpt = AzureGPTExtractor(azure_endpoint, azure_key, deployment_name)
    
    pdfs = list(input_folder.glob("*.pdf"))
    if not pdfs:
        LOG.warning(f"No PDFs found in {input_folder}")
        return rows
    
    native_success = 0
    gpt_used = 0
    gpt_failed = 0
    
    for pdf_path in tqdm(pdfs, desc="Processing PDFs"):
        try:
            # Try PyMuPDF native first
            result = extract_native_pymupdf(pdf_path)
            
            if result and result.value and result.value not in ["", "NO_REV"]:
                native_success += 1
            else:
                # Fallback to GPT
                gpt_used += 1
                result = gpt.extract_rev(pdf_path)
                if result.engine == "gpt_failed":
                    gpt_failed += 1
            
            rows.append({
                "file": result.file,
                "value": result.value,
                "engine": result.engine,
                "confidence": result.confidence,
                "notes": result.notes[:100]
            })
            
        except Exception as e:
            LOG.error(f"Failed {pdf_path.name}: {e}")
            rows.append({
                "file": pdf_path.name,
                "value": "",
                "engine": "error",
                "confidence": "none",
                "notes": str(e)[:100]
            })
    
    # Write CSV
    try:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, 'w', newline='', encoding='utf-8-sig') as outf:
            writer = csv.DictWriter(outf, fieldnames=['file', 'value', 'engine', 'confidence', 'notes'])
            writer.writeheader()
            writer.writerows(rows)
        
        LOG.info(f"Wrote {output_csv.resolve()} with {len(rows)} rows")
        LOG.info(f"Stats: Native={native_success}, GPT={gpt_used}, Failed={gpt_failed}")
        if gpt_used > 0:
            LOG.info(f"Cost≈${(gpt_used - gpt_failed) * 0.010:.2f}")
    except Exception as e:
        LOG.error(f"Failed to write CSV: {e}")
    
    return rows

def parse_args(argv=None):
    a = argparse.ArgumentParser(description="PyMuPDF + GPT hybrid (FIXED)")
    a.add_argument("input_folder", type=Path)
    a.add_argument("-o", "--output", type=Path, default=Path("rev_results.csv"))
    a.add_argument("--azure-endpoint", type=str,
                   default=os.getenv("AZURE_OPENAI_ENDPOINT"))
    a.add_argument("--azure-key", type=str,
                   default=os.getenv("AZURE_OPENAI_KEY"))
    a.add_argument("--deployment-name", type=str, default="gpt-4.1")
    return a.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    
    if not args.azure_endpoint or not args.azure_key:
        LOG.error("Azure credentials required")
        return []
    
    return run_hybrid_pipeline(
        args.input_folder,
        args.output,
        args.azure_endpoint,
        args.azure_key,
        args.deployment_name
    )

if __name__ == "__main__":
    main()
