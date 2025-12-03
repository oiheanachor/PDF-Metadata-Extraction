#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REV Extractor — SIMPLE Production Version
No rotation logic, clean output, just works
"""

from __future__ import annotations
import argparse, base64, csv, logging, os, re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
from tqdm import tqdm

# OpenAI SDK for Azure
try:
    from openai import AzureOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

LOG = logging.getLogger("rev_extractor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ----------------------------- SYSTEM PROMPT (SIMPLE & BALANCED) --------------------

GPT_SYSTEM_PROMPT = """You are an expert at analyzing engineering drawings and extracting revision information.

**YOUR TASK:**
Extract the REV (revision) value from the title block of this engineering drawing.

⚠️ READ THE ACTUAL VALUE IN THIS DRAWING - Do not memorize or default to example values!

**TITLE BLOCK LOCATION:**
The title block is typically in the bottom-right corner and contains:
- Company logo/name (ROTORK, FAIRCHILD, etc.)
- DWG NO / DRAWING NO
- SHEET X OF Y
- SCALE
- DRAWN BY / CHECKED BY / APPROVED BY
- The REV field is WITHIN or ADJACENT to this cluster

**CRITICAL - AVOID THESE:**
❌ Revision history tables (top-right, columns: REV | DATE | DESCRIPTION)
❌ Grid reference letters (A, B, C along edges)
❌ Section markers ("SECTION C-C", "SECTION B-B")
❌ View indicators or part numbers

**REV VALUE FORMATS (ALL EQUALLY VALID):**

**NUMERIC REVISIONS:**
- Pattern: [number]-[number]
- Examples: 1-0, 2-0, 3-0, 4-0, 5-0, 5-1, 12-01
- ⚠️ These are JUST AS VALID as letter revisions!

**LETTER REVISIONS:**
- Single: A, B, C, D, E, F, ... Z
- Double: AA, AB, AC, etc.
- ⚠️ These are JUST AS VALID as numeric revisions!

**SPECIAL CASE:**
- No REV field exists OR field is empty → return "NO_REV"

**EXTRACTION STRATEGY:**
1. Find title block (look for DWG NO, SHEET, SCALE, company name)
2. Locate REV field within or adjacent to title block
3. Read the ACTUAL value you see
4. Verify it's not from revision history table
5. Verify it's not a grid letter or section marker

**VALIDATION CHECKLIST:**
✓ In title block area? (near DWG NO, SHEET, SCALE)
✓ Has "REV:" or "REV" label nearby?
✓ Standalone value (not part of another number)?
✓ Valid format (letter OR hyphenated number)?
✓ NOT from revision history table?
✓ NOT a grid letter or section marker?
✓ Reading THIS drawing's actual value (not copying examples)?

**RESPONSE FORMAT:**
{
  "rev_value": "2-0",
  "confidence": "high",
  "notes": "Clear REV 2-0 in title block next to DWG NO"
}

**EXAMPLES (ALL FORMATS EQUALLY VALID):**

Example 1: Numeric "1-0"
Title block: "DWG NO: 12345 | REV: 1-0"
✅ Output: "1-0"

Example 2: Numeric "2-0"
Title block: "DWG NO: 21620 | REV: 2-0 | SHEET 1"
✅ Output: "2-0"

Example 3: Numeric "3-0"
Title block: "DWG NO: 22416 | REV: 3-0"
✅ Output: "3-0"

Example 4: Numeric "4-0"
Title block: "DRAWING NO: 54321 | REV: 4-0"
✅ Output: "4-0"

Example 5: Numeric "5-1"
Title block: "DWG NO: 67890 | REV: 5-1"
✅ Output: "5-1"

Example 6: Letter "A"
Title block: "DWG NO: 21837 | REV: A"
✅ Output: "A"

Example 7: Letter "C"
Title block: "DWG NO: 18301 | REV: C | SHEET 1"
✅ Output: "C"

Example 8: Letter "E"
Title block: "DWG NO: 032-IPI-008 | REV: E"
✅ Output: "E"

Example 9: Letter "F"
Title block: "DWG NO: EB-00131 | REV: F"
✅ Output: "F"

Example 10: Double Letter "AB"
Title block: "DRAWING NO: 14579 | REV: AB"
✅ Output: "AB"

Example 11: NO_REV (no field or empty)
Title block has DWG NO, SHEET, SCALE but NO REV field OR REV field is empty
✅ Output: "NO_REV"

Example 12: False Positive (section marker)
Drawing has "SECTION C-C" and letters on edges, but no REV in title block
✅ Output: "NO_REV"
❌ Don't output: "C" (that's a section marker)

**CRITICAL REMINDERS:**
1. Numeric formats (1-0, 2-0, 3-0, etc.) are EQUALLY VALID as letters (A, B, C)
2. When you see "REV: 2-0" → output "2-0"
3. When you see "REV: 3-0" → output "3-0"
4. When you see "REV: A" → output "A"
5. Read THIS drawing's ACTUAL value - every drawing is different
6. If no REV field or empty → output "NO_REV"

**CONFIDENCE LEVELS:**
- "high": Clear REV label with unambiguous value
- "medium": Value found near title block but no explicit REV label
- "low": Uncertain

⚠️ Before responding: Am I reading THIS drawing's actual value, or copying from examples?"""

# ----------------------------- Data Structures ---------------------------------

@dataclass
class RevResult:
    file: str
    value: str
    engine: str
    confidence: str = "unknown"
    notes: str = ""

# ----------------------------- Native Extraction -------------------------------

def extract_native_pymupdf(pdf_path: Path) -> Optional[RevResult]:
    """Try native PyMuPDF extraction."""
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
            # Standardize: use "NO_REV" not "EMPTY"
            if value == "EMPTY":
                value = "NO_REV"
            return RevResult(
                file=pdf_path.name,
                value=value,
                engine=f"pymupdf_{best.engine}",
                confidence="high" if best.score > 100 else "medium",
                notes=best.context_snippet[:100]
            )
        return None
    except Exception as e:
        LOG.debug(f"Native extraction failed for {pdf_path.name}: {e}")
        return None

# ----------------------------- GPT-4.1 Vision Extractor ------------------------

class AzureGPTExtractor:
    def __init__(self, endpoint: str, api_key: str, deployment_name: str = "gpt-4.1"):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai not installed. Run: pip install openai")
        
        # Clean endpoint
        endpoint = endpoint.rstrip('/')
        if '/openai/deployments' in endpoint:
            endpoint = endpoint.split('/openai/deployments')[0]
        
        LOG.info(f"Initializing GPT-4.1")
        LOG.info(f"Endpoint: {endpoint}")
        LOG.info(f"Deployment: {deployment_name}")
        
        try:
            self.client = AzureOpenAI(
                api_key=api_key,
                api_version="2024-02-15-preview",
                azure_endpoint=endpoint
            )
            self.deployment_name = deployment_name
            LOG.info("✓ GPT-4.1 initialized")
        except Exception as e:
            LOG.error(f"Failed to initialize GPT: {e}")
            raise
    
    def pdf_to_base64_image(self, pdf_path: Path, dpi: int = 150) -> str:
        """Convert PDF to base64 image - NO ROTATION."""
        with fitz.open(pdf_path) as doc:
            page = doc[0]
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            png_bytes = pix.tobytes("png")
            return base64.b64encode(png_bytes).decode('utf-8')
    
    def extract_rev(self, pdf_path: Path) -> RevResult:
        """Extract REV using GPT-4.1 Vision - SIMPLE, NO ROTATION."""
        try:
            # Convert to image (no rotation)
            img_base64 = self.pdf_to_base64_image(pdf_path, dpi=150)
            
            # Simple user message
            user_text = """Extract the REV value from this engineering drawing.

CRITICAL:
1. Find the title block (DWG NO, SHEET, SCALE)
2. Locate REV field in or near title block
3. Read the ACTUAL value you see
4. Numeric (1-0, 2-0, 3-0) and letter (A, B, C) are equally valid
5. If no REV field or empty → return "NO_REV"

Read THIS drawing's actual value!"""
            
            # Call GPT
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": GPT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_text},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0
            )
            
            result_text = response.choices[0].message.content
            
            # Parse JSON
            import json
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', result_text, re.DOTALL)
            if json_match:
                result_text = json_match.group(1)
            elif '```' in result_text:
                result_text = re.sub(r'```.*?```', '', result_text, flags=re.DOTALL)
            
            result_data = json.loads(result_text.strip())
            
            value = result_data.get("rev_value", "")
            
            # Standardize: use "NO_REV" not "EMPTY"
            if value == "EMPTY":
                value = "NO_REV"
            
            return RevResult(
                file=pdf_path.name,
                value=value,
                engine="gpt_vision",
                confidence=result_data.get("confidence", "unknown"),
                notes=result_data.get("notes", "")[:200]
            )
            
        except Exception as e:
            LOG.error(f"GPT failed for {pdf_path.name}: {e}")
            return RevResult(
                file=pdf_path.name,
                value="",
                engine="gpt_failed",
                confidence="none",
                notes=str(e)[:100]
            )

# ----------------------------- Pipeline ----------------------------------------

def run_hybrid_pipeline(
    input_folder: Path,
    output_csv: Path,
    azure_endpoint: str,
    azure_key: str,
    deployment_name: str = "gpt-4.1"
) -> List[Dict[str, Any]]:
    """Simple hybrid pipeline."""
    rows: List[Dict[str, Any]] = []
    
    # Initialize GPT
    gpt = AzureGPTExtractor(azure_endpoint, azure_key, deployment_name)
    
    # Get PDFs
    pdfs = sorted(input_folder.glob("*.pdf"))
    if not pdfs:
        LOG.warning(f"No PDFs found in {input_folder}")
        return rows
    
    LOG.info(f"Found {len(pdfs)} PDFs")
    
    native_success = 0
    gpt_used = 0
    gpt_failed = 0
    
    for pdf_path in tqdm(pdfs, desc="Processing"):
        try:
            # Try native first
            result = extract_native_pymupdf(pdf_path)
            
            if result and result.value and result.value != "NO_REV":
                native_success += 1
                rows.append({
                    "file": result.file,
                    "value": result.value,
                    "engine": result.engine,
                    "confidence": result.confidence,
                    "notes": result.notes
                })
                LOG.debug(f"✓ Native: {pdf_path.name} → {result.value}")
                continue
            
            # Fall back to GPT
            LOG.info(f"→ GPT: {pdf_path.name}")
            gpt_used += 1
            result = gpt.extract_rev(pdf_path)
            
            if result.engine == "gpt_failed":
                gpt_failed += 1
            
            rows.append({
                "file": result.file,
                "value": result.value,
                "engine": result.engine,
                "confidence": result.confidence,
                "notes": result.notes
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
    
    # Write results - SIMPLE FORMAT (no "actual" column)
    try:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, 'w', newline='', encoding='utf-8-sig') as outf:
            writer = csv.DictWriter(outf, fieldnames=['file', 'value', 'engine', 'confidence', 'notes'])
            writer.writeheader()
            writer.writerows(rows)
        
        LOG.info(f"\n{'='*60}")
        LOG.info(f"Results: {output_csv.resolve()}")
        LOG.info(f"{'='*60}")
        LOG.info(f"Total: {len(rows)} files")
        LOG.info(f"Native (free): {native_success}")
        LOG.info(f"GPT (paid): {gpt_used}")
        LOG.info(f"Failed: {gpt_failed}")
        if gpt_used > 0:
            cost = (gpt_used - gpt_failed) * 0.010
            LOG.info(f"Cost: ${cost:.2f}")
        LOG.info(f"{'='*60}\n")
    except Exception as e:
        LOG.error(f"Failed to write CSV: {e}")
    
    return rows

def parse_args(argv=None):
    a = argparse.ArgumentParser(description="REV Extractor - Simple Production")
    a.add_argument("input_folder", type=Path)
    a.add_argument("-o", "--output", type=Path, default=Path("rev_results.csv"))
    a.add_argument("--azure-endpoint", type=str, default=os.getenv("AZURE_OPENAI_ENDPOINT"))
    a.add_argument("--azure-key", type=str, default=os.getenv("AZURE_OPENAI_KEY"))
    a.add_argument("--deployment-name", type=str, default="gpt-4.1")
    return a.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    
    if not args.azure_endpoint or not args.azure_key:
        LOG.error("Azure credentials required")
        LOG.error("Set: $env:AZURE_OPENAI_ENDPOINT and $env:AZURE_OPENAI_KEY")
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
