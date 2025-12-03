#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REV Extractor — PyMuPDF + Azure GPT-4 Fallback (OPTIMIZED)
Fixed to handle: numeric REVs (2-0, 3-0), NO_REV cases, and false positives
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

LOG = logging.getLogger("rev_extractor_gpt_optimized")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# IMPROVED SYSTEM PROMPT - Better handling of numeric REVs
GPT_SYSTEM_PROMPT = """You are an expert at analyzing engineering drawings and extracting revision information.

**YOUR TASK:**
Extract the REV (revision) value from the title block of this engineering drawing.

**CRITICAL RULES:**

1. **Title Block Location**: 
   - The REV value is in the TITLE BLOCK, typically in the BOTTOM-RIGHT corner
   - Title blocks contain: DWG NO, SHEET, SCALE, DRAWN BY, CHECKED BY, APPROVED BY
   - Usually has company logo/name (ROTORK, FAIRCHILD, etc.)

2. **Avoid These Common Mistakes**:
   ❌ DO NOT extract from REVISION TABLES (top-right, shows history: REV A | DATE | DESCRIPTION)
   ❌ DO NOT extract grid reference letters (A, B, C along edges)
   ❌ DO NOT extract part numbers or item callouts
   ❌ DO NOT extract section markers (e.g., "SECTION C-C")
   ❌ DO NOT extract view indicators

3. **REV Value Formats** (in order of priority):
   
   **NUMERIC REVISIONS (Common in some companies):**
   - Hyphenated: 1-0, 2-0, 3-0, 12-01, 15-02
   - Format: [number]-[number]
   - Often used for major-minor version (2-0 = version 2.0)
   
   **LETTER REVISIONS:**
   - Single letter: A, B, C, D, ... Z
   - Double letters: AA, AB, AC, etc.
   - Letters progress alphabetically (A → B → C)
   
   **Special Cases:**
   - "OF" in REV field → return "EMPTY" (means not applicable)
   - No REV field or marking → return "NO_REV"

4. **Validation Checklist**:
   ✓ Is it in the title block (bottom corner)?
   ✓ Is there a "REV:" or "REV" label nearby?
   ✓ Is it near DWG NO, SHEET, or SCALE fields?
   ✓ Is it isolated (not part of another number)?
   ✓ Does it follow a valid format (letter or hyphenated number)?

**RESPONSE FORMAT:**
Return ONLY a JSON object:
{
  "rev_value": "2-0",
  "confidence": "high",
  "location": "bottom-right title block, next to DWG NO field",
  "notes": "Clear hyphenated numeric REV 2-0, distinct from drawing number"
}

**EXAMPLES:**

Example 1 - Numeric REV (PRIORITY):
Drawing shows:
- Top-right: Revision table with "A | ECO 123 | 1/15/20" and "B | ECO 456 | 3/20/21"
- Bottom-right title block: "REV: 2-0" next to "DWG NO: 21620"

✅ CORRECT Response:
{
  "rev_value": "2-0",
  "confidence": "high",
  "location": "bottom-right title block",
  "notes": "Hyphenated numeric REV 2-0 in title block, ignoring revision history table"
}

❌ WRONG Response:
{
  "rev_value": "B",  ← Wrong! From revision table, not title block
  "confidence": "high",
  "location": "revision table"
}

Example 2 - Letter REV:
Title block shows: "REV: F" near "DWG NO: EB-00131" and "SHEET 1 OF 1"

✅ CORRECT Response:
{
  "rev_value": "F",
  "confidence": "high",
  "location": "bottom-right title block",
  "notes": "Clear REV F in title block near DWG NO"
}

Example 3 - Grid Letter (FALSE POSITIVE):
Drawing has:
- Letter "A" along top edge (grid reference)
- Letter "C" in "SECTION C-C" label
- No actual REV field in title block

✅ CORRECT Response:
{
  "rev_value": "NO_REV",
  "confidence": "high",
  "location": "title block checked",
  "notes": "No REV field present; letters A and C are grid references and section markers"
}

❌ WRONG Response:
{
  "rev_value": "A",  ← Wrong! This is a grid letter
  "confidence": "medium",
  "location": "top of drawing"
}

Example 4 - Empty REV:
Title block shows "REV: OF" or just "OF" in the REV field

✅ CORRECT Response:
{
  "rev_value": "EMPTY",
  "confidence": "high",
  "location": "bottom-right title block",
  "notes": "REV field shows 'OF' indicating not applicable"
}

**IMPORTANT REMINDERS:**
- Hyphenated numbers (2-0, 3-0) are VALID and COMMON REV formats
- Check the TITLE BLOCK first, not revision history tables
- If you see both a letter and a number, report the one in the title block
- Grid letters and section markers are NOT revision values
- When in doubt between a letter and a hyphenated number, choose the one in the title block"""

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

class AzureGPTExtractor:
    def __init__(self, endpoint: str, api_key: str, deployment_name: str = "gpt-4o"):
        if not AZURE_AVAILABLE:
            raise ImportError("openai not installed. Run: pip install openai")
        
        # Remove trailing slash and /openai/deployments if present
        endpoint = endpoint.rstrip('/')
        if '/openai/deployments' in endpoint:
            endpoint = endpoint.split('/openai/deployments')[0]
        
        LOG.info(f"Initializing GPT client with endpoint: {endpoint}")
        LOG.info(f"Using deployment: {deployment_name}")
        
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
    
    def pdf_to_base64_image(self, pdf_path: Path, page_idx: int = 0, dpi: int = 150) -> str:
        """Convert PDF page to base64-encoded PNG."""
        with fitz.open(pdf_path) as doc:
            page = doc[page_idx]
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            png_bytes = pix.tobytes("png")
            return base64.b64encode(png_bytes).decode('utf-8')
    
    def extract_rev(self, pdf_path: Path) -> RevResult:
        """Extract REV using GPT-4 Vision."""
        try:
            LOG.debug(f"Converting {pdf_path.name} to image...")
            img_base64 = self.pdf_to_base64_image(pdf_path, page_idx=0, dpi=150)
            
            LOG.debug(f"Sending to GPT API...")
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {
                        "role": "system",
                        "content": GPT_SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extract the REV value from this engineering drawing. Remember to prioritize hyphenated numeric REVs (like 2-0) and avoid grid letters."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0
            )
            
            result_text = response.choices[0].message.content
            LOG.debug(f"GPT response received: {result_text[:100]}...")
            
            # Parse JSON
            import json
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', result_text, re.DOTALL)
            if json_match:
                result_text = json_match.group(1)
            elif '```' in result_text:
                result_text = re.sub(r'```.*?```', '', result_text, flags=re.DOTALL)
            
            result_data = json.loads(result_text.strip())
            
            return RevResult(
                file=pdf_path.name,
                value=result_data.get("rev_value", ""),
                engine="gpt_vision",
                confidence=result_data.get("confidence", "unknown"),
                notes=result_data.get("notes", "")
            )
            
        except Exception as e:
            LOG.error(f"GPT extraction failed for {pdf_path.name}: {e}")
            return RevResult(
                file=pdf_path.name,
                value="",
                engine="gpt_failed",
                confidence="none",
                notes=str(e)[:100]
            )

def run_hybrid_pipeline(
    input_folder: Path,
    output_csv: Path,
    azure_endpoint: str,
    azure_key: str,
    deployment_name: str = "gpt-4o"
) -> List[Dict[str, Any]]:
    """Hybrid pipeline."""
    rows: List[Dict[str, Any]] = []
    
    # Initialize GPT client
    LOG.info("Initializing Azure GPT client...")
    gpt = AzureGPTExtractor(azure_endpoint, azure_key, deployment_name)
    
    # Get PDFs
    pdfs = list(input_folder.glob("*.pdf"))
    if not pdfs:
        LOG.warning(f"No PDFs found in {input_folder}")
        return rows
    
    native_success = 0
    gpt_used = 0
    gpt_failed = 0
    
    for pdf_path in tqdm(pdfs, desc="Processing PDFs"):
        try:
            # Step 1: Try PyMuPDF native
            result = extract_native_pymupdf(pdf_path)
            
            if result and result.value and result.value not in ["", "NO_REV"]:
                native_success += 1
                rows.append({
                    "file": result.file,
                    "value": result.value,
                    "engine": result.engine,
                    "confidence": result.confidence,
                    "notes": result.notes[:100]
                })
                LOG.debug(f"✓ Native: {pdf_path.name} → {result.value}")
                continue
            
            # Step 2: Fall back to GPT
            LOG.info(f"→ Using GPT for {pdf_path.name}")
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
    a = argparse.ArgumentParser(description="PyMuPDF + GPT hybrid (optimized)")
    a.add_argument("input_folder", type=Path)
    a.add_argument("-o", "--output", type=Path, default=Path("rev_results.csv"))
    a.add_argument("--azure-endpoint", type=str,
                   default=os.getenv("AZURE_OPENAI_ENDPOINT"),
                   help="Azure OpenAI endpoint URL")
    a.add_argument("--azure-key", type=str,
                   default=os.getenv("AZURE_OPENAI_KEY"),
                   help="Azure OpenAI API key")
    a.add_argument("--deployment-name", type=str,
                   default="gpt-4o",
                   help="Azure OpenAI deployment name")
    return a.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    
    if not args.azure_endpoint or not args.azure_key:
        LOG.error("❌ Azure credentials required!")
        LOG.error("Set environment variables or use flags:")
        LOG.error('  $env:AZURE_OPENAI_ENDPOINT = "https://..."')
        LOG.error('  $env:AZURE_OPENAI_KEY = "your_key"')
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
