#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REV Extractor — Only Azure OpenAI GPT-4o Vision
All PDFs processed with GPT-4o Vision API
Simple, reliable, cost-effective
"""

from __future__ import annotations
import argparse, base64, csv, json, logging, os, re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import fitz  # PyMuPDF for PDF→Image conversion
from tqdm import tqdm

# OpenAI SDK for Azure
try:
    from openai import AzureOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

LOG = logging.getLogger("rev_extractor_gpt_only")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ----------------------------- Configuration -----------------------------------

GPT_SYSTEM_PROMPT = """You are an expert at analyzing engineering drawings and extracting revision information.

**YOUR TASK:**
Extract the REV (revision) value from the title block of this engineering drawing.

**CRITICAL RULES:**
1. **Title Block Location**: The REV value is in the TITLE BLOCK, typically in the BOTTOM-RIGHT corner of the drawing
2. **Avoid Revision Tables**: Do NOT extract from REVISION TABLES (usually top-right with columns like DESCRIPTION, DATE, EC)
   - Revision tables show the HISTORY of changes (multiple rows: REV A, REV B, REV C...)
   - We want the CURRENT REV from the title block (single value)
3. **Format**: REV values are typically:
   - Single uppercase letter: A, B, C, ... Z
   - Double letters: AA, AB, etc.
   - Hyphenated numbers: 1-0, 2-1, 12-01
4. **Empty REV**: If you see "OF" in the REV field → return "EMPTY"
5. **No REV Found**: If no REV marking exists → return "NO_REV"

**TITLE BLOCK IDENTIFIERS:**
Look for these anchors near the REV field:
- DWG, DWG. NO, DRAWING NO
- SHEET, SCALE, SIZE
- DRAWN, CHECKED, APPROVED
- Company logo/name (FAIRCHILD, ROTORK, etc.)

**RESPONSE FORMAT:**
Return ONLY a JSON object:
{
  "rev_value": "F",
  "confidence": "high",
  "location": "bottom-right title block",
  "notes": "Found near DWG NO and SHEET fields"
}

**CONFIDENCE LEVELS:**
- "high": Clear REV label with nearby value in title block
- "medium": Value found in title block area without clear REV label  
- "low": Ambiguous or multiple possible values
- "none": No REV found

**Example 1 - Correct:**
Drawing shows:
- Top-right: Revision table with columns "REV | DESCRIPTION | DATE" showing "E | CHANGE XYZ | 3/27/15" and "F | UPDATE ABC | 8/24/17"
- Bottom-right title block: "REV: F" near "DWG NO: EB-00131" and "SHEET 1 OF 1"

Correct response:
{
  "rev_value": "F",
  "confidence": "high",
  "location": "bottom-right title block",
  "notes": "Clear REV F in title block, ignoring revision history table"
}

**Example 2 - Empty:**
Title block shows "REV: OF" or just "OF" in REV field
Response:
{
  "rev_value": "EMPTY",
  "confidence": "high",
  "location": "bottom-right title block",
  "notes": "REV field contains 'OF' indicating empty/not applicable"
}

**Example 3 - Scanned Drawing:**
Low-quality scanned drawing with faint text in bottom-right corner showing "REV A"
Response:
{
  "rev_value": "A",
  "confidence": "medium",
  "location": "bottom-right title block",
  "notes": "Scanned drawing, slightly unclear but appears to be REV A"
}

**Example 4 - A1 Large Format:**
Very large drawing with small title block in bottom-right corner
Response:
{
  "rev_value": "C",
  "confidence": "high",
  "location": "bottom-right title block",
  "notes": "A1 format drawing, clear REV C in title block"
}

IMPORTANT: Focus on the TITLE BLOCK (bottom-right), NOT the revision history table (top-right)."""

# ----------------------------- Data Structures ---------------------------------

@dataclass
class RevResult:
    file: str
    value: str
    engine: str
    confidence: str = "unknown"
    notes: str = ""

# ----------------------------- GPT-4o Vision -----------------------------------

class AzureGPTExtractor:
    def __init__(self, endpoint: str, api_key: str, deployment_name: str = "gpt-4o"):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai not installed. Run: pip install openai")
        
        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version="2024-02-15-preview"
        )
        self.deployment = deployment_name
    
    def pdf_to_base64_image(self, pdf_path: Path, page_idx: int = 0, dpi: int = 150) -> str:
        """Convert PDF page to base64-encoded PNG."""
        with fitz.open(pdf_path) as doc:
            page = doc[page_idx]
            
            # Adaptive DPI for large pages (A1 format handling)
            rect = page.rect
            w_mm = rect.width * 0.3528
            h_mm = rect.height * 0.3528
            max_dim = max(w_mm, h_mm)
            
            if max_dim > 750:  # A0/A1 - use lower DPI
                dpi = 100
            elif max_dim > 500:  # Large format
                dpi = 120
            
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            png_bytes = pix.tobytes("png")
            return base64.b64encode(png_bytes).decode('utf-8')
    
    def extract_rev(self, pdf_path: Path) -> RevResult:
        """Extract REV using GPT-4o Vision."""
        try:
            # Convert PDF to image
            img_base64 = self.pdf_to_base64_image(pdf_path, page_idx=0)
            
            # Create messages
            response = self.client.chat.completions.create(
                model=self.deployment,
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
                                "text": "Extract the REV value from this engineering drawing:"
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
                temperature=0  # Deterministic
            )
            
            # Parse JSON response
            result_text = response.choices[0].message.content
            
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', result_text, re.DOTALL)
            if json_match:
                result_text = json_match.group(1)
            elif '```' in result_text:
                result_text = re.sub(r'```.*?```', '', result_text, flags=re.DOTALL)
            
            result_data = json.loads(result_text.strip())
            
            return RevResult(
                file=pdf_path.name,
                value=result_data.get("rev_value", ""),
                engine="gpt4o_vision",
                confidence=result_data.get("confidence", "unknown"),
                notes=result_data.get("notes", "")
            )
            
        except Exception as e:
            LOG.error(f"GPT-4o extraction failed for {pdf_path.name}: {e}")
            return RevResult(
                file=pdf_path.name,
                value="",
                engine="gpt4o_failed",
                confidence="none",
                notes=str(e)
            )

# ----------------------------- Pipeline ----------------------------------------

def run_gpt_only_pipeline(
    input_folder: Path,
    output_csv: Path,
    azure_endpoint: str,
    azure_key: str,
    deployment_name: str = "gpt-4o"
) -> List[Dict[str, Any]]:
    """
    GPT-4o-only pipeline: All PDFs processed with GPT-4o Vision.
    Advantages:
    - Very reliable
    - Handles all edge cases (scanned, A1, complex layouts)
    - Lower cost than Claude (~$0.01 vs $0.015-0.02 per page)
    
    Cost: ~$0.01 per page
    """
    rows: List[Dict[str, Any]] = []
    
    # Initialize GPT client
    gpt = AzureGPTExtractor(azure_endpoint, azure_key, deployment_name)
    
    # Get PDFs
    pdfs = list(input_folder.glob("*.pdf"))
    if not pdfs:
        LOG.warning(f"No PDFs found in {input_folder}")
        return rows
    
    for pdf_path in tqdm(pdfs, desc="Processing with GPT-4o"):
        try:
            result = gpt.extract_rev(pdf_path)
            
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
        LOG.info(f"Total cost: ≈${len(pdfs) * 0.01:.2f} (at $0.01/page)")
    except Exception as e:
        LOG.error(f"Failed to write CSV: {e}")
    
    return rows

def parse_args(argv=None):
    a = argparse.ArgumentParser(description="GPT-4o-only REV extraction")
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
        LOG.error("Azure credentials required. Set --azure-endpoint and --azure-key or env vars")
        return []
    
    return run_gpt_only_pipeline(
        args.input_folder,
        args.output,
        args.azure_endpoint,
        args.azure_key,
        args.deployment_name
    )

if __name__ == "__main__":
    main()
