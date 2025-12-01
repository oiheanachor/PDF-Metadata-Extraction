#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REV Extractor — PyMuPDF + Azure Claude Fallback
Hybrid: Try native extraction first, fall back to Claude Vision for failures
"""

from __future__ import annotations
import argparse, base64, csv, logging, os, re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
from tqdm import tqdm

# Azure AI Inference SDK
try:
    from azure.ai.inference import ChatCompletionsClient
    from azure.ai.inference.models import (
        UserMessage, SystemMessage,
        ImageContentItem, TextContentItem
    )
    from azure.core.credentials import AzureKeyCredential
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

LOG = logging.getLogger("rev_extractor_claude_hybrid")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ----------------------------- Configuration -----------------------------------

CLAUDE_SYSTEM_PROMPT = """You are an expert at analyzing engineering drawings and extracting revision information.

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

IMPORTANT: Focus on the TITLE BLOCK (bottom-right), NOT the revision history table (top-right)."""

# ----------------------------- Data Structures ---------------------------------

@dataclass
class RevResult:
    file: str
    value: str
    engine: str
    confidence: str = "unknown"
    notes: str = ""

# ----------------------------- PyMuPDF Native Extraction -----------------------

def extract_native_pymupdf(pdf_path: Path) -> Optional[RevResult]:
    """
    Try native PyMuPDF extraction using the fixed logic.
    Returns RevResult if successful, None if fails.
    """
    try:
        # Import the fixed extractor logic
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

# ----------------------------- Claude Vision -----------------------------------

class AzureClaudeExtractor:
    def __init__(self, endpoint: str, api_key: str, model: str = "claude-sonnet-4-20250514"):
        if not AZURE_AVAILABLE:
            raise ImportError("azure-ai-inference not installed. Run: pip install azure-ai-inference")
        
        self.client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key)
        )
        self.model = model
    
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
        """Extract REV using Claude Vision."""
        try:
            # Convert PDF to image
            img_base64 = self.pdf_to_base64_image(pdf_path, page_idx=0, dpi=150)
            
            # Create messages
            user_content = [
                TextContentItem(text="Extract the REV value from this engineering drawing:"),
                ImageContentItem(image_url=f"data:image/png;base64,{img_base64}")
            ]
            
            messages = [
                SystemMessage(content=CLAUDE_SYSTEM_PROMPT),
                UserMessage(content=user_content)
            ]
            
            # Call Claude
            response = self.client.complete(
                messages=messages,
                model=self.model,
                max_tokens=500,
                temperature=0  # Deterministic
            )
            
            # Parse JSON response
            import json
            result_text = response.choices[0].message.content
            
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', result_text, re.DOTALL)
            if json_match:
                result_text = json_match.group(1)
            elif '```' in result_text:
                # Remove any code blocks
                result_text = re.sub(r'```.*?```', '', result_text, flags=re.DOTALL)
            
            result_data = json.loads(result_text.strip())
            
            return RevResult(
                file=pdf_path.name,
                value=result_data.get("rev_value", ""),
                engine="claude_vision",
                confidence=result_data.get("confidence", "unknown"),
                notes=result_data.get("notes", "")
            )
            
        except Exception as e:
            LOG.error(f"Claude extraction failed for {pdf_path.name}: {e}")
            return RevResult(
                file=pdf_path.name,
                value="",
                engine="claude_failed",
                confidence="none",
                notes=str(e)
            )

# ----------------------------- Pipeline ----------------------------------------

def run_hybrid_pipeline(
    input_folder: Path,
    output_csv: Path,
    azure_endpoint: str,
    azure_key: str,
    claude_model: str = "claude-sonnet-4-20250514"
) -> List[Dict[str, Any]]:
    """
    Hybrid pipeline:
    1. Try PyMuPDF native extraction (fast, free)
    2. Fall back to Claude Vision (robust, paid)
    """
    rows: List[Dict[str, Any]] = []
    
    # Initialize Claude client
    claude = AzureClaudeExtractor(azure_endpoint, azure_key, claude_model)
    
    # Get PDFs
    pdfs = list(input_folder.glob("*.pdf"))
    if not pdfs:
        LOG.warning(f"No PDFs found in {input_folder}")
        return rows
    
    native_success = 0
    claude_used = 0
    
    for pdf_path in tqdm(pdfs, desc="Processing PDFs"):
        try:
            # Step 1: Try PyMuPDF native
            result = extract_native_pymupdf(pdf_path)
            
            if result and result.value and result.value not in ["", "NO_REV"]:
                # Native extraction succeeded
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
            
            # Step 2: Fall back to Claude
            LOG.info(f"→ Using Claude for {pdf_path.name}")
            claude_used += 1
            result = claude.extract_rev(pdf_path)
            
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
        LOG.info(f"Stats: Native={native_success}, Claude={claude_used}, Cost≈${claude_used * 0.015:.2f}")
    except Exception as e:
        LOG.error(f"Failed to write CSV: {e}")
    
    return rows

def parse_args(argv=None):
    a = argparse.ArgumentParser(description="PyMuPDF + Claude hybrid extraction")
    a.add_argument("input_folder", type=Path)
    a.add_argument("-o", "--output", type=Path, default=Path("rev_results.csv"))
    a.add_argument("--azure-endpoint", type=str, 
                   default=os.getenv("AZURE_AI_ENDPOINT"),
                   help="Azure AI Foundry endpoint URL")
    a.add_argument("--azure-key", type=str,
                   default=os.getenv("AZURE_AI_KEY"),
                   help="Azure AI Foundry API key")
    a.add_argument("--claude-model", type=str,
                   default="claude-sonnet-4-20250514",
                   help="Claude model name")
    return a.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    
    if not args.azure_endpoint or not args.azure_key:
        LOG.error("Azure credentials required. Set --azure-endpoint and --azure-key or AZURE_AI_ENDPOINT and AZURE_AI_KEY env vars")
        return []
    
    return run_hybrid_pipeline(
        args.input_folder,
        args.output,
        args.azure_endpoint,
        args.azure_key,
        args.claude_model
    )

if __name__ == "__main__":
    main()
