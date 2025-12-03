#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REV Extractor — PyMuPDF + GPT-4o with Rotation Handling (PRODUCTION)
Handles rotated PDFs intelligently: detects rotation, corrects, then extracts
"""

from __future__ import annotations
import argparse, base64, csv, logging, os, re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from tqdm import tqdm

# OpenAI SDK for Azure
try:
    from openai import AzureOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

LOG = logging.getLogger("rev_extractor_gpt_production")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ----------------------------- PRODUCTION SYSTEM PROMPT -----------------------------------

GPT_SYSTEM_PROMPT = """You are an expert at analyzing engineering drawings and extracting revision information.

**YOUR TASK:**
Extract the REV (revision) value from the title block of this engineering drawing.

⚠️ CRITICAL: Read the ACTUAL value you see. DO NOT use example values as templates!

**DRAWING ORIENTATION:**
The drawing image has been pre-processed to correct rotation. The title block should now be in a standard readable orientation (typically bottom-right corner). If you still find the drawing difficult to read, look for title block elements at any edge.

**TITLE BLOCK IDENTIFICATION (Rotation-Aware):**
The title block can appear at any edge of the drawing. Look for these anchor elements:
- Company logo/name (ROTORK, FAIRCHILD, etc.)
- DWG NO / DRAWING NO / DWG. NO
- SHEET [X] OF [Y]
- SCALE (e.g., "1:1", "1:2")
- DRAWN BY / CHECKED BY / APPROVED BY / DATE fields
- Drawing number (format: XX-XXXXX or similar)

The REV field is typically ADJACENT to or WITHIN this title block cluster.

**REV FIELD LOCATION STRATEGIES:**
1. **Primary**: Look near "DWG NO" - REV is often immediately adjacent
2. **Secondary**: Look near "SHEET" field - REV often nearby
3. **Tertiary**: Scan the title block area for explicit "REV:" label
4. **Last resort**: Look for isolated single letters (A-Z) or hyphenated numbers (X-X) near company name

**AVOID THESE (Critical):**
❌ Revision history tables (columns: REV | DESCRIPTION | DATE | EC)
❌ Grid reference letters (A, B, C along edges without context)
❌ Section markers ("SECTION C-C", "SECTION B-B")
❌ View indicators ("DETAIL A", "VIEW B")
❌ Part numbers or item callouts
❌ Example values from this prompt (don't output "2-0" or "3-0" unless actually present)

**REV VALUE FORMATS:**

**NUMERIC REVISIONS** (Read carefully - don't default to examples!):
- Hyphenated: [number]-[number]
- Real examples: 1-0, 3-0, 5-1, 12-01, 15-02
- ⚠️ Output the EXACT number you see, not a template value!

**LETTER REVISIONS**:
- Single: A, B, C, ... Z
- Double: AA, AB, etc.

**SPECIAL CASES**:
- Empty field or "OF" in REV field → "EMPTY"
- No REV field exists anywhere in title block → "NO_REV"
- REV field present but value unclear/illegible → "NO_REV" with low confidence

**VALIDATION CHECKLIST:**
✓ Found title block using anchor elements?
✓ REV field is adjacent to or within title block?
✓ Value matches valid format (letter or hyphenated number)?
✓ NOT from revision history table?
✓ NOT a grid letter or section marker?
✓ Reading ACTUAL value (not copying examples)?

**RESPONSE FORMAT:**
{
  "rev_value": "3-0",
  "confidence": "high",
  "location": "bottom-right title block, adjacent to DWG NO: 22416",
  "notes": "Clear hyphenated numeric REV 3-0, distinct from grid references"
}

**EXAMPLES (DIVERSE VALUES - DON'T MEMORIZE THESE):**

Example 1 - Numeric REV 3-0:
Title block contains: "DWG NO: 22416 | REV: 3-0 | SHEET 1 OF 1"
✅ Output: "3-0"
❌ Don't output: "2-0" (that's a different drawing's value!)

Example 2 - Numeric REV 1-0:
Title block contains: "DRAWING NO: EB-12345 | REV: 1-0"
✅ Output: "1-0"
❌ Don't output: "3-0" or "2-0" (those are from other drawings!)

Example 3 - Numeric REV 5-1:
Title block contains: "DWG NO: 21837 | REV: 5-1"
✅ Output: "5-1"

Example 4 - Letter REV:
Title block contains: "DWG NO: EB-00131 | REV: F | SHEET 1"
✅ Output: "F"

Example 5 - Double Letter:
Title block contains: "DRAWING NO: 14579 | REV: AB"
✅ Output: "AB"

Example 6 - Empty REV:
Title block contains: "DWG NO: 055-IPI-057 | REV: [empty or 'OF']"
✅ Output: "EMPTY"
Notes: "REV field present but empty or shows 'OF'"

Example 7 - No REV Field:
Title block present with DWG NO, SHEET, SCALE, but no REV field anywhere
✅ Output: "NO_REV"
Notes: "Title block found but no REV field exists"

Example 8 - Grid Letter (FALSE POSITIVE):
Drawing has "SECTION C-C" label and letters "A", "B", "C" on edges
Title block present but NO REV field
✅ Output: "NO_REV"
❌ Don't output: "C" (that's a section marker, not a REV!)

**ANTI-HALLUCINATION SAFEGUARDS:**
Before responding, verify:
1. Did I FIND the title block using anchor elements?
2. Is there a REV field or label in/near the title block?
3. Am I reading the ACTUAL value in THIS drawing?
4. Or am I defaulting to an example value like "2-0" or "3-0"?
5. If uncertain → return "NO_REV" with medium/low confidence

**CONFIDENCE LEVELS:**
- "high": Clear REV label with unambiguous value in identified title block
- "medium": Value found in title block area but no explicit REV label
- "low": Uncertain - title block found but REV unclear
- "none": No title block or REV field found → return "NO_REV"

⚠️ FINAL REMINDER: The image may have been rotated to correct orientation. Focus on FINDING the title block using anchor elements (DWG NO, SHEET, company name), then extract the REV value from within or adjacent to that cluster."""

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

# ----------------------------- Rotation Detection & Correction -----------------

def detect_pdf_rotation(pdf_path: Path) -> int:
    """
    Detect if a PDF page is rotated.
    Returns rotation angle: 0, 90, 180, or 270 degrees
    """
    try:
        with fitz.open(pdf_path) as doc:
            page = doc[0]
            # PyMuPDF stores rotation in page.rotation
            rotation = page.rotation
            LOG.debug(f"{pdf_path.name} - Detected rotation: {rotation}°")
            return rotation
    except Exception as e:
        LOG.warning(f"Could not detect rotation for {pdf_path.name}: {e}")
        return 0

def correct_rotation(pix: fitz.Pixmap, rotation: int) -> fitz.Pixmap:
    """
    Rotate a pixmap to correct orientation (0°).
    rotation: current rotation angle (90, 180, or 270)
    """
    if rotation == 0:
        return pix
    
    # PyMuPDF rotation is counterclockwise, we need to rotate back
    # If page is rotated 90° clockwise → need to rotate -90° (or +270°)
    correction = (360 - rotation) % 360
    
    if correction == 0:
        return pix
    
    LOG.debug(f"Applying {correction}° correction")
    
    # Note: PyMuPDF's rotation is in 90° increments
    # We need to convert pixmap to PIL, rotate, convert back
    try:
        from PIL import Image
        import io
        
        # Convert pixmap to PIL Image
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))
        
        # Rotate (PIL rotation is counterclockwise)
        # correction is already calculated correctly
        img_rotated = img.rotate(-correction, expand=True)  # Negative for clockwise
        
        # Convert back to bytes
        img_buffer = io.BytesIO()
        img_rotated.save(img_buffer, format='PNG')
        img_bytes_rotated = img_buffer.getvalue()
        
        # Create new pixmap from rotated image
        # Note: We'll return the bytes, not pixmap, for simplicity
        return img_bytes_rotated
        
    except ImportError:
        LOG.warning("PIL not available, cannot rotate image. Install: pip install pillow")
        return pix.tobytes("png")
    except Exception as e:
        LOG.warning(f"Rotation correction failed: {e}")
        return pix.tobytes("png")

# ----------------------------- GPT-4 Vision Extractor --------------------------

class AzureGPTExtractor:
    def __init__(self, endpoint: str, api_key: str, deployment_name: str = "gpt-4o"):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai not installed. Run: pip install openai")
        
        # Clean endpoint
        endpoint = endpoint.rstrip('/')
        if '/openai/deployments' in endpoint:
            endpoint = endpoint.split('/openai/deployments')[0]
        
        LOG.info(f"Initializing GPT-4o with rotation handling")
        LOG.info(f"Endpoint: {endpoint}")
        LOG.info(f"Deployment: {deployment_name}")
        
        try:
            self.client = AzureOpenAI(
                api_key=api_key,
                api_version="2024-02-15-preview",
                azure_endpoint=endpoint
            )
            self.deployment_name = deployment_name
            LOG.info("✓ GPT client initialized")
        except Exception as e:
            LOG.error(f"Failed to initialize GPT client: {e}")
            raise
    
    def pdf_to_base64_image(self, pdf_path: Path, page_idx: int = 0, dpi: int = 150,
                           auto_rotate: bool = True) -> Tuple[str, int]:
        """
        Convert PDF page to base64-encoded PNG with optional rotation correction.
        Returns: (base64_string, rotation_applied)
        """
        with fitz.open(pdf_path) as doc:
            page = doc[page_idx]
            
            # Detect rotation
            rotation = page.rotation if auto_rotate else 0
            
            # Render page to pixmap
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            # Correct rotation if needed
            if rotation != 0 and auto_rotate:
                LOG.info(f"  Correcting {rotation}° rotation for {pdf_path.name}")
                img_bytes = correct_rotation(pix, rotation)
                if isinstance(img_bytes, bytes):
                    # Already rotated bytes
                    return base64.b64encode(img_bytes).decode('utf-8'), rotation
                else:
                    # Fallback: couldn't rotate
                    png_bytes = pix.tobytes("png")
                    return base64.b64encode(png_bytes).decode('utf-8'), 0
            else:
                png_bytes = pix.tobytes("png")
                return base64.b64encode(png_bytes).decode('utf-8'), rotation
    
    def extract_rev(self, pdf_path: Path) -> RevResult:
        """Extract REV using GPT-4o Vision with rotation handling."""
        try:
            # Convert PDF to image with auto-rotation
            img_base64, rotation = self.pdf_to_base64_image(pdf_path, page_idx=0, dpi=150, auto_rotate=True)
            
            rotation_note = f" (corrected {rotation}° rotation)" if rotation != 0 else ""
            LOG.debug(f"Processing {pdf_path.name}{rotation_note}")
            
            # Build user message
            user_text = """Extract the REV value from this engineering drawing.

CRITICAL INSTRUCTIONS:
1. FIND the title block using anchor elements (DWG NO, SHEET, SCALE, company name)
2. LOCATE the REV field within or adjacent to the title block
3. READ the ACTUAL value in THIS specific drawing
4. DO NOT copy example values like "2-0" or "3-0" from the prompt
5. If title block found but no REV field exists → return "NO_REV"
6. If REV field exists but value is unclear → return "NO_REV" with low confidence

The image has been pre-processed to correct any rotation."""
            
            # Call GPT-4o
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": GPT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_text},
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
            
            result_text = response.choices[0].message.content
            
            # Parse JSON response
            import json
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', result_text, re.DOTALL)
            if json_match:
                result_text = json_match.group(1)
            elif '```' in result_text:
                result_text = re.sub(r'```.*?```', '', result_text, flags=re.DOTALL)
            
            result_data = json.loads(result_text.strip())
            
            # Add rotation info to notes
            notes = result_data.get("notes", "")
            if rotation != 0:
                notes = f"[Auto-corrected {rotation}° rotation] {notes}"
            
            return RevResult(
                file=pdf_path.name,
                value=result_data.get("rev_value", ""),
                engine="gpt_vision",
                confidence=result_data.get("confidence", "unknown"),
                notes=notes[:200]
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

# ----------------------------- Pipeline ----------------------------------------

def run_hybrid_pipeline(
    input_folder: Path,
    output_csv: Path,
    azure_endpoint: str,
    azure_key: str,
    deployment_name: str = "gpt-4o"
) -> List[Dict[str, Any]]:
    """
    Hybrid pipeline with rotation handling:
    1. Try PyMuPDF native extraction (fast, free)
    2. Fall back to GPT-4o Vision with auto-rotation (robust, paid)
    """
    rows: List[Dict[str, Any]] = []
    
    # Initialize GPT-4o client
    gpt = AzureGPTExtractor(azure_endpoint, azure_key, deployment_name)
    
    # Scan for PDFs
    pdfs = list(input_folder.glob("*.pdf"))
    if not pdfs:
        LOG.warning(f"No PDFs found in {input_folder}")
        return rows
    
    LOG.info(f"Found {len(pdfs)} PDFs to process")
    
    native_success = 0
    gpt_used = 0
    gpt_failed = 0
    rotated_handled = 0
    
    for pdf_path in tqdm(pdfs, desc="Scanning PDFs"):
        try:
            # Step 1: Try native PyMuPDF extraction
            result = extract_native_pymupdf(pdf_path)
            
            if result and result.value and result.value not in ["", "NO_REV"]:
                # Native succeeded
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
            
            # Step 2: Fall back to GPT-4o with rotation handling
            rotation = detect_pdf_rotation(pdf_path)
            if rotation != 0:
                rotated_handled += 1
                LOG.info(f"→ Using GPT for {pdf_path.name} (rotated {rotation}°)")
            else:
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
    
    # Write results
    try:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, 'w', newline='', encoding='utf-8-sig') as outf:
            writer = csv.DictWriter(outf, fieldnames=['file', 'value', 'engine', 'confidence', 'notes'])
            writer.writeheader()
            writer.writerows(rows)
        
        LOG.info(f"Wrote {output_csv.resolve()} with {len(rows)} rows")
        LOG.info(f"Stats: Native={native_success}, GPT={gpt_used} (rotated={rotated_handled}), Failed={gpt_failed}")
        if gpt_used > 0:
            gpt_cost = (gpt_used - gpt_failed) * 0.010
            LOG.info(f"Cost≈${gpt_cost:.2f}")
    except Exception as e:
        LOG.error(f"Failed to write CSV: {e}")
    
    return rows

def parse_args(argv=None):
    a = argparse.ArgumentParser(description="PyMuPDF + GPT-4o with rotation handling (production)")
    a.add_argument("input_folder", type=Path, help="Folder containing PDF files")
    a.add_argument("-o", "--output", type=Path, default=Path("rev_results.csv"),
                   help="Output CSV file path")
    a.add_argument("--azure-endpoint", type=str,
                   default=os.getenv("AZURE_OPENAI_ENDPOINT"),
                   help="Azure OpenAI endpoint URL")
    a.add_argument("--azure-key", type=str,
                   default=os.getenv("AZURE_OPENAI_KEY"),
                   help="Azure OpenAI API key")
    a.add_argument("--deployment-name", type=str,
                   default="gpt-4o",
                   help="Azure OpenAI deployment name (e.g., gpt-4.1, gpt-4o)")
    return a.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    
    if not args.azure_endpoint or not args.azure_key:
        LOG.error("Azure credentials required. Set --azure-endpoint and --azure-key or env vars:")
        LOG.error("  $env:AZURE_OPENAI_ENDPOINT = 'https://...'")
        LOG.error("  $env:AZURE_OPENAI_KEY = 'your_key'")
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
