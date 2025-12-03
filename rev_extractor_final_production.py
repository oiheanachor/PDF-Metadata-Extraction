#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REV Extractor — PyMuPDF + GPT-4.1 Production (FINAL)
Balanced prompt, smart rotation handling, optimized for 4000+ files
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

LOG = logging.getLogger("rev_extractor_production")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ----------------------------- BALANCED SYSTEM PROMPT -----------------------------------

GPT_SYSTEM_PROMPT = """You are an expert at analyzing engineering drawings and extracting revision information.

**YOUR TASK:**
Extract the REV (revision) value from the title block of this engineering drawing.

⚠️ CRITICAL: Read the ACTUAL value you see in THIS drawing. DO NOT memorize or default to example values!

**TITLE BLOCK IDENTIFICATION:**
The title block is typically in the bottom-right corner and contains:
- Company logo/name (ROTORK, FAIRCHILD, etc.)
- DWG NO / DRAWING NO / DWG. NO
- SHEET [X] OF [Y]
- SCALE (e.g., "1:1", "1:2")
- DRAWN BY / CHECKED BY / APPROVED BY
- The REV field is WITHIN or ADJACENT to this cluster

**CRITICAL RULES - AVOID THESE:**
❌ DO NOT extract from REVISION HISTORY TABLES (columns: REV | DESCRIPTION | DATE)
   - These show change history (multiple rows)
   - We want the CURRENT REV from title block (single value)
❌ DO NOT extract grid reference letters (A, B, C along edges)
❌ DO NOT extract section markers ("SECTION C-C", "SECTION B-B")
❌ DO NOT extract view indicators or part numbers

**REV VALUE FORMATS (All are equally valid):**

**NUMERIC REVISIONS** (hyphenated format):
- Pattern: [number]-[number]
- Valid examples: 1-0, 2-0, 3-0, 5-1, 12-01, 15-02
- ⚠️ Read the ACTUAL number - don't default to any specific value!
- These are EQUALLY VALID as letter revisions

**LETTER REVISIONS**:
- Single: A, B, C, D, ... Z
- Double: AA, AB, AC, etc.
- ⚠️ These are EQUALLY VALID as numeric revisions

**SPECIAL CASES**:
- Empty field or "OF" → return "EMPTY"
- No REV field exists → return "NO_REV"

**EXTRACTION STRATEGY:**
1. Locate title block using anchors (DWG NO, SHEET, SCALE, company name)
2. Find REV field within or adjacent to title block
3. Look for "REV:" or "REV" label
4. Extract the value immediately adjacent to or in that field
5. Verify it's not from revision history table or grid references

**VALIDATION CHECKLIST (use EVERY time):**
✓ Is it in the title block area?
✓ Is there a "REV:" or "REV" label nearby?
✓ Is it near DWG NO, SHEET, or SCALE?
✓ Is it a standalone value (not part of another number)?
✓ Does it match a valid format (letter OR hyphenated number)?
✓ Am I reading THIS drawing's actual value (not memorizing examples)?
✓ Is it NOT from a revision history table?
✓ Is it NOT a grid letter or section marker?

**RESPONSE FORMAT:**
{
  "rev_value": "2-0",
  "confidence": "high",
  "location": "bottom-right title block, adjacent to DWG NO: 21620",
  "notes": "Clear hyphenated numeric REV 2-0 in title block"
}

**DIVERSE EXAMPLES (showing ALL valid formats):**

Example 1 - Numeric REV "2-0" (common):
Title block: "DWG NO: 21620 | REV: 2-0 | SHEET 1"
✅ Output: "2-0"
Notes: "Hyphenated numeric REV 2-0 clearly visible in title block"

Example 2 - Numeric REV "3-0" (equally valid):
Title block: "DWG NO: 22416 | REV: 3-0 | SHEET 1"
✅ Output: "3-0"
Notes: "Hyphenated numeric REV 3-0 in title block"

Example 3 - Numeric REV "1-0" (equally valid):
Title block: "DRAWING NO: EB-12345 | REV: 1-0"
✅ Output: "1-0"

Example 4 - Numeric REV "5-1" (minor version):
Title block: "DWG NO: 21837 | REV: 5-1"
✅ Output: "5-1"

Example 5 - Single Letter "A":
Title block: "DWG NO: 21620 | REV: A | SHEET 1"
✅ Output: "A"
Notes: "Clear single letter REV A in title block"

Example 6 - Single Letter "F":
Title block: "DWG NO: EB-00131 | REV: F"
✅ Output: "F"

Example 7 - Single Letter "C":
Title block: "DWG NO: 18301 | REV: C"
✅ Output: "C"

Example 8 - Double Letter "AB":
Title block: "DRAWING NO: 14579 | REV: AB"
✅ Output: "AB"

Example 9 - Single Letter "E":
Title block: "DWG NO: 032-IPI-008 | REV: E"
✅ Output: "E"

Example 10 - NO_REV (no field exists):
Title block has DWG NO, SHEET, SCALE, but NO REV field anywhere
✅ Output: "NO_REV"
Notes: "Title block identified but no REV field present"

Example 11 - EMPTY (field exists but empty):
Title block: "DWG NO: 055-IPI-057 | REV: [empty/OF]"
✅ Output: "EMPTY"
Notes: "REV field present but empty or shows 'OF'"

Example 12 - False Positive (grid letters):
Drawing has "SECTION C-C" and letters A, B, C on edges
Title block has DWG NO but NO REV field
✅ Output: "NO_REV"
❌ Don't output: "C" (that's a section marker)

**CRITICAL REMINDERS:**
1. Numeric REVs (1-0, 2-0, 3-0, etc.) are EQUALLY VALID as letter REVs (A, B, C)
2. Check the TITLE BLOCK first, not revision history tables
3. Read the ACTUAL value in THIS drawing - every drawing is different
4. When you see "REV: 2-0" → output "2-0", NOT "A" or any other value
5. When you see "REV: A" → output "A", NOT "2-0" or any other value
6. Grid letters and section markers are NOT revision values

**CONFIDENCE LEVELS:**
- "high": Clear REV label with unambiguous value in title block
- "medium": Value found near title block elements but no explicit REV label
- "low": Uncertain - title block found but REV unclear

⚠️ ANTI-HALLUCINATION: Before responding, verify you are reading THIS drawing's actual value, not copying from examples!"""

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

# ----------------------------- Smart Rotation Detection ------------------------

def detect_and_validate_rotation(pdf_path: Path) -> Tuple[int, bool]:
    """
    Detect PDF rotation and validate if correction is actually needed.
    Returns: (rotation_angle, should_correct)
    
    Strategy: Only correct rotation if text orientation indicates it's needed.
    """
    try:
        with fitz.open(pdf_path) as doc:
            page = doc[0]
            metadata_rotation = page.rotation
            
            # If metadata says no rotation, trust it
            if metadata_rotation == 0:
                return 0, False
            
            # Check text orientation to see if rotation is really needed
            # Sample text from page to verify orientation
            try:
                text_blocks = page.get_text("dict")["blocks"]
                
                # If we can extract readable text, the page might already be correctly oriented
                # This is a heuristic: if text extraction works well, rotation might be visual-only
                text_count = sum(1 for block in text_blocks if block.get("type") == 0)
                
                if text_count > 10:
                    # Plenty of text extracted - page is likely readable as-is
                    # Only rotate if metadata rotation is significant (90, 270)
                    if metadata_rotation in [90, 270]:
                        LOG.debug(f"{pdf_path.name}: Metadata says {metadata_rotation}° but text is readable - may not need correction")
                        # Still apply rotation correction to be safe
                        return metadata_rotation, True
                    else:
                        return metadata_rotation, False
                else:
                    # Limited text - rotation is likely real
                    return metadata_rotation, True
                    
            except Exception:
                # If text extraction fails, trust metadata
                return metadata_rotation, True
                
    except Exception as e:
        LOG.warning(f"Could not detect rotation for {pdf_path.name}: {e}")
        return 0, False

def correct_rotation(pix: fitz.Pixmap, rotation: int) -> bytes:
    """
    Rotate a pixmap to correct orientation.
    rotation: current rotation angle (90, 180, or 270)
    Returns: PNG bytes of corrected image
    """
    if rotation == 0:
        return pix.tobytes("png")
    
    try:
        from PIL import Image
        import io
        
        # Convert pixmap to PIL Image
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))
        
        # Calculate correction angle
        # PyMuPDF rotation is counterclockwise from normal orientation
        # To correct: rotate in opposite direction
        correction_angle = (360 - rotation) % 360
        
        # Rotate (PIL rotation: positive = counterclockwise)
        # We want clockwise correction, so negate
        img_rotated = img.rotate(-correction_angle, expand=True)
        
        # Convert back to bytes
        buffer = io.BytesIO()
        img_rotated.save(buffer, format='PNG')
        return buffer.getvalue()
        
    except ImportError:
        LOG.warning("PIL not available, cannot rotate. Install: pip install pillow")
        return pix.tobytes("png")
    except Exception as e:
        LOG.warning(f"Rotation correction failed: {e}")
        return pix.tobytes("png")

# ----------------------------- GPT-4.1 Vision Extractor ------------------------

class AzureGPTExtractor:
    def __init__(self, endpoint: str, api_key: str, deployment_name: str = "gpt-4.1"):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai not installed. Run: pip install openai")
        
        # Clean endpoint
        endpoint = endpoint.rstrip('/')
        if '/openai/deployments' in endpoint:
            endpoint = endpoint.split('/openai/deployments')[0]
        
        LOG.info(f"Initializing GPT-4.1 Vision Extractor")
        LOG.info(f"Endpoint: {endpoint}")
        LOG.info(f"Deployment: {deployment_name}")
        
        try:
            self.client = AzureOpenAI(
                api_key=api_key,
                api_version="2024-02-15-preview",
                azure_endpoint=endpoint
            )
            self.deployment_name = deployment_name
            LOG.info("✓ GPT-4.1 client initialized")
        except Exception as e:
            LOG.error(f"Failed to initialize GPT client: {e}")
            raise
    
    def pdf_to_base64_image(self, pdf_path: Path, page_idx: int = 0, dpi: int = 150) -> Tuple[str, int, bool]:
        """
        Convert PDF page to base64-encoded PNG with smart rotation handling.
        Returns: (base64_string, rotation_detected, was_corrected)
        """
        # Detect rotation and whether correction is needed
        rotation, should_correct = detect_and_validate_rotation(pdf_path)
        
        with fitz.open(pdf_path) as doc:
            page = doc[page_idx]
            
            # Render page
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            # Apply rotation correction if needed
            if rotation != 0 and should_correct:
                LOG.debug(f"  Applying {rotation}° rotation correction for {pdf_path.name}")
                img_bytes = correct_rotation(pix, rotation)
                return base64.b64encode(img_bytes).decode('utf-8'), rotation, True
            else:
                # No correction needed
                png_bytes = pix.tobytes("png")
                return base64.b64encode(png_bytes).decode('utf-8'), rotation, False
    
    def extract_rev(self, pdf_path: Path) -> RevResult:
        """Extract REV using GPT-4.1 Vision with smart rotation handling."""
        try:
            # Convert PDF to image
            img_base64, rotation, was_corrected = self.pdf_to_base64_image(pdf_path, page_idx=0, dpi=150)
            
            if was_corrected:
                LOG.info(f"  Auto-corrected {rotation}° rotation for {pdf_path.name}")
            elif rotation != 0:
                LOG.debug(f"  Detected {rotation}° rotation but visual appears correct - no correction applied")
            
            # Build user message
            user_text = """Extract the REV value from this engineering drawing.

CRITICAL INSTRUCTIONS:
1. LOCATE the title block (bottom-right area with DWG NO, SHEET, SCALE)
2. FIND the REV field within or adjacent to the title block
3. READ the ACTUAL value you see in THIS drawing
4. BOTH numeric (2-0, 3-0) and letter (A, B, C) formats are equally valid
5. Output EXACTLY what you see - don't substitute or default to example values
6. If no REV field exists → return "NO_REV"

Remember: Every drawing is different. Read THIS drawing's value!"""
            
            # Call GPT-4.1
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
            
            # Parse JSON
            import json
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', result_text, re.DOTALL)
            if json_match:
                result_text = json_match.group(1)
            elif '```' in result_text:
                result_text = re.sub(r'```.*?```', '', result_text, flags=re.DOTALL)
            
            result_data = json.loads(result_text.strip())
            
            # Add rotation info to notes if corrected
            notes = result_data.get("notes", "")
            if was_corrected:
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
    deployment_name: str = "gpt-4.1"
) -> List[Dict[str, Any]]:
    """
    Production hybrid pipeline optimized for 4000+ files.
    """
    rows: List[Dict[str, Any]] = []
    
    # Initialize GPT client
    gpt = AzureGPTExtractor(azure_endpoint, azure_key, deployment_name)
    
    # Scan for PDFs
    pdfs = sorted(input_folder.glob("*.pdf"))  # Sort for consistent processing
    if not pdfs:
        LOG.warning(f"No PDFs found in {input_folder}")
        return rows
    
    LOG.info(f"Found {len(pdfs)} PDFs to process")
    
    native_success = 0
    gpt_used = 0
    gpt_failed = 0
    rotated_corrected = 0
    
    for pdf_path in tqdm(pdfs, desc="Processing PDFs"):
        try:
            # Step 1: Try native PyMuPDF extraction (fast, free)
            result = extract_native_pymupdf(pdf_path)
            
            if result and result.value and result.value not in ["", "NO_REV"]:
                native_success += 1
                rows.append({
                    "file": result.file,
                    "value": result.value,
                    "actual": result.value,  # For comparison
                    "engine": result.engine,
                    "confidence": result.confidence,
                    "notes": result.notes[:100]
                })
                LOG.debug(f"✓ Native: {pdf_path.name} → {result.value}")
                continue
            
            # Step 2: Fall back to GPT-4.1 Vision
            LOG.info(f"→ GPT: {pdf_path.name}")
            gpt_used += 1
            result = gpt.extract_rev(pdf_path)
            
            if result.engine == "gpt_failed":
                gpt_failed += 1
            elif "[Auto-corrected" in result.notes:
                rotated_corrected += 1
            
            rows.append({
                "file": result.file,
                "value": result.value,
                "actual": result.value,  # For comparison
                "engine": result.engine,
                "confidence": result.confidence,
                "notes": result.notes[:100]
            })
            
        except Exception as e:
            LOG.error(f"Failed {pdf_path.name}: {e}")
            rows.append({
                "file": pdf_path.name,
                "value": "",
                "actual": "",
                "engine": "error",
                "confidence": "none",
                "notes": str(e)[:100]
            })
    
    # Write results
    try:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, 'w', newline='', encoding='utf-8-sig') as outf:
            writer = csv.DictWriter(outf, fieldnames=['file', 'value', 'actual', 'engine', 'confidence', 'notes'])
            writer.writeheader()
            writer.writerows(rows)
        
        LOG.info(f"\n{'='*60}")
        LOG.info(f"Wrote {output_csv.resolve()}")
        LOG.info(f"{'='*60}")
        LOG.info(f"Total files: {len(rows)}")
        LOG.info(f"Native (free): {native_success}")
        LOG.info(f"GPT-4.1 (paid): {gpt_used} (rotated: {rotated_corrected})")
        LOG.info(f"Failed: {gpt_failed}")
        if gpt_used > 0:
            gpt_cost = (gpt_used - gpt_failed) * 0.010
            LOG.info(f"Estimated cost: ${gpt_cost:.2f}")
            if len(pdfs) > 100:
                total_estimated = (len(pdfs) * 0.010 * (gpt_used / len(rows)))
                LOG.info(f"If all {len(pdfs)} files needed GPT: ~${total_estimated:.2f}")
        LOG.info(f"{'='*60}\n")
    except Exception as e:
        LOG.error(f"Failed to write CSV: {e}")
    
    return rows

def parse_args(argv=None):
    a = argparse.ArgumentParser(
        description="Production REV extractor: PyMuPDF + GPT-4.1 (optimized for 4000+ files)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    a.add_argument("input_folder", type=Path, help="Folder containing PDF files")
    a.add_argument("-o", "--output", type=Path, default=Path("rev_results.csv"),
                   help="Output CSV file (default: rev_results.csv)")
    a.add_argument("--azure-endpoint", type=str,
                   default=os.getenv("AZURE_OPENAI_ENDPOINT"),
                   help="Azure OpenAI endpoint URL")
    a.add_argument("--azure-key", type=str,
                   default=os.getenv("AZURE_OPENAI_KEY"),
                   help="Azure OpenAI API key")
    a.add_argument("--deployment-name", type=str,
                   default="gpt-4.1",
                   help="Azure OpenAI deployment name (default: gpt-4.1)")
    return a.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    
    if not args.azure_endpoint or not args.azure_key:
        LOG.error("=" * 60)
        LOG.error("ERROR: Azure credentials required")
        LOG.error("=" * 60)
        LOG.error("Set environment variables:")
        LOG.error("  $env:AZURE_OPENAI_ENDPOINT = 'https://your-endpoint.com'")
        LOG.error("  $env:AZURE_OPENAI_KEY = 'your-api-key'")
        LOG.error("\nOr use command-line flags:")
        LOG.error("  --azure-endpoint 'https://...' --azure-key 'your-key'")
        LOG.error("=" * 60)
        return []
    
    LOG.info("=" * 60)
    LOG.info("REV EXTRACTOR - Production Mode")
    LOG.info("=" * 60)
    LOG.info(f"Input folder: {args.input_folder}")
    LOG.info(f"Output CSV: {args.output}")
    LOG.info(f"Deployment: {args.deployment_name}")
    LOG.info("=" * 60 + "\n")
    
    return run_hybrid_pipeline(
        args.input_folder,
        args.output,
        args.azure_endpoint,
        args.azure_key,
        args.deployment_name
    )

if __name__ == "__main__":
    main()
