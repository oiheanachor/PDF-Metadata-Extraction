#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REV Extractor — PyMuPDF + Azure GPT-4 Fallback (OPTIMIZED)
Fixed to handle: numeric REVs (2-0, 3-0), NO_REV cases, and false positives
"""

from __future__ import annotations
import argparse, base64, csv, logging, os, re, json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import time

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

# ---------------------------------------------------------------------
# GPT SYSTEM PROMPT
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
   - DO NOT extract view indicators

3. REV Value Formats (in order of priority):

   NUMERIC REVISIONS (Common in some companies):
   - Hyphenated: 1-0, 2-0, 3-0, 12-01, 15-02
   - Format: [number]-[number]
   - Often used for major-minor version (2-0 = version 2.0)

   LETTER REVISIONS:
   - Single letter: A, B, C, D, ... Z
   - Double letters: AA, AB, AC, etc.

   Special Cases:
   - "OF" in REV field → return "EMPTY" (means not applicable)
   - No REV field or marking → return "NO_REV"

4. Validation Checklist:
   - Is it in the title block (bottom corner)?
   - Is there a "REV:" or "REV" label nearby?
   - Is it near DWG NO, SHEET, or SCALE fields?
   - Is it isolated (not part of another number)?
   - Does it follow a valid format (letter or hyphenated number)?

RESPONSE FORMAT:
Return ONLY a JSON object like:
{
  "rev_value": "2-0",
  "confidence": "high",
  "location": "bottom-right title block, next to DWG NO field",
  "notes": "Clear hyphenated numeric REV 2-0, distinct from drawing number"
}"""

# ---------------------------------------------------------------------
# DATA CLASS
# ---------------------------------------------------------------------
@dataclass
class RevResult:
    file: str
    value: str
    engine: str
    confidence: str = "unknown"
    notes: str = ""

# ---------------------------------------------------------------------
# NATIVE EXTRACTION (YOUR EXISTING WORKING CODE)
# ---------------------------------------------------------------------
def extract_native_pymupdf(pdf_path: Path) -> Optional[RevResult]:
    """Try native PyMuPDF extraction using the fixed logic from rev_extractor_fixed."""
    try:
        from rev_extractor_fixed_v2 import (process_pdf_native, _normalize_output_value,
            DEFAULT_BR_X, DEFAULT_BR_Y, DEFAULT_EDGE_MARGIN, DEFAULT_REV_2L_BLOCKLIST, canonicalize_rev_value, is_plausible_rev_value, is_suspicious_rev_value)
        
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

# ---------------------------------------------------------------------
# GPT VALIDATION HELPERS
# ---------------------------------------------------------------------

# Pattern to detect hyphenated numeric REVs like 1-0, 2-0, 12-01, etc.
REV_NUMERIC_PATTERN = re.compile(r"\b\d{1,3}-\d{1,2}\b")

def _normalize_rev_value(v: str) -> str:
    """Normalise model/OCR output to our canonical REV value set."""
    if v is None:
        return "NO_REV"
    s = str(v).strip()

    # normalise unicode dashes
    s = (s.replace("–", "-")
           .replace("—", "-")
           .replace("−", "-")
           .replace("‑", "-")
           .replace("﹣", "-")
           .replace("－", "-"))
    s = s.replace("‾", "_").replace("¯", "_")

    # Empty / common placeholders
    if s == "" or s.upper() in {"NONE", "N/A", "NA"}:
        return "NO_REV"

    s_up = s.upper()

    # Some fonts/ocr can read 0 as O or vice-versa; don't "correct" aggressively here
    # (we rely on plausibility + anchors instead)

    # Canonicalise dash/underscore variants
    if re.fullmatch(r"-+", s_up):
        return "-"
    if re.fullmatch(r"_+", s_up):
        return "_"
    if re.fullmatch(r"\.-+", s_up):
        return ".-"
    if re.fullmatch(r"\._+", s_up):
        return "._"

    # Dot+letter like .A
    if re.fullmatch(r"\.[A-Z]{1,2}", s_up):
        return s_up

    # Letters (A, AB, etc.)
    if re.fullmatch(r"[A-Z]{1,2}", s_up):
        return s_up

    # Numeric (1-0 etc.) keep as-is
    if re.fullmatch(r"\d{1,3}-\d{1,3}", s_up):
        return s_up

    if s_up in {"NO_REV", "OF"}:
        return "NO_REV"

    return s_up


def _validate_gpt_result(pdf_path: Path, result_data: Dict[str, Any]) -> RevResult:
    """
    Validate and correct GPT output using the real PDF text (if available).

    Behaviour:
    - If there is NO text layer (scanned-only), we simply trust GPT (after normalisation).
    - If GPT predicts numeric hyphenated REV (e.g. 2-0, 3-0), we ensure it appears in text.
      * If not, but there is exactly one numeric candidate near 'REV', we correct to that.
      * If not and there are 0 or >1 candidates, we force NO_REV (low confidence).
    - If GPT predicts NO_REV / EMPTY and there is exactly one numeric REV near 'REV' in text,
      we override GPT with that numeric value (medium confidence).
    """
    raw_value = result_data.get("rev_value", "")
    value = _normalize_rev_value(raw_value)
    confidence = (result_data.get("confidence") or "unknown").lower()
    notes = (result_data.get("notes") or "").strip()
    engine = "gpt_vision"

    # Read page text once. This is the text _validate_gpt_result uses.
    try:
        with fitz.open(pdf_path) as doc:
            page_text = (doc[0].get_text("text") or "")
    except Exception:
        page_text = ""
    text_upper = page_text.upper()

    # If there is no text layer at all, we cannot validate against text.
    # In that case we just trust GPT's output (after normalisation).
    if not page_text.strip():
        if not notes:
            notes = "No text layer; GPT result not validated against PDF text"
        return RevResult(
            file=pdf_path.name,
            value=value,
            engine=engine,
            confidence=confidence or "unknown",
            notes=notes,
        )

    # Collect numeric candidates from the text
    numeric_candidates = set(REV_NUMERIC_PATTERN.findall(text_upper))

    # ------------------------------
    # 1) GPT said NO_REV / EMPTY
    # ------------------------------
    if value in {"NO_REV", "EMPTY"}:
        # Try to salvage: if there's exactly ONE numeric REV near a REV label,
        # assume GPT was too conservative and use that.
        if len(numeric_candidates) == 1:
            candidate = next(iter(numeric_candidates))
            near_pattern = re.compile(
                rf"REV[^A-Z0-9]{{0,12}}{re.escape(candidate)}\b",
                re.IGNORECASE,
            )
            if near_pattern.search(page_text):
                if notes:
                    notes += " | "
                notes += (
                    f"GPT returned {value} but a single numeric REV {candidate} "
                    "was found near a 'REV' label in page text; overriding."
                )
                return RevResult(
                    file=pdf_path.name,
                    value=candidate,
                    engine=engine,
                    confidence="medium",
                    notes=notes,
                )

        # Otherwise, keep GPT's NO_REV/EMPTY but tidy confidence a bit
        if "REV" not in text_upper and confidence in {"unknown", "low"}:
            confidence = "medium"

        if not notes:
            notes = "Explicit NO_REV/EMPTY from GPT"
        return RevResult(
            file=pdf_path.name,
            value=value,
            engine=engine,
            confidence=confidence,
            notes=notes,
        )

    # ------------------------------
    # 2) GPT predicted numeric REV
    # ------------------------------
    if REV_NUMERIC_PATTERN.fullmatch(value or ""):
        if value in numeric_candidates:
            # Supported by page text – good
            if confidence in {"unknown", ""}:
                confidence = "high"
        elif len(numeric_candidates) == 1:
            # Snap to the single real candidate on the page
            real_value = next(iter(numeric_candidates))
            if notes:
                notes += " | "
            notes += (
                f"GPT suggested {value}, corrected to {real_value} based on page text"
            )
            value = real_value
            confidence = "high"
        else:
            # Numeric hallucination
            if notes:
                notes += " | "
            notes += "GPT numeric value not found on page; forcing NO_REV"
            return RevResult(
                file=pdf_path.name,
                value="NO_REV",
                engine=engine,
                confidence="low",
                notes=notes,
            )

        return RevResult(
            file=pdf_path.name,
            value=value,
            engine=engine,
            confidence=confidence or "unknown",
            notes=notes,
        )

    # ------------------------------
    # 3) Letter or other non-numeric value
    # ------------------------------
    if value:
        near_rev_pattern = re.compile(
            rf"REV[^A-Z0-9]{{0,8}}{re.escape(value)}\b",
            re.IGNORECASE,
        )
        if not near_rev_pattern.search(page_text):
            if notes:
                notes += " | "
            notes += "Value not found near a 'REV' label in text; may be unreliable"
            if confidence == "high":
                confidence = "medium"

    return RevResult(
        file=pdf_path.name,
        value=value,
        engine=engine,
        confidence=confidence or "unknown",
        notes=notes,
    )

# ---------------------------------------------------------------------
# GPT EXTRACTOR
# ---------------------------------------------------------------------
class AzureGPTExtractor:
    def __init__(self, endpoint: str, api_key: str, deployment_name: str = "gpt-4.1"):
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
    
    def pdf_to_base64_image(self, pdf_path: Path, page_idx: int = 0, dpi: int = 300) -> str:
        """
        Convert PDF page to base64-encoded PNG.

        We crop to the bottom-right ~quarter of the page to focus GPT on the title block,
        which helps it see small numeric REVs like 2-0 / 3-0 and ignore other clutter.
        """
        with fitz.open(pdf_path) as doc:
            page = doc[page_idx]
            rect = page.rect
            # bottom-right quarter of the page
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
        """Extract REV using GPT-4 Vision, with light post-processing of GPT output."""
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
                                "text": (
                                    "Extract the REV value from this engineering drawing. "
                                    "Focus on the TITLE BLOCK in the bottom-right. "
                                    "Prioritise hyphenated numeric REVs (like 2-0, 3-0) and "
                                    "avoid grid letters or revision history tables. "
                                    "Return ONLY the JSON object described in the instructions."
                                )
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
            
            result_text = response.choices[0].message.content or ""
            LOG.debug(f"GPT response received: {result_text[:200]}...")
            
            # Parse JSON (handle ```json``` fences)
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', result_text, re.DOTALL | re.IGNORECASE)
            if json_match:
                result_text = json_match.group(1)
            elif '```' in result_text:
                result_text = re.sub(r'```.*?```', '', result_text, flags=re.DOTALL)
            
            result_data = json.loads(result_text.strip())

            # ---------------------------
            # POST-PROCESSING / FIXUP
            # ---------------------------
            raw_value = result_data.get("rev_value", "")
            notes = (result_data.get("notes") or "").strip()
            confidence = (result_data.get("confidence") or "unknown").lower()

            # Normalise GPT's raw value (2.0 -> 2-0, etc.)
            value = _normalize_rev_value(raw_value)

            # Normalize NO_REV -> EMPTY
            if value == "NO_REV":
                value = "EMPTY"

            # If GPT said NO_REV/EMPTY but its notes clearly say there's a single
            # numeric REV in the TITLE BLOCK, override NO_REV with that numeric value.
            if value in {"NO_REV", "EMPTY"} and notes:
                notes_lower = notes.lower()
                numeric_in_notes = set(REV_NUMERIC_PATTERN.findall(notes))

                # Only override when:
                #   - exactly one numeric candidate is mentioned
                #   - notes explicitly mention the title block
                #   - notes do NOT say the field is blank, no revision, or not applicable
                if len(numeric_in_notes) == 1 and "title block" in notes_lower:
                    bad_phrases = re.compile(
                        r"\b(no\s+rev|no\s+revision|field\s+blank|no\s+value|not\s+applicable)\b",
                        re.IGNORECASE,
                    )
                    if not bad_phrases.search(notes):
                        candidate = next(iter(numeric_in_notes))
                        value = candidate
                        if confidence in {"unknown", "low"}:
                            confidence = "medium"
                        # annotate that we overrode NO_REV based on GPT's own notes
                        notes = (
                            notes + " | "
                            if notes
                            else ""
                        ) + f"Overrode {raw_value or 'NO_REV'} to {candidate} based on notes"

            # Build final result object
            return RevResult(
                file=pdf_path.name,
                value=value,
                engine="gpt_vision",
                confidence=confidence or "unknown",
                notes=notes,
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

# ---------------------------------------------------------------------
# HYBRID PIPELINE
# ---------------------------------------------------------------------
def run_hybrid_pipeline(
    input_folder: Path,
    output_csv: Path,
    azure_endpoint: str,
    azure_key: str,
    deployment_name: str = "gpt-4.1"
) -> List[Dict[str, Any]]:
    """Hybrid pipeline: native first, then GPT+validation fallback."""
    rows: List[Dict[str, Any]] = []
    
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
            # Step 1: Try PyMuPDF native
            result = extract_native_pymupdf(pdf_path)
            
            # Normalise / canonicalise native value
            if result:
                result.value = canonicalize_rev_value(result.value)

            # If native produced a plausible value (including NO_REV / '-' / '_' etc.), accept.
            # If it produced a suspicious value (e.g., 8-32, 5-40, single numbers), trigger GPT fallback.
            if result and result.value and result.value != "":
                if not is_suspicious_rev_value(result.value) and is_plausible_rev_value(result.value):
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
                else:
                    LOG.info(f"Native suspicious for {pdf_path.name}: {result.value} → trying GPT/OCR fallback")
            # Step 2: Fall back to GPT + validation
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

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args(argv=None):
    a = argparse.ArgumentParser(description="PyMuPDF + GPT hybrid (optimized)")
    a.add_argument("input_folder", type=Path)
    a.add_argument("-o", "--output", type=Path, default=Path("rev_results_date.csv"))
    a.add_argument("--azure-endpoint", type=str,
                   default=os.getenv("AZURE_OPENAI_ENDPOINT"),
                   help="Azure OpenAI endpoint URL")
    a.add_argument("--azure-key", type=str,
                   default=os.getenv("AZURE_OPENAI_KEY"),
                   help="Azure OpenAI API key")
    a.add_argument("--deployment-name", type=str,
                   default="gpt-4.1",
                   help="Azure OpenAI deployment name")
    return a.parse_args(argv)

def main(argv=None):
    start_time = time.time() # <-- Start time for performance measurement
    args = parse_args(argv)
    
    if not args.azure_endpoint or not args.azure_key:
        LOG.error("❌ Azure credentials required!")
        LOG.error("Set environment variables or use flags:")
        LOG.error('  $env:AZURE_OPENAI_ENDPOINT = "https://..."')
        LOG.error('  $env:AZURE_OPENAI_KEY = "your_key"')
        return []
    
    results = run_hybrid_pipeline(
        args.input_folder,
        args.output,
        args.azure_endpoint,
        args.azure_key,
        args.deployment_name
    )

    end_time = time.time() # <-- End time for performance measurement
    total_seconds = end_time - start_time

    # Pretty formatting
    mins = int(total_seconds // 60)
    secs = int(total_seconds % 60)

    LOG.info(f"Script completed in {mins}m {secs}s ({total_seconds:.2f} seconds total)")

    return results

if __name__ == "__main__":
    main()
