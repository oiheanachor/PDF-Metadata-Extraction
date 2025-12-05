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
# IMPROVED SYSTEM PROMPT - strongly discourage guessing
GPT_SYSTEM_PROMPT = """You are an expert at analyzing engineering drawings and extracting revision (REV) information.

YOUR TASK:
Extract the REV (revision) value from the TITLE BLOCK of this engineering drawing.

CRITICAL RULES

1. TITLE BLOCK ONLY
- The REV value is in the TITLE BLOCK, usually in the BOTTOM-RIGHT corner.
- The title block typically contains fields like: DWG NO, SHEET, SCALE, DRAWN BY, CHECKED BY, APPROVED BY, DATE, etc.
- It often includes the company name or logo (e.g. ROTORK, FAIRCHILD).

2. NEVER GUESS
- You must ONLY copy a REV value that is clearly visible in the image.
- If the REV text is blurred, cut off, hidden, or you are not 100% sure what it is:
  → return rev_value = "NO_REV".
- Do NOT infer a value from patterns or expectations.
- If the REV field appears blank or you cannot confidently read any value in the REV field:
  → return rev_value = "NO_REV".

3. VALID REV FORMATS (in order of priority)

NUMERIC REVISIONS (very common)
- Hyphenated numbers: 1-0, 2-0, 3-0, 12-01, 15-02, etc.
- These are MAJOR-MINOR versions. Example: 2-0 ≈ "2.0".
- You MUST copy the digits exactly as shown (2-0 is NOT the same as 3-0).

LETTER REVISIONS
- Single letter: A, B, C, D, …, Z
- Double letters: AA, AB, AC, etc.

SPECIAL CASES
- If the REV field shows "OF" or something clearly meaning "not applicable":
  → rev_value = "EMPTY"
- If there is no REV field at all, OR the field is clearly empty:
  → rev_value = "NO_REV"

4. THINGS TO IGNORE (COMMON MISTAKES)
DO NOT return any of the following as the REV value:
- Entries in REVISION HISTORY TABLES (often top-right, columns like: REV | ECO | DATE | DESCRIPTION).
- Grid letters around the border (A, B, C, 1, 2, 3).
- Section markers (e.g. "SECTION C-C", "DETAIL A").
- Part numbers, item callouts, or drawing numbers.
- Any text that is not in or immediately next to the REV field in the TITLE BLOCK.

5. VALIDATION CHECKLIST BEFORE ANSWERING
Before you answer, confirm ALL of these:
- The value is in or right next to the REV field in the title block.
- There is a "REV" or "REV:" label nearby.
- The value matches a valid format (letter or hyphenated number).
- You are confident you have read it correctly.

If ANY of these checks fail, respond with:
- rev_value = "NO_REV"
- confidence = "low"
- notes = "REV field blank, missing, or not readable with certainty"

RESPONSE FORMAT (IMPORTANT)
Return ONLY a JSON object, no extra text:

{
  "rev_value": "<REV_VALUE_OR_NO_REV_OR_EMPTY>",
  "confidence": "<high|medium|low>",
  "location": "short description of where you found (or did not find) the REV",
  "notes": "very short justification; mention if field was blank, not present, or unreadable"
}

EXAMPLES

Example 1 – Numeric REV in title block
- Bottom-right title block shows: "REV: 2-0" next to DWG NO and SHEET fields.

Correct:
{
  "rev_value": "2-0",
  "confidence": "high",
  "location": "bottom-right title block",
  "notes": "Hyphenated numeric REV 2-0 in title block, ignoring revision history table"
}

Example 2 – Letter REV
- Title block shows "REV: E" clearly labeled in the REV field.

Correct:
{
  "rev_value": "E",
  "confidence": "high",
  "location": "bottom-right title block",
  "notes": "Letter REV E in title block"
}

Example 3 – NO REV
- Title block has no REV field at all, or the REV field looks blank.

Correct:
{
  "rev_value": "NO_REV",
  "confidence": "high",
  "location": "title block",
  "notes": "No REV field or value present in the title block"
}

Example 4 – EMPTY / OF
- Title block shows "REV: OF" or a code clearly meaning "not applicable".

Correct:
{
  "rev_value": "EMPTY",
  "confidence": "high",
  "location": "bottom-right title block",
  "notes": "REV field marked as not applicable (OF)"
}
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
            _detect_numeric_rev, _detect_letter_rev
        )
    except ImportError:
        LOG.warning("rev_extractor_fixed.py not available, skipping native extraction.")
        return None
    
    try:
        value, info = process_pdf_native(pdf_path)
        if not value:
            return None
        
        norm = _normalize_output_value(value)
        confidence = "high" if info.get("source") == "title_block" else "medium"
        
        notes_parts = []
        if info.get("source"):
            notes_parts.append(f"source={info['source']}")
        if info.get("reason"):
            notes_parts.append(info["reason"])
        
        return RevResult(
            file=pdf_path.name,
            value=norm,
            engine="native_pymupdf",
            confidence=confidence,
            notes="; ".join(notes_parts)
        )
    except Exception as e:
        LOG.error(f"Native PyMuPDF extraction failed for {pdf_path.name}: {e}")
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

    Behaviours:
    - If GPT predicts a numeric hyphenated REV (e.g. 2-0, 3-0), we ensure it exists in text.
      * If not, but there is exactly one numeric candidate, we correct to that.
      * If not and there are 0 or >1 candidates, we force NO_REV (low confidence).
    - NEW: If GPT predicts NO_REV/EMPTY, we try to salvage a single numeric REV found near a 'REV'
      label in the text.
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
    # 1) GPT said NO_REV / EMPTY
    # ------------------------------
    if value in {"NO_REV", "EMPTY"}:
        # Try to salvage: if there's exactly ONE numeric REV near a REV label,
        # we assume GPT was too conservative and use that.
        if page_text and len(numeric_candidates) == 1:
            candidate = next(iter(numeric_candidates))
            near_pattern = re.compile(
                rf"REV[^A-Z0-9]{{0,12}}{re.escape(candidate)}\b",
                re.IGNORECASE,
            )
            if near_pattern.search(page_text):
                # Override GPT's NO_REV
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
        if not page_text:
            if confidence == "high":
                confidence = "medium"
        else:
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
            # Supported by text – good
            if confidence in {"unknown", ""}:
                confidence = "high"
        elif len(numeric_candidates) == 1:
            # Snap to the single real candidate
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
    # 3) Letter or other value
    # ------------------------------
    if value and page_text:
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
            LOG.info("✓ GPT client initialized successfully")
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

    def extract_rev(self, pdf_path: Path) -> "RevResult":
        """Extract REV using Azure GPT-4 vision with strong post-validation."""
        try:
            LOG.debug(f"Converting {pdf_path.name} to image for GPT...")
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
                                    "Look at this engineering drawing and extract the REV value "
                                    "from the TITLE BLOCK only. Do NOT use revision history tables "
                                    "or grid letters. If the REV field is missing, blank, or you are "
                                    "not completely sure what it says, respond with rev_value='NO_REV'. "
                                    "Return ONLY the JSON object as described in the instructions."
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
            LOG.debug(f"Raw GPT response for {pdf_path.name}: {result_text[:200]!r}")

            # Strip code fences if present
            import json
            json_match = re.search(r"```json\s*(\{.*?\})\s*```", result_text, re.DOTALL | re.IGNORECASE)
            if json_match:
                result_text = json_match.group(1)
            elif "```" in result_text:
                result_text = re.sub(r"```.*?```", "", result_text, flags=re.DOTALL)

            result_data = json.loads(result_text.strip())

            # Post-process and validate against actual PDF text
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
    
    # Validate Azure config
    if not azure_endpoint or not azure_key:
        LOG.error("Azure endpoint or key not provided. Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY.")
        return rows
    
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
            else:
                # Step 2: Fallback to GPT
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
                   help="Azure OpenAI key")
    a.add_argument("--deployment-name", type=str, default="gpt-4.1",
                   help="Azure OpenAI deployment name (default: gpt-4.1)")
    return a.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    
    if not args.azure_endpoint or not args.azure_key:
        LOG.error("Azure configuration missing. Set environment variables or pass flags:")
        LOG.error("  export AZURE_OPENAI_ENDPOINT='https://...'")
        LOG.error("  export AZURE_OPENAI_KEY='your_key'")
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
