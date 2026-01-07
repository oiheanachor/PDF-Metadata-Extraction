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
        from rev_extractor_fixed_v3 import (process_pdf_native, _normalize_output_value,
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


def _validate_gpt_result(
    result: "RevResult",
    page_text: str = "",
) -> "RevResult":
    """
    Post-validate GPT output.
    If page_text is provided, we can sanity check existence; regardless we enforce
    the same plausibility rules as native.
    """
    from rev_extractor_fixed_v3 import canonicalize_rev_value, is_plausible_rev_value, is_empty_rev_value

    value = canonicalize_rev_value(result.value)
    notes = (result.notes or "").strip()
    confidence = result.confidence

    # Empty markers are allowed (NO_REV, OF->NO_REV, '-', '_', etc)
    if is_empty_rev_value(value):
        return RevResult(
            file=result.file,
            value=value,
            engine=result.engine,
            confidence=confidence,
            notes=notes,
        )

    # Never accept out-of-scope values as "high"
    if not is_plausible_rev_value(value):
        if notes:
            notes += " | "
        notes += f"GPT returned out-of-scope value {value}; flagged for review"
        return RevResult(
            file=result.file,
            value=value,
            engine=result.engine,
            confidence="low",
            notes=notes,
        )

    # Optional existence check (best-effort)
    if page_text:
        if value not in page_text and value.lstrip(".") not in page_text:
            if notes:
                notes += " | "
            notes += "Value not found in extracted page text; confidence downgraded"
            confidence = "medium" if confidence == "high" else "low"

    return RevResult(
        file=result.file,
        value=value,
        engine=result.engine,
        confidence=confidence,
        notes=notes,
    )
def run_hybrid_pipeline(
    input_folder: str,
    output_csv: str,
    azure_endpoint: str,
    azure_key: str,
    deployment_name: str,
    *,
    dpi: int = 300,
    max_workers: int = 2,
    validate_empty_with_gpt: bool = False,
    gpt_on_suspicious_native: bool = True,
    gpt_on_no_native: bool = True,
) -> List["RevResult"]:
    """
    Native-first + GPT fallback (cost-aware):
      - Always try PyMuPDF native first (preferred).
      - Only call GPT when native returns no result OR native result is suspicious/out-of-scope,
        or optionally when native returns empty/OF and you want validation.
    """
    from rev_extractor_fixed_v3 import (
        canonicalize_rev_value,
        is_empty_rev_value,
        is_suspicious_rev_value,
        is_plausible_rev_value,
    )

    input_folder = Path(input_folder)
    pdf_files = sorted([p for p in input_folder.rglob("*.pdf")])
    LOG.info(f"Scanning PDFs: {len(pdf_files)}")

    gpt = AzureGPTExtractor(
        endpoint=azure_endpoint,
        api_key=azure_key,
        deployment=deployment_name,
    )

    results: List[RevResult] = []

    for pdf_path in tqdm(pdf_files, desc="Scanning PDFs"):
        file_name = pdf_path.name

        # 1) Native extraction (preferred)
        native_res = extract_native_pymupdf(str(pdf_path), dpi=dpi)

        n_val = canonicalize_rev_value(native_res.value) if native_res else ""
        n_is_empty = is_empty_rev_value(n_val) if native_res else False
        n_is_suspicious = (
            is_suspicious_rev_value(n_val) or (n_val and not is_plausible_rev_value(n_val))
        ) if native_res else True

        need_gpt = False
        reason = ""
        if native_res is None and gpt_on_no_native:
            need_gpt = True
            reason = "native_no_result"
        elif native_res and n_is_empty and validate_empty_with_gpt:
            need_gpt = True
            reason = "native_empty_validate"
        elif native_res and n_is_suspicious and gpt_on_suspicious_native:
            need_gpt = True
            reason = "native_suspicious"

        # 2) GPT fallback if needed
        if need_gpt:
            gpt_res = gpt.extract_rev(str(pdf_path), dpi=dpi)
            gpt_res = _validate_gpt_result(gpt_res)

            # Reconcile native vs GPT per your rules
            if native_res is None:
                gpt_res.notes = (gpt_res.notes or "") + (f" | reason={reason}" if reason else "")
                results.append(gpt_res)
                continue

            g_val = canonicalize_rev_value(gpt_res.value)
            g_is_empty = is_empty_rev_value(g_val)
            g_is_plausible = is_plausible_rev_value(g_val) or g_is_empty
            n_is_plausible = is_plausible_rev_value(n_val) or n_is_empty

            final = None
            if n_is_plausible and not g_is_plausible:
                final = native_res
                final.confidence = "medium" if n_is_suspicious else final.confidence
                final.notes = (final.notes or "") + f" | GPT out-of-scope ({g_val}); using native; reason={reason}"
            elif g_is_plausible and not n_is_plausible:
                final = gpt_res
                final.confidence = "high" if is_plausible_rev_value(g_val) else "medium"
                final.notes = (final.notes or "") + f" | Native out-of-scope ({n_val}); using GPT; reason={reason}"
            elif n_is_plausible and g_is_plausible:
                if n_val == g_val:
                    final = native_res
                    final.confidence = "high"
                    final.notes = (final.notes or "") + f" | GPT matched native; reason={reason}"
                else:
                    # Disagreement -> flag review, prefer non-empty plausible
                    final = gpt_res if (is_plausible_rev_value(g_val) and not g_is_empty) else native_res
                    final.confidence = "medium"
                    final.notes = (final.notes or "") + f" | Native/GPT disagree (native={n_val}, gpt={g_val}); flagged for review; reason={reason}"
            else:
                final = gpt_res if gpt_res else native_res
                final.confidence = "low"
                final.notes = (final.notes or "") + f" | Both native and GPT out-of-scope (native={n_val}, gpt={g_val}); flagged for review; reason={reason}"

            results.append(final)
            continue

        # 3) No GPT needed -> accept native
        if native_res:
            results.append(native_res)
        else:
            results.append(
                RevResult(
                    file=file_name,
                    value="NO_REV",
                    engine="pymupdf_native",
                    confidence="low",
                    notes="No native result",
                )
            )

    write_results_csv(results, output_csv)
    return results
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
        args.azure_deployment,
        dpi=args.dpi,
        max_workers=args.max_workers,
        validate_empty_with_gpt=args.validate_empty_with_gpt,
        gpt_on_suspicious_native=not args.no_gpt_on_suspicious_native,
        gpt_on_no_native=not args.no_gpt_on_no_native,
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
