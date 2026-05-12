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
from typing import Any, Dict, List, Optional, Sequence, Tuple
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
GPT_SYSTEM_PROMPT = """You are an expert at analyzing engineering drawings and extracting current revision / issue values.

YOUR TASK:
Extract the current REV / ISSUE value from this engineering drawing.

CRITICAL RULES:

1. Valid Locations:
   - Primary: title block fields labelled REV, REV., REVISION, ISS, ISS., ISSUE, ISSUE NO, or ISSUE NUMBER.
   - Valid title block regions include bottom-right and top-left. Rotated drawings may place these visually in another crop.
   - Secondary: revision / issue tables, especially when a title block value is blank, missing, or stale.
   - In revision tables, newest chronological entries are usually above older entries. Use the latest row when it conflicts with a stale title block.
   - In title blocks with stacked values, the newer/current issue is often lower than the older/original issue.

2. Avoid These Common Mistakes:
   - DO NOT extract grid reference letters (A, B, C along edges)
   - DO NOT extract part numbers or item callouts
   - DO NOT extract section markers (e.g., "SECTION C-C")
   - DO NOT extract view indicators
   - DO NOT extract dates, drawing numbers, material grades, dimensions, sheet counts, scale values, or names/initials.

3. Valid REV / ISSUE Value Formats:
   - Single numbers are valid and common: 0, 1, 2, 3, 5, 10, 16, 01, 08.
   - Hyphenated numbers are valid: 0-0, 1-0, 1-1, 2-1, 8-0, 12-01.
   - Single letters are valid and common: A, B, C, D.
   - Double letters can be valid only when clearly field-associated.
   - Blank, dash, underscore, or "OF" in a revision field means EMPTY.
   - No credible revision / issue field or table means NO_REV.

4. Validation Checklist:
   - Is there a REV / ISSUE label nearby?
   - If it is from a table, is it the latest chronological entry rather than an old history row?
   - Is it near title block anchors such as DWG NO, SHEET, SCALE, DRAWN, CHECKED, APPROVED, TITLE, or Drawing Number?
   - Is it isolated (not part of another number)?
   - Does it follow a valid revision / issue format?
   - If multiple candidates exist, explain why the returned value is current.

RESPONSE FORMAT:
Return ONLY a JSON object with this exact shape:
{
  "rev_value": "1",
  "confidence": "high",
  "source": "revision_table",
  "location": "bottom-left issue table, latest row above original issue",
  "evidence": ["ISSUE table header", "rows show A then 1 with 1 as latest"],
  "notes": "Title block is blank/stale; table chronology indicates current issue 1"
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


VALID_CONFIDENCE = {"high", "medium", "low", "none", "unknown"}

# ---------------------------------------------------------------------
# NATIVE EXTRACTION (YOUR EXISTING WORKING CODE)
# ---------------------------------------------------------------------
def extract_native_pymupdf(pdf_path: Path, enable_ocr: bool = False) -> Optional[RevResult]:
    """Try native PyMuPDF extraction using the fixed logic from rev_extractor_fixed."""
    try:
        from rev_extractor_fixed_v2 import (process_pdf_native, _normalize_output_value,
            DEFAULT_BR_X, DEFAULT_BR_Y, DEFAULT_EDGE_MARGIN, DEFAULT_REV_2L_BLOCKLIST, canonicalize_rev_value, is_plausible_rev_value, is_suspicious_rev_value)
        
        best = process_pdf_native(
            pdf_path,
            brx=DEFAULT_BR_X,
            bry=DEFAULT_BR_Y,
            blocklist=DEFAULT_REV_2L_BLOCKLIST,
            edge_margin=DEFAULT_EDGE_MARGIN,
            enable_ocr=enable_ocr,
        )
        
        if best and best.value:
            value = _normalize_output_value(best.value)
            if "revision_table" in best.engine:
                confidence = "high"
            elif best.score >= 70:
                confidence = "medium"
            else:
                confidence = "low"
            return RevResult(
                file=pdf_path.name,
                value=value,
                engine=f"pymupdf_{best.engine}",
                confidence=confidence,
                notes=best.context_snippet
            )
        return None
    except Exception as e:
        LOG.warning(f"Native extraction failed for {pdf_path.name}: {e}")
        return None

# ---------------------------------------------------------------------
# GPT NORMALIZATION HELPERS
# ---------------------------------------------------------------------

# Patterns used for lightweight post-processing. GPT remains the authority in
# vision mode; deterministic text correction is intentionally conservative.
REV_NUMERIC_PATTERN = re.compile(r"\b\d{1,3}-\d{1,3}\b")

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

    # Single numeric issues are common on the new site.
    if re.fullmatch(r"\d{1,3}", s_up):
        return s_up

    if s_up in {"NO_REV", "OF"}:
        return "NO_REV"

    return s_up

def _parse_json_object(text: str) -> Dict[str, Any]:
    """Parse a model JSON object robustly from plain or fenced output."""
    raw = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL | re.IGNORECASE)
    if fence:
        raw = fence.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start:end + 1])
        raise

# ---------------------------------------------------------------------
# GPT EXTRACTOR
# ---------------------------------------------------------------------
class AzureGPTExtractor:
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        deployment_name: str = "gpt-4.1",
        api_version: str = "2024-12-01-preview",
    ):
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
                api_version=api_version,
                azure_endpoint=endpoint
            )
            self.deployment_name = deployment_name
            LOG.info("✓ GPT client initialized successfully")
        except Exception as e:
            LOG.error(f"Failed to initialize GPT client: {e}")
            raise
    
    def _render_clip_to_base64(
        self,
        page: fitz.Page,
        clip: Optional[fitz.Rect],
        dpi: int,
    ) -> str:
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False, clip=clip)
        return base64.b64encode(pix.tobytes("png")).decode("utf-8")

    def pdf_to_base64_images(self, pdf_path: Path, page_idx: int = 0) -> List[Tuple[str, str]]:
        """
        Render a compact multi-view packet for GPT.

        Full page gives orientation/layout context. Focused crops keep title
        blocks and revision tables legible without paying for high-DPI full-page
        images on every call.
        """
        with fitz.open(pdf_path) as doc:
            page = doc[page_idx]
            rect = page.rect
            crops = [
                ("full_page_context", None, 110),
                ("top_left_title_or_issue_block", fitz.Rect(rect.x0, rect.y0, rect.x0 + rect.width * 0.45, rect.y0 + rect.height * 0.35), 220),
                ("bottom_left_revision_table", fitz.Rect(rect.x0, rect.y0 + rect.height * 0.55, rect.x0 + rect.width * 0.45, rect.y1), 240),
                ("bottom_right_title_block", fitz.Rect(rect.x0 + rect.width * 0.52, rect.y0 + rect.height * 0.55, rect.x1, rect.y1), 240),
                ("top_right_title_or_revision_block", fitz.Rect(rect.x0 + rect.width * 0.52, rect.y0, rect.x1, rect.y0 + rect.height * 0.35), 220),
            ]
            return [
                (label, self._render_clip_to_base64(page, clip, dpi))
                for label, clip, dpi in crops
            ]

    def extract_rev(self, pdf_path: Path) -> RevResult:
        """Extract REV using GPT-4 Vision, with light post-processing of GPT output."""
        try:
            LOG.debug(f"Converting {pdf_path.name} to GPT image packet...")
            images = self.pdf_to_base64_images(pdf_path, page_idx=0)
            content = [
                {
                    "type": "text",
                    "text": (
                        "Extract the current REV / ISSUE value. You are given one full-page context image "
                        "and focused crops. Inspect all images before answering. Prefer deterministic visual "
                        "evidence: marker labels, title-block fields, and latest revision-table rows. "
                        "Return ONLY the JSON object described in the system instructions."
                    ),
                }
            ]
            for label, img_base64 in images:
                content.append({"type": "text", "text": f"Image: {label}"})
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                })
            
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
                        "content": content,
                    }
                ],
                max_tokens=700,
                temperature=0
            )
            
            result_text = response.choices[0].message.content or ""
            LOG.debug(f"GPT response received: {result_text[:200]}...")
            
            result_data = _parse_json_object(result_text)

            # ---------------------------
            # POST-PROCESSING / FIXUP
            # ---------------------------
            raw_value = result_data.get("rev_value", "")
            notes = (result_data.get("notes") or "").strip()
            confidence = (result_data.get("confidence") or "unknown").lower()
            if confidence not in VALID_CONFIDENCE:
                confidence = "unknown"
            evidence = result_data.get("evidence") or []
            source = result_data.get("source") or ""
            location = result_data.get("location") or ""

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

            if isinstance(evidence, list) and evidence:
                evidence_text = "; evidence=" + " | ".join(str(x) for x in evidence[:4])
            else:
                evidence_text = ""
            if source or location:
                prefix = f"source={source}; location={location}".strip("; ")
                notes = f"{prefix}; {notes}".strip("; ")
            notes = (notes + evidence_text).strip()

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
    deployment_name: str = "gpt-4.1",
    api_version: str = "2024-12-01-preview",
    workflow: str = "pymupdf-gpt",
    enable_native_ocr: bool = False,
    fallback_below: str = "high",
) -> List[Dict[str, Any]]:
    """Run revision extraction.

    workflow:
      - pymupdf-gpt: PyMuPDF first, Azure GPT fallback when native confidence is below threshold.
      - gpt-only: Azure GPT vision for every file.
      - pymupdf-only: deterministic native path only.
    """
    rows: List[Dict[str, Any]] = []

    confidence_rank = {"none": 0, "low": 1, "unknown": 1, "medium": 2, "high": 3}
    fallback_rank = confidence_rank.get(fallback_below, 3)
    use_gpt = workflow in {"pymupdf-gpt", "gpt-only"}
    gpt = None
    if use_gpt:
        LOG.info("Initializing Azure GPT client...")
        gpt = AzureGPTExtractor(azure_endpoint, azure_key, deployment_name, api_version=api_version)
    
    pdfs = list(input_folder.glob("*.pdf"))
    if not pdfs:
        LOG.warning(f"No PDFs found in {input_folder}")
        return rows
    
    native_attempted = 0
    native_success = 0
    native_no_result = 0
    native_fallback = 0
    gpt_used = 0
    gpt_failed = 0
    
    for pdf_path in tqdm(pdfs, desc="Processing PDFs"):
        try:
            result = None

            if workflow in {"pymupdf-gpt", "pymupdf-only"}:
                native_attempted += 1
                result = extract_native_pymupdf(pdf_path, enable_ocr=enable_native_ocr)

            if workflow == "pymupdf-only":
                if result:
                    native_success += 1
                    rows.append({
                        "file": result.file,
                        "value": result.value,
                        "engine": result.engine,
                        "confidence": result.confidence,
                        "notes": result.notes[:240],
                    })
                else:
                    rows.append({
                        "file": pdf_path.name,
                        "value": "",
                        "engine": "pymupdf_no_result",
                        "confidence": "none",
                        "notes": "",
                    })
                continue

            if workflow == "pymupdf-gpt" and result and result.value:
                native_level = confidence_rank.get(result.confidence, 1)
                if native_level >= fallback_rank:
                    native_success += 1
                    rows.append({
                        "file": result.file,
                        "value": result.value,
                        "engine": result.engine,
                        "confidence": result.confidence,
                        "notes": result.notes[:240],
                    })
                    LOG.info(f"✓ PyMuPDF accepted for {pdf_path.name}: {result.value} ({result.confidence})")
                    continue
                native_fallback += 1
                LOG.info(
                    f"Native confidence {result.confidence} for {pdf_path.name}: "
                    f"{result.value or '<blank>'} → Azure GPT fallback"
                )
            elif workflow == "pymupdf-gpt":
                native_no_result += 1
                LOG.info(f"PyMuPDF found no result for {pdf_path.name} → Azure GPT fallback")

            LOG.info(f"→ Using Azure GPT for {pdf_path.name}")
            gpt_used += 1
            result = gpt.extract_rev(pdf_path) if gpt else RevResult(pdf_path.name, "", "gpt_unavailable", "none", "")
            
            if result.engine == "gpt_failed":
                gpt_failed += 1
            
            rows.append({
                "file": result.file,
                "value": result.value,
                "engine": result.engine,
                "confidence": result.confidence,
                "notes": result.notes[:240]
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
        LOG.info(
            f"Stats: NativeAttempted={native_attempted}, NativeAccepted={native_success}, "
            f"NativeNoResult={native_no_result}, NativeFallback={native_fallback}, "
            f"GPT={gpt_used}, Failed={gpt_failed}"
        )
        if gpt_used > 0:
            LOG.info(f"Cost≈${(gpt_used - gpt_failed) * 0.010:.2f}")
    except Exception as e:
        LOG.error(f"Failed to write CSV: {e}")
    
    return rows

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args(argv=None):
    a = argparse.ArgumentParser(description="Revision / issue extraction with PyMuPDF and Azure GPT workflows")
    a.add_argument("input_folder", type=Path)
    a.add_argument("-o", "--output", type=Path, default=Path("rev_results_date.csv"))
    a.add_argument(
        "--workflow",
        choices=["pymupdf-gpt", "gpt-only", "pymupdf-only"],
        default="pymupdf-gpt",
        help="pymupdf-gpt is the default: deterministic extraction first, Azure GPT fallback.",
    )
    a.add_argument(
        "--fallback-below",
        choices=["high", "medium", "low"],
        default="high",
        help="In pymupdf-gpt mode, use Azure GPT when PyMuPDF confidence is below this level.",
    )
    a.add_argument(
        "--native-ocr",
        action="store_true",
        help="Allow the PyMuPDF/native stage to use local Tesseract OCR. Disabled by default.",
    )
    a.add_argument("--azure-endpoint", type=str,
                   default=os.getenv("AZURE_OPENAI_ENDPOINT"),
                   help="Azure OpenAI endpoint URL")
    a.add_argument("--azure-key", type=str,
                   default=os.getenv("AZURE_OPENAI_KEY"),
                   help="Azure OpenAI API key")
    a.add_argument("--deployment-name", type=str,
                   default="gpt-4.1",
                   help="Azure OpenAI deployment name")
    a.add_argument("--azure-api-version", type=str,
                   default=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
                   help="Azure OpenAI API version")
    return a.parse_args(argv)

def main(argv=None):
    start_time = time.time() # <-- Start time for performance measurement
    args = parse_args(argv)
    
    needs_gpt = args.workflow in {"pymupdf-gpt", "gpt-only"}
    if needs_gpt and (not args.azure_endpoint or not args.azure_key):
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
        args.deployment_name,
        args.azure_api_version,
        workflow=args.workflow,
        enable_native_ocr=args.native_ocr,
        fallback_below=args.fallback_below,
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
