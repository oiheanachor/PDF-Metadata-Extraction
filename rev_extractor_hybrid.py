#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REV Extractor — HYBRID (Best of Both Worlds)
Combines: optimized's "2-0" extraction + production's anti-hallucination
"""

from __future__ import annotations
import argparse, base64, csv, logging, os, re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
from tqdm import tqdm

# OpenAI SDK
try:
    from openai import AzureOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

LOG = logging.getLogger("rev_extractor_hybrid")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ----------------------------- HYBRID PROMPT (BEST OF BOTH) -------------------

GPT_SYSTEM_PROMPT = """You are an expert at analyzing engineering drawings and extracting revision information.

**YOUR TASK:**
Extract the REV (revision) value from the title block of this engineering drawing.

⚠️ READ THE EXACT VALUE IN THIS SPECIFIC DRAWING - Every drawing is different!

**TITLE BLOCK:**
Typically bottom-right, contains: DWG NO, SHEET, SCALE, DRAWN BY, company logo
REV field is WITHIN or ADJACENT to this cluster

**AVOID:**
❌ Revision history tables (top-right, columns: REV | DATE | DESCRIPTION)
❌ Grid letters (A, B, C along edges)
❌ Section markers ("SECTION C-C")
❌ Defaulting to "2-0" when uncertain!

**FORMATS (ALL EQUALLY VALID):**
- Numeric: 1-0, 2-0, 3-0, 4-0, 5-1 (each is DIFFERENT!)
- Letters: A, B, C, E, F, AB
- No field: NO_REV

**EXTRACTION STEPS:**
1. Find title block (DWG NO, SHEET, SCALE)
2. Locate REV field
3. Read EXACT value
4. Verify not from history table/grid/section marker
5. Output EXACT value (don't substitute!)

**VALIDATION:**
✓ Title block found?
✓ REV field exists?
✓ What's the EXACT value THIS drawing shows?
✓ Not from history table?
✓ Not grid letter/section marker?
✓ Am I reading actual value (not defaulting to "2-0")?

**FORMAT:**
{"rev_value": "3-0", "confidence": "high", "notes": "Clear REV 3-0 next to DWG NO"}

**EXAMPLES:**

Ex1: Numeric "2-0"
Title: "DWG NO: 21620 | REV: 2-0"
✅ {"rev_value": "2-0"}
Notes: "Clear 2-0 in title block, distinct from drawing number"
❌ {"rev_value": "A"} ← Don't pick nearby letters!

Ex2: Numeric "3-0" (DIFFERENT from 2-0!)
Title: "DWG NO: 22416 | REV: 3-0"
✅ {"rev_value": "3-0"}
Notes: "Clear 3-0 next to DWG NO"
❌ {"rev_value": "2-0"} ← Read actual, don't default!

Ex3: Numeric "4-0"
Title: "DWG NO: 54321 | REV: 4-0"
✅ {"rev_value": "4-0"}
❌ {"rev_value": "2-0"} ← Stop defaulting!

Ex4: Letter "A"
Title: "DWG NO: 21837 | REV: A"
✅ {"rev_value": "A"}

Ex5: Letter "C"
Title: "DWG NO: 18301 | REV: C"
Drawing has "SECTION C-C" elsewhere
✅ {"rev_value": "C"}
Notes: "REV C in title block, distinct from section marker"

Ex6: Letter "E"
Title: "DWG NO: 032-IPI-008 | REV: E"
✅ {"rev_value": "E"}

Ex7: Double "AB"
Title: "DWG NO: 14579 | REV: AB"
✅ {"rev_value": "AB"}

Ex8: NO_REV (no field)
Title has DWG NO, SHEET, SCALE but NO REV field
✅ {"rev_value": "NO_REV"}
Notes: "Title block found but no REV field present"
❌ {"rev_value": "2-0"} ← Don't hallucinate!

Ex9: NO_REV (empty/rotated)
Title block present, REV field empty OR drawing rotated/unclear
✅ {"rev_value": "NO_REV"}
Notes: "REV field empty or drawing unclear"
❌ {"rev_value": "2-0"} ← Don't default when empty!

Ex10: NO_REV (section marker only)
Drawing has "SECTION C-C", letters on edges, but NO REV in title block
✅ {"rev_value": "NO_REV"}
Notes: "No REV field; C is section marker only"
❌ {"rev_value": "C"} ← Don't pick section marker!

**ANTI-HALLUCINATION:**
Before responding, ask:
1. Did I find title block with DWG NO/SHEET?
2. Is there REV field in that title block?
3. What EXACT value do I see in THIS drawing?
4. Am I reading actual value or defaulting to "2-0"?
5. If I see "3-0" but want to output "2-0" → STOP! Output "3-0"!
6. If no REV field → return "NO_REV" (NOT "2-0")!

**KEY POINTS:**
1. Each numeric value is DIFFERENT: 1-0 ≠ 2-0 ≠ 3-0 ≠ 4-0
2. Do NOT default to "2-0"
3. See "2-0" → output "2-0"
4. See "3-0" → output "3-0" (NOT "2-0"!)
5. See no REV → output "NO_REV" (NOT "2-0"!)
6. See "A" → output "A" (NOT "2-0"!)
7. Numeric and letter equally valid

**CONFIDENCE:**
- "high": Clear REV with unambiguous value
- "medium": Near title block, no explicit label
- "low": Uncertain

⚠️ FINAL CHECK:
If outputting "2-0":
- Did I actually SEE "2-0" in THIS drawing?
- Or am I uncertain and defaulting?
- If uncertain → return "NO_REV" (low confidence), NOT "2-0"!"""

# ----------------------------- Data & Functions --------------------------------

@dataclass
class RevResult:
    file: str
    value: str
    engine: str
    confidence: str = "unknown"
    notes: str = ""

def extract_native_pymupdf(pdf_path: Path) -> Optional[RevResult]:
    try:
        from rev_extractor_fixed import (
            process_pdf_native, _normalize_output_value,
            DEFAULT_BR_X, DEFAULT_BR_Y, DEFAULT_EDGE_MARGIN, DEFAULT_REV_2L_BLOCKLIST
        )
        best = process_pdf_native(pdf_path, brx=DEFAULT_BR_X, bry=DEFAULT_BR_Y,
                                  blocklist=DEFAULT_REV_2L_BLOCKLIST, edge_margin=DEFAULT_EDGE_MARGIN)
        if best and best.value:
            value = _normalize_output_value(best.value)
            if value == "EMPTY":
                value = "NO_REV"
            return RevResult(file=pdf_path.name, value=value, engine=f"pymupdf_{best.engine}",
                           confidence="high" if best.score > 100 else "medium",
                           notes=best.context_snippet[:100])
        return None
    except Exception as e:
        LOG.debug(f"Native failed {pdf_path.name}: {e}")
        return None

class AzureGPTExtractor:
    def __init__(self, endpoint: str, api_key: str, deployment_name: str = "gpt-4.1"):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai not installed")
        endpoint = endpoint.rstrip('/').split('/openai/deployments')[0]
        LOG.info(f"GPT-4.1 Hybrid | Endpoint: {endpoint} | Deployment: {deployment_name}")
        self.client = AzureOpenAI(api_key=api_key, api_version="2024-02-15-preview", azure_endpoint=endpoint)
        self.deployment_name = deployment_name
    
    def pdf_to_base64(self, pdf_path: Path, dpi: int = 150) -> str:
        with fitz.open(pdf_path) as doc:
            page = doc[0]
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72.0, dpi/72.0), alpha=False)
            return base64.b64encode(pix.tobytes("png")).decode('utf-8')
    
    def extract_rev(self, pdf_path: Path) -> RevResult:
        try:
            img_b64 = self.pdf_to_base64(pdf_path)
            user_msg = """Extract REV from this drawing.

CRITICAL:
1. Find title block (DWG NO, SHEET, SCALE)
2. Locate REV field
3. Read EXACT value in THIS drawing
4. DO NOT default to "2-0"!
5. Each value different: 1-0, 2-0, 3-0, 4-0 are all different
6. See "3-0" → output "3-0" (NOT "2-0"!)
7. No REV field → output "NO_REV" (NOT "2-0"!)
8. Numeric and letter equally valid

ANTI-HALLUCINATION:
Before outputting "2-0", verify you SEE "2-0" in THIS drawing.
If uncertain, return "NO_REV" instead of guessing."""

            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": GPT_SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_msg},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                    ]}
                ],
                max_tokens=500,
                temperature=0
            )
            
            txt = response.choices[0].message.content
            import json
            m = re.search(r'```json\s*(\{.*?\})\s*```', txt, re.DOTALL)
            if m:
                txt = m.group(1)
            elif '```' in txt:
                txt = re.sub(r'```.*?```', '', txt, flags=re.DOTALL)
            
            data = json.loads(txt.strip())
            val = data.get("rev_value", "")
            if val == "EMPTY":
                val = "NO_REV"
            
            return RevResult(file=pdf_path.name, value=val, engine="gpt_vision",
                           confidence=data.get("confidence", "unknown"),
                           notes=data.get("notes", "")[:200])
        except Exception as e:
            LOG.error(f"GPT failed {pdf_path.name}: {e}")
            return RevResult(file=pdf_path.name, value="", engine="gpt_failed",
                           confidence="none", notes=str(e)[:100])

def run_pipeline(input_folder, output_csv, endpoint, key, deployment):
    rows = []
    gpt = AzureGPTExtractor(endpoint, key, deployment)
    pdfs = sorted(input_folder.glob("*.pdf"))
    if not pdfs:
        LOG.warning(f"No PDFs in {input_folder}")
        return rows
    
    LOG.info(f"Found {len(pdfs)} PDFs")
    native_ok, gpt_used, gpt_fail = 0, 0, 0
    
    for pdf in tqdm(pdfs, desc="Processing"):
        try:
            res = extract_native_pymupdf(pdf)
            if res and res.value and res.value != "NO_REV":
                native_ok += 1
                rows.append({"file": res.file, "value": res.value, "engine": res.engine,
                           "confidence": res.confidence, "notes": res.notes})
                LOG.debug(f"✓ Native: {pdf.name} → {res.value}")
                continue
            
            LOG.info(f"→ GPT: {pdf.name}")
            gpt_used += 1
            res = gpt.extract_rev(pdf)
            if res.engine == "gpt_failed":
                gpt_fail += 1
            rows.append({"file": res.file, "value": res.value, "engine": res.engine,
                       "confidence": res.confidence, "notes": res.notes})
        except Exception as e:
            LOG.error(f"Failed {pdf.name}: {e}")
            rows.append({"file": pdf.name, "value": "", "engine": "error",
                       "confidence": "none", "notes": str(e)[:100]})
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['file', 'value', 'engine', 'confidence', 'notes'])
        w.writeheader()
        w.writerows(rows)
    
    LOG.info(f"\n{'='*60}\nResults: {output_csv.resolve()}\n{'='*60}")
    LOG.info(f"Total: {len(rows)} | Native: {native_ok} | GPT: {gpt_used} | Failed: {gpt_fail}")
    if gpt_used > 0:
        LOG.info(f"Cost: ${(gpt_used - gpt_fail) * 0.010:.2f}")
    LOG.info(f"{'='*60}\n")
    return rows

def parse_args(argv=None):
    a = argparse.ArgumentParser(description="Hybrid REV Extractor")
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
        return []
    return run_pipeline(args.input_folder, args.output, args.azure_endpoint,
                       args.azure_key, args.deployment_name)

if __name__ == "__main__":
    main()
