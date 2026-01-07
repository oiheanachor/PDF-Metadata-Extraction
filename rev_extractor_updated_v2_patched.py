#!/usr/bin/env python3
"""
REV Extractor (Native-first + GPT fallback)

- Primary extractor: PyMuPDF "native" text extraction from the title block corners (uses rev_extractor_fixed_v2.process_pdf_native)
- Fallback/2nd-source: GPT-4.1 (Azure OpenAI Vision) ONLY when:
    a) native returns no result, OR
    b) native returns a value that fails plausibility rules / needs validation (edge cases), OR
    c) optional: validate empty/OF results

Adds:
- confidence scoring (0..1 + label)
- human-review flag + reason
- per-engine usage metrics (native vs gpt, validation vs fallback) printed at end + optional JSON

This file intentionally replaces the incomplete v2 wrapper and is runnable as-is.

Usage (examples):
  python rev_extractor_updated_v2.py --input "C:\\pdfs" --output rev_results.csv

  # Native only (no GPT at all)
  python rev_extractor_updated_v2.py --input "C:\\pdfs" --output rev_results.csv --disable-gpt

  # GPT fallback + validate suspicious native outputs
  python rev_extractor_updated_v2.py --input "C:\\pdfs" --output rev_results.csv \
      --azure-endpoint https://<your-resource>.openai.azure.com \
      --azure-deployment gpt-4.1 \
      --azure-api-key %AZURE_OPENAI_API_KEY%

Notes:
- If you don't pass --azure-api-key, the script will look for AZURE_OPENAI_API_KEY env var.
- GPT is used sparingly by design to control cost.
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

# Native extraction helpers
try:
    from rev_extractor_fixed_v2 import process_pdf_native, RevHit
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Could not import rev_extractor_fixed_v2. Ensure rev_extractor_fixed_v2.py is in the same folder or on PYTHONPATH."
    ) from e


# -----------------------------
# Configuration / validation
# -----------------------------

EMPTY_MARKERS = {
    "", "-", "–", "—", "_", "__", "___", "____", "_____", "______",
    ".-", "._", ".__", ".___", ".—", ".–",
}
# normalize any run of underscores to single '_' and any run of hyphens to '-'
_RE_UNDERS = re.compile(r"^_+$")
_RE_DOT_UNDERS = re.compile(r"^\._+$")
_RE_DOT_HYPHEN = re.compile(r"^\.-+$")
_RE_HYPHENS = re.compile(r"^[\-–—]+$")

# Acceptable rev patterns (broad). We apply stricter plausibility rules after.
REV_PATTERNS = [
    re.compile(r"^[A-Z]{1,2}$"),              # A, B, OF, AB
    re.compile(r"^\.[A-Z]$"),                 # .A
    re.compile(r"^\d{1,3}-\d{1,3}$"),          # 1-0, 18-8, 5-40
    re.compile(r"^\d+$"),                     # rare: "1", "202" (flag usually)
]

DOUBLE_LETTER_FIRST_OK = set("ABC")  # if first letter >= D -> suspicious per user note


def normalize_rev(v: Optional[str]) -> str:
    if v is None:
        return ""
    v = str(v).strip()
    # common OCR artifacts
    v = v.replace("O-", "0-").replace("o-", "0-")
    # collapse underscore runs
    if _RE_UNDERS.match(v):
        return "_"
    if _RE_DOT_UNDERS.match(v):
        return "._"
    if _RE_DOT_HYPHEN.match(v):
        return ".-"
    if _RE_HYPHENS.match(v):
        return "-"
    return v


def is_empty_like(v: str) -> bool:
    v = normalize_rev(v)
    return v in EMPTY_MARKERS


def matches_any_pattern(v: str) -> bool:
    v = normalize_rev(v)
    if is_empty_like(v):
        return True
    return any(p.match(v) for p in REV_PATTERNS)


@dataclass
class FinalResult:
    file: str
    value: str
    engine: str            # "native" | "gpt" | "native+gpt"
    confidence: float      # 0..1
    confidence_label: str  # high/medium/low
    human_review: bool
    review_reason: str
    notes: str


def confidence_label(score: float) -> str:
    if score >= 0.80:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def plausibility_checks(v: str) -> List[str]:
    """
    Return list of issues (empty list means plausible).
    These are *heuristics* to reduce overfitting/hallucinations.
    """
    v0 = normalize_rev(v)

    issues: List[str] = []

    # "OF" is treated as empty in your domain
    if v0 == "OF":
        issues.append("OF_treated_as_empty")

    # single/multi-digit numeric (no hyphen) is highly unlikely
    if re.fullmatch(r"\d+", v0 or ""):
        issues.append("bare_number_unlikely")

    # numeric with hyphen: prefer ending with -0
    m = re.fullmatch(r"(\d{1,3})-(\d{1,3})", v0 or "")
    if m:
        if m.group(2) != "0":
            issues.append("numeric_suffix_nonzero")

    # double letters: first letter usually A/B/C
    if re.fullmatch(r"[A-Z]{2}", v0 or ""):
        if v0[0] not in DOUBLE_LETTER_FIRST_OK:
            issues.append("double_letter_first_out_of_scope")

    # invalid token examples seen
    if v0 in {"LTR", "RTL"}:
        issues.append("rotation_token_invalid")

    # anything not matching allowed patterns is suspicious
    if not matches_any_pattern(v0):
        issues.append("pattern_mismatch")

    return issues


# -----------------------------
# GPT (Azure OpenAI) helper
# -----------------------------

def _get_azure_client(api_key: str, endpoint: str):
    """
    Lazy import so native-only runs don't require openai package.
    """
    try:
        from openai import AzureOpenAI  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "openai package not available. Install with: pip install openai>=1.0.0"
        ) from e
    return AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version="2024-02-15-preview")


def _render_crops(pdf_path: Path, dpi: int, crops: List[Tuple[str, Tuple[float, float, float, float]]]) -> List[Tuple[str, bytes]]:
    """
    Render cropped PNGs (bytes) from page 0 using PyMuPDF.
    crops: list of (name, (x0,y0,x1,y1)) in *relative* [0..1] coords.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    page = doc.load_page(0)

    # render whole page
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    # convert to PIL image for cropping
    from PIL import Image
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    out: List[Tuple[str, bytes]] = []
    W, H = img.size
    for name, (x0r, y0r, x1r, y1r) in crops:
        x0 = int(max(0, min(W, x0r * W)))
        y0 = int(max(0, min(H, y0r * H)))
        x1 = int(max(0, min(W, x1r * W)))
        y1 = int(max(0, min(H, y1r * H)))
        if x1 <= x0 or y1 <= y0:
            continue
        crop_img = img.crop((x0, y0, x1, y1))
        import io
        buf = io.BytesIO()
        crop_img.save(buf, format="PNG")
        out.append((name, buf.getvalue()))
    return out


def gpt_extract_rev_from_pdf(
    pdf_path: Path,
    *,
    azure_api_key: str,
    azure_endpoint: str,
    azure_deployment: str,
    dpi: int,
    max_retries: int = 2,
    retry_sleep: float = 0.6,
) -> Tuple[str, float, str]:
    """
    Returns (value, confidence, notes).
    Confidence is model-provided (0..1) but we still validate/normalize afterwards.
    """
    client = _get_azure_client(azure_api_key, azure_endpoint)

    # Try multiple corners for rotated/layout variants.
    # These are relative crops; tuned to title block areas.
    crops = [
        ("bottom_right", (0.62, 0.72, 0.995, 0.995)),
        ("bottom_left",  (0.00, 0.72, 0.38, 0.995)),
        ("top_right",    (0.62, 0.00, 0.995, 0.28)),
        ("top_left",     (0.00, 0.00, 0.38, 0.28)),
    ]
    images = _render_crops(pdf_path, dpi=dpi, crops=crops)

    system = (
        "You extract drawing metadata from engineering PDF title blocks. "
        "Return ONLY JSON. Do not guess. If the REV field is blank/empty, return an empty marker exactly as seen "
        "(e.g. '-', '_', '._', '.-', or ''). If the field is genuinely empty with no visible mark, return 'NO_REV'. "
        "Valid REV examples: A, B, OF (but treat as empty), AB, .A, 1-0, 2-0, 18-8, 5-40, '-', '_', '._', '.-'."
    )

    user = (
        "Task: Find the REV value in the title block. "
        "Rules:\n"
        "- Do NOT read revision history tables; only the title block REV field.\n"
        "- If REV box is empty, return NO_REV (unless you can see a mark like '-', '_', '._', '.-' etc).\n"
        "- Return JSON with keys: value (string), confidence (0..1), evidence (short).\n"
    )

    # We'll ask the model to decide across multiple crops; provide all images.
    content = [{"type": "text", "text": user}]
    for name, png_bytes in images:
        b64 = base64.b64encode(png_bytes).decode("ascii")
        content.append({"type": "text", "text": f"Crop: {name}"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=azure_deployment,
                temperature=0,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": content},
                ],
                response_format={"type": "json_object"},
            )
            txt = resp.choices[0].message.content or "{}"
            data = json.loads(txt)
            val = normalize_rev(str(data.get("value", "")).strip())
            conf = float(data.get("confidence", 0.5) or 0.5)
            evidence = str(data.get("evidence", "")).strip()
            # normalize NO_REV -> empty string sentinel in downstream logic
            if val == "NO_REV":
                val = ""
            return val, max(0.0, min(1.0, conf)), f"gpt_evidence={evidence}"
        except Exception as e:  # pragma: no cover
            last_err = e
            if attempt < max_retries:
                time.sleep(retry_sleep * (attempt + 1))
            else:
                break
    raise RuntimeError(f"GPT extraction failed for {pdf_path.name}: {last_err}")


# -----------------------------
# Orchestration + metrics
# -----------------------------

@dataclass
class Metrics:
    total: int = 0
    native_attempts: int = 0
    native_success: int = 0
    gpt_calls: int = 0
    gpt_used: int = 0
    gpt_validations: int = 0
    human_review: int = 0

    engine_counts: Dict[str, int] = None  # type: ignore

    def __post_init__(self):
        if self.engine_counts is None:
            self.engine_counts = {"native": 0, "gpt": 0, "native+gpt": 0}

    def bump_engine(self, engine: str):
        self.engine_counts[engine] = self.engine_counts.get(engine, 0) + 1


def choose_best(native: Optional[RevHit], gpt_val: Optional[str], gpt_conf: Optional[float]) -> Tuple[str, str, float, str, bool, str]:
    """
    Decide final value and review flags.
    Returns: value, engine, confidence, notes, human_review, review_reason
    """
    notes_parts: List[str] = []
    human_review = False
    review_reason = ""

    native_val = normalize_rev(native.value) if native else ""
    native_score = float(native.score) if native else 0.0
    native_notes = (native.notes or "") if native else ""

    if native:
        notes_parts.append(f"native_score={native_score:.3f}")
        if native_notes:
            notes_parts.append(f"native_notes={native_notes}")

    # Apply 'OF' -> empty treatment at decision time (domain rule)
    if native_val == "OF":
        native_val = ""
        notes_parts.append("native_OF_treated_as_empty")

    native_issues = plausibility_checks(native_val) if native else ["no_native"]
    gpt_val_n = normalize_rev(gpt_val) if gpt_val is not None else None
    gpt_issues = plausibility_checks(gpt_val_n) if gpt_val_n is not None else []

    # If we don't have GPT result, rely on native (even if empty) but flag if suspicious
    if gpt_val_n is None:
        final = native_val
        engine = "native" if native else "native"
        conf = max(0.0, min(1.0, native_score))
        if (not native) or (native_issues and native_val not in ("", "_", "-", "._", ".-")):
            # empty values are okay; but other issues should be reviewed
            if native and native_val != "" and native_issues:
                human_review = True
                review_reason = "native_suspicious:" + ",".join(native_issues)
        return final, engine, conf, "; ".join(notes_parts), human_review, review_reason

    # If both provided: compare & pick plausible
    # Prefer a value with fewer issues; if tie, prefer native unless GPT clearly higher confidence.
    notes_parts.append(f"gpt_conf={gpt_conf if gpt_conf is not None else 0.0:.2f}")
    if gpt_val_n == "OF":
        gpt_val_n = ""
        notes_parts.append("gpt_OF_treated_as_empty")

    # If native missing, use GPT
    if not native:
        final = gpt_val_n
        engine = "gpt"
        conf = float(gpt_conf or 0.6)
        # review if GPT empty/low confidence or has issues beyond allowed empties
        if conf < 0.55:
            human_review = True
            review_reason = "gpt_low_confidence"
        if gpt_issues and not is_empty_like(gpt_val_n):
            human_review = True
            review_reason = (review_reason + ";" if review_reason else "") + "gpt_suspicious:" + ",".join(gpt_issues)
        return final, engine, conf, "; ".join(notes_parts), human_review, review_reason

    # both present
    native_issue_count = len([x for x in native_issues if x not in {"OF_treated_as_empty"}])
    gpt_issue_count = len([x for x in gpt_issues if x not in {"OF_treated_as_empty"}])

    # If native is empty-like and GPT found a non-empty plausible value -> take GPT but mark as validated
    if is_empty_like(native_val) and (not is_empty_like(gpt_val_n)) and gpt_issue_count == 0:
        return gpt_val_n, "native+gpt", float(gpt_conf or 0.6), "; ".join(notes_parts + ["native_empty_gpt_nonempty"]), False, ""

    # If native non-empty but suspicious and GPT returns plausible -> take GPT
    if (not is_empty_like(native_val)) and native_issue_count > 0 and gpt_issue_count == 0:
        return gpt_val_n, "native+gpt", float(gpt_conf or 0.6), "; ".join(notes_parts + ["native_suspicious_gpt_plausible"]), False, ""

    # If GPT is suspicious but native is plausible -> keep native
    if native_issue_count == 0 and gpt_issue_count > 0:
        return native_val, "native+gpt", max(native_score, float(gpt_conf or 0.5) * 0.7), "; ".join(notes_parts + ["gpt_suspicious_keep_native"]), True, "gpt_disagrees_or_suspicious"

    # If they agree -> great
    if native_val == gpt_val_n:
        conf = max(native_score, float(gpt_conf or 0.6))
        return native_val, "native+gpt", conf, "; ".join(notes_parts + ["native_gpt_agree"]), False, ""

    # Otherwise: pick the one with fewer issues; if tie, pick higher confidence; if still tie, flag review.
    if gpt_issue_count < native_issue_count:
        chosen = ("gpt", gpt_val_n, float(gpt_conf or 0.6))
    elif native_issue_count < gpt_issue_count:
        chosen = ("native", native_val, native_score)
    else:
        # tie
        if float(gpt_conf or 0.0) > native_score + 0.10:
            chosen = ("gpt", gpt_val_n, float(gpt_conf or 0.6))
        else:
            chosen = ("native", native_val, native_score)

    engine = "native+gpt"
    final = chosen[1]
    conf = chosen[2]
    human_review = True
    review_reason = f"native_gpt_disagree(native={native_val},gpt={gpt_val_n})"
    return final, engine, conf, "; ".join(notes_parts), human_review, review_reason


def should_call_gpt(native: Optional[RevHit], *, validate_empty: bool, validate_of: bool) -> Tuple[bool, str]:
    if native is None:
        return True, "no_native"
    v = normalize_rev(native.value)
    if v == "OF" and validate_of:
        return True, "validate_OF"
    if is_empty_like(v) and validate_empty:
        return True, "validate_empty"
    issues = plausibility_checks(v)
    # trigger GPT for suspicious values (excluding empty-like)
    if issues and not is_empty_like(v):
        return True, "native_suspicious:" + ",".join(issues)
    # If native score is very low, validate
    if float(native.score) < 0.45:
        return True, "native_low_score"
    return False, ""


def scan_pdfs(input_dir: Path) -> List[Path]:
    exts = {".pdf"}
    pdfs = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    pdfs.sort()
    return pdfs


def run_pipeline(
    input_dir: Path,
    output_csv: Path,
    *,
    brx: float,
    bry: float,
    edge_margin: float,
    dpi: int,
    max_workers: int,
    disable_gpt: bool,
    azure_endpoint: str,
    azure_api_key: str,
    azure_deployment: str,
    validate_empty: bool,
    validate_of: bool,
    metrics_json: Optional[Path] = None,
) -> Metrics:
    pdfs = scan_pdfs(input_dir)
    metrics = Metrics(total=len(pdfs))

    # blocklist used in native token scoring (reuse fixed_v2 defaults)
    blocklist = {"DWG", "DWG.", "NO", "NO.", "SCALE", "SHEET", "OF", "SIZE", "TITLE", "REV"}  # safe baseline

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "value", "engine", "confidence", "confidence_label", "human_review", "review_reason", "notes"],
        )
        writer.writeheader()

        for pdf_path in tqdm(pdfs, desc="Scanning PDFs"):
            metrics.native_attempts += 1
            native_hit: Optional[RevHit] = None
            try:
                native_hit = process_pdf_native(pdf_path, brx=brx, bry=bry, blocklist=blocklist, edge_margin=edge_margin)
            except Exception as e:
                # treat as no native hit
                native_hit = None

            if native_hit:
                metrics.native_success += 1

            gpt_val = None
            gpt_conf = None
            gpt_reason = ""
            if not disable_gpt:
                do_gpt, gpt_reason = should_call_gpt(native_hit, validate_empty=validate_empty, validate_of=validate_of)
                if do_gpt:
                    metrics.gpt_calls += 1
                    # native exists => this is validation
                    if native_hit is not None:
                        metrics.gpt_validations += 1
                    try:
                        gpt_val, gpt_conf, gpt_note = gpt_extract_rev_from_pdf(
                            pdf_path,
                            azure_api_key=azure_api_key,
                            azure_endpoint=azure_endpoint,
                            azure_deployment=azure_deployment,
                            dpi=dpi,
                        )
                        metrics.gpt_used += 1
                        gpt_reason = gpt_reason + (("; " + gpt_note) if gpt_note else "")
                    except Exception as e:
                        gpt_reason = gpt_reason + f"; gpt_failed={type(e).__name__}"

            final_value, engine, conf, notes, review, reason = choose_best(native_hit, gpt_val, gpt_conf)
            if gpt_reason:
                notes = (notes + "; " if notes else "") + f"gpt_reason={gpt_reason}"

            # final empty markers
            final_value = normalize_rev(final_value)
            if final_value == "":
                # represent truly empty as NO_REV in output for clarity
                out_val = "NO_REV"
            elif final_value == "OF":
                out_val = "NO_REV"
            else:
                out_val = final_value

            # If final value still fails patterns (should be rare), force review
            if not matches_any_pattern(final_value) and out_val != "NO_REV":
                review = True
                reason = (reason + ";" if reason else "") + "final_pattern_mismatch"

            if review:
                metrics.human_review += 1

            metrics.bump_engine(engine)

            writer.writerow(
                {
                    "file": pdf_path.name,
                    "value": out_val,
                    "engine": engine,
                    "confidence": f"{conf:.3f}",
                    "confidence_label": confidence_label(conf),
                    "human_review": str(bool(review)),
                    "review_reason": reason,
                    "notes": notes,
                }
            )

    # Print summary for stakeholders
    print("\n=== REV Extraction Metrics ===")
    print(f"Total PDFs:                {metrics.total}")
    print(f"Native attempts:           {metrics.native_attempts}")
    print(f"Native success (any hit):  {metrics.native_success}")
    print(f"GPT calls (attempted):     {metrics.gpt_calls}")
    print(f"GPT used (succeeded):      {metrics.gpt_used}")
    print(f"GPT validations:           {metrics.gpt_validations}")
    print(f"Human review flagged:      {metrics.human_review}")
    print("Engine usage breakdown:")
    for k, v in sorted(metrics.engine_counts.items(), key=lambda kv: kv[0]):
        print(f"  - {k:10s}: {v}")

    if metrics_json:
        metrics_json.parent.mkdir(parents=True, exist_ok=True)
        with metrics_json.open("w", encoding="utf-8") as mf:
            json.dump(
                {
                    **asdict(metrics),
                    "engine_counts": metrics.engine_counts,
                    "notes": {
                        "gpt_calls_vs_total_pct": (metrics.gpt_calls / metrics.total * 100.0) if metrics.total else 0.0,
                        "gpt_used_vs_total_pct": (metrics.gpt_used / metrics.total * 100.0) if metrics.total else 0.0,
                    },
                },
                mf,
                indent=2,
            )
        print(f"Metrics JSON written to: {metrics_json}")

    return metrics


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract REV values from PDFs (native-first, GPT fallback).")
    p.add_argument("--input", required=True, help="Input folder containing PDFs (scans recursively).")
    p.add_argument("--output", required=True, help="Output CSV path.")
    p.add_argument("--brx", type=float, default=0.97, help="Native search: bottom-right X (0..1) of title block window.")
    p.add_argument("--bry", type=float, default=0.92, help="Native search: bottom-right Y (0..1) of title block window.")
    p.add_argument("--edge-margin", type=float, default=0.02, help="Native search: extra margin around window.")
    p.add_argument("--dpi", type=int, default=200, help="DPI for GPT vision rendering (only used if GPT enabled).")
    p.add_argument("--max-workers", type=int, default=1, help="Reserved (future parallelism). Keep 1 for stability.")
    p.add_argument("--disable-gpt", action="store_true", help="Disable GPT entirely (native-only).")
    p.add_argument("--validate-empty", action="store_true", help="Use GPT to validate empty native results.")
    p.add_argument("--validate-of", action="store_true", help="Use GPT to validate native 'OF' results (treated as empty).")
    p.add_argument("--metrics-json", default="", help="Optional metrics JSON output path.")
    # Azure OpenAI
    p.add_argument("--azure-endpoint", default=os.getenv("AZURE_OPENAI_ENDPOINT", ""), help="Azure OpenAI endpoint.")
    p.add_argument("--azure-api-key", default=os.getenv("AZURE_OPENAI_API_KEY", ""), help="Azure OpenAI API key.")
    p.add_argument("--azure-deployment", default=os.getenv("AZURE_OPENAI_DEPLOYMENT", ""), help="Azure model deployment name.")
    args = p.parse_args(argv)

    # basic validation
    in_dir = Path(args.input)
    if not in_dir.exists():
        raise SystemExit(f"INPUT dir does not exist: {in_dir}")
    if not args.disable_gpt:
        if not args.azure_endpoint or not args.azure_api_key or not args.azure_deployment:
            raise SystemExit(
                "GPT enabled but Azure settings missing. Provide --azure-endpoint, --azure-api-key, --azure-deployment "
                "or set AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY / AZURE_OPENAI_DEPLOYMENT env vars."
            )
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    metrics_json = Path(args.metrics_json) if args.metrics_json else None

    run_pipeline(
        Path(args.input),
        Path(args.output),
        brx=args.brx,
        bry=args.bry,
        edge_margin=args.edge_margin,
        dpi=args.dpi,
        max_workers=int(args.max_workers),
        disable_gpt=bool(args.disable_gpt),
        azure_endpoint=args.azure_endpoint,
        azure_api_key=args.azure_api_key,
        azure_deployment=args.azure_deployment,
        validate_empty=bool(args.validate_empty),
        validate_of=bool(args.validate_of),
        metrics_json=metrics_json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
