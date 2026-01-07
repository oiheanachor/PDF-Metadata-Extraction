
# rev_extractor_updated_v5.py
# Patched: fixes argparse errors and enforces native-first, GPT-conditional logic

import argparse
import os
import re

VALID_PLACEHOLDER_REVS = {"-", "_", ".-", "._", "__", "___", "____", "OF"}
VALID_DOUBLE_LETTER_PREFIX = {"A", "B", "C"}
VALID_NUMERIC_SUFFIX = "0"

def is_placeholder_rev(val):
    return val in VALID_PLACEHOLDER_REVS

def is_valid_numeric_rev(val):
    if "-" not in val:
        return False
    return val.split("-")[-1].endswith(VALID_NUMERIC_SUFFIX)

def is_valid_double_letter(val):
    return len(val) == 2 and val[0] in VALID_DOUBLE_LETTER_PREFIX and val.isalpha()

def needs_gpt(val):
    if not val:
        return True
    if is_placeholder_rev(val):
        return False
    if val.isdigit():
        return True
    if "-" in val and not is_valid_numeric_rev(val):
        return True
    if len(val) == 2 and val.isalpha() and not is_valid_double_letter(val):
        return True
    return False

def extract_with_pymupdf(pdf):
    return None

def extract_with_gpt(pdf):
    return None

def extract_rev(pdf, enable_gpt=True):
    native = extract_with_pymupdf(pdf)
    if native and not needs_gpt(native):
        return native, "pymupdf"

    if enable_gpt:
        gpt = extract_with_gpt(pdf)
        if gpt:
            if native == gpt:
                return gpt, "validated"
            if not needs_gpt(gpt):
                return gpt, "gpt"

    return native or "", "needs_review"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--enable-gpt", action="store_true", default=True)
    parser.add_argument("--azure-deployment", default=None)

    args = parser.parse_args()

    if os.path.isdir(args.input):
        files = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.lower().endswith(".pdf")]
    else:
        files = [args.input]

    for f in files:
        rev, engine = extract_rev(f, args.enable_gpt)
        print(f"{os.path.basename(f)},{rev},{engine}")

if __name__ == "__main__":
    main()
