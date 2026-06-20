"""
jsonl_to_csv.py
───────────────
Convert the LLM-generated extended texts JSONL file to a CSV ready for
use with any topic model or downstream classifier.

Output CSV columns
──────────────────
  text      — LLM-extended long text  (reconstruction target for PVTM)
  label     — integer cluster / class label
  original  — original short text     (kept for reference / alignment check)
  idx       — original document index (keeps alignment with main.py ordering)

Usage
─────
# StackOverflow extended texts
python jsonl_to_csv.py \
    --input  /data4/home/nirajv/small_text/LLM/extended/stackoverflow_mistral.jsonl \
    --output /data4/home/nirajv/small_text/LLM/extended/stackoverflow_mistral.csv

# GoogleNews (if you generate it later)
python jsnol_csv.py \
    --input  /data4/home/nirajv/small_text/LLM/extended/googlenewst_mistral.jsonl \
    --output /data4/home/nirajv/small_text/LLM/extended/googlenewst_mistral.csv

# Only keep text + label (minimal CSV for direct model input)
python jsonl_to_csv.py \
    --input  /data4/home/nirajv/small_text/LLM/extended/stackoverflow_mistral.jsonl \
    --output /data4/home/nirajv/small_text/LLM/extended/stackoverflow_mistral.csv \
    --minimal
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def convert(input_path: Path, output_path: Path, minimal: bool = False) -> None:
    """
    Read JSONL → write CSV.

    JSONL line format:
        {"idx": int, "label": int|null, "original": str, "extended": str}

    CSV columns (full):
        idx, label, original, text

    CSV columns (--minimal):
        text, label
    """
    # ── Read all records ──────────────────────────────────────────────────────
    records = []
    n_no_label   = 0
    n_bad_lines  = 0

    print(f"[read]  {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_bad_lines += 1
                print(f"  [WARN] line {lineno}: JSON decode error — skipped")
                continue

            # Validate required fields
            if "extended" not in rec:
                n_bad_lines += 1
                print(f"  [WARN] line {lineno}: missing 'extended' field — skipped")
                continue

            if rec.get("label") is None:
                n_no_label += 1

            records.append({
                "idx":      rec.get("idx",      lineno - 1),
                "label":    rec.get("label",    ""),
                "original": rec.get("original", ""),
                "text":     rec["extended"],
            })

    if not records:
        print("[error] No valid records found. Is the JSONL path correct?")
        sys.exit(1)

    # ── Sort by idx so row order matches docs[] in main.py ───────────────────
    records.sort(key=lambda r: r["idx"])

    # ── Write CSV ─────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if minimal:
        fieldnames = ["text", "label"]
    else:
        fieldnames = ["idx", "label", "original", "text"]

    print(f"[write] {output_path}  ({len(records)} rows, columns: {fieldnames})")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames  = fieldnames,
            extrasaction = "ignore",   # drop fields not in fieldnames
            quoting     = csv.QUOTE_ALL,  # quote all fields — extended texts
                                          # contain commas, newlines, quotes
        )
        writer.writeheader()
        writer.writerows(records)

    # ── Summary ───────────────────────────────────────────────────────────────
    labels = [r["label"] for r in records if r["label"] != ""]
    print(f"\n[summary]")
    print(f"  Total rows        : {len(records)}")
    print(f"  Unique labels     : {len(set(labels)) if labels else 'N/A (no labels)'}")
    print(f"  Rows without label: {n_no_label}")
    print(f"  Skipped bad lines : {n_bad_lines}")

    if labels:
        from collections import Counter
        counts = Counter(labels)
        print(f"  Label distribution:")
        for lbl, cnt in sorted(counts.items()):
            bar = "█" * (cnt * 30 // max(counts.values()))
            print(f"    label {lbl:>3}:  {cnt:>5}  {bar}")

    avg_len = sum(len(r["text"].split()) for r in records) / len(records)
    print(f"  Avg extended text length: {avg_len:.0f} words")
    print(f"\n[done]  Saved → {output_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Convert PVTM extended texts JSONL → CSV"
    )
    ap.add_argument(
        "--input", required=True,
        help="Path to .jsonl file (e.g. LLM/extended/stackoverflow_mistral.jsonl)"
    )
    ap.add_argument(
        "--output", required=True,
        help="Path to output .csv file"
    )
    ap.add_argument(
        "--minimal", action="store_true",
        help="Write only 'text' and 'label' columns (skip idx and original)"
    )
    args = ap.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"[error] Input file not found: {input_path}")
        sys.exit(1)

    convert(input_path, output_path, minimal=args.minimal)


if __name__ == "__main__":
    main()