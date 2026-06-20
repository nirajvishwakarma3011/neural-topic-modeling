"""
Merge Mistral-generated extended texts into tweet CSV.
Only uses extended_text for rare-class docs (cluster in RARE_CLUSTERS).
All other docs keep original text as extended_text.

Output: data/tweet_10_labels_extended.csv
  Same columns as tweet_10_labels.csv + extended_text column
"""
import json
import pandas as pd
from pathlib import Path
from src.preprocess import _clean_text

RARE_CLUSTERS = {20, 55, 60, 71, 75, 79, 87, 88, 99, 104}
JSONL_PATH = Path("LLM/extended/tweet_10_mistral.jsonl")
INPUT_CSV  = Path("data/tweet_10_labels.csv")
OUTPUT_CSV = Path("data/tweet_10_labels_extended_all.csv")


def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded CSV: {len(df)} rows")

    # Build idx→extended map from JSONL
    extended_map = {}
    with open(JSONL_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            extended_map[rec["idx"]] = rec["extended"]
    print(f"Loaded JSONL: {len(extended_map)} records")

    # Compute CSV-row → doc-idx mapping (mirrors loader: skip empty cleaned text)
    doc_idx = 0
    csv_to_doc_idx = {}
    for csv_row, row in df.iterrows():
        cleaned = _clean_text(row["text"])
        if cleaned:
            csv_to_doc_idx[csv_row] = doc_idx
            doc_idx += 1

    print(f"Non-empty docs after cleaning: {doc_idx}")

    # Build extended_text column
    extended_texts = []
    used_extended = 0
    missing = 0
    for csv_row, row in df.iterrows():
        cluster = int(float(row["cluster"]))
        doc_idx_val = csv_to_doc_idx.get(csv_row)
        if cluster in RARE_CLUSTERS and doc_idx_val is not None and doc_idx_val in extended_map:
            extended_texts.append(extended_map[doc_idx_val])
            used_extended += 1
        else:
            if cluster in RARE_CLUSTERS and doc_idx_val not in extended_map:
                missing += 1
            extended_texts.append(row["text"])

    df["extended_text"] = extended_texts
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Rare-class docs with extended text: {used_extended}")
    print(f"Rare-class docs missing from JSONL: {missing}")
    print(f"Saved → {OUTPUT_CSV}")

    # Verify
    rare_mask = df["cluster"].astype(float).isin(RARE_CLUSTERS)
    same = (df.loc[rare_mask, "text"] == df.loc[rare_mask, "extended_text"]).sum()
    print(f"Rare-class rows where extended==original (should be {missing}): {same}")


if __name__ == "__main__":
    main()
