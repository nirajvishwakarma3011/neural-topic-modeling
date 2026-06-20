"""
Expert-Label Analysis for MoE-NTM runs.

For each run with gate_weights.npy:
  1. Hard assignment: group docs by argmax expert, show label distribution
  2. Soft affinity: gate^T @ labels / gate_mass → weighted label affinity per expert
  3. Sample docs: top-3 docs per expert (highest gate weight)

Outputs (written into each run's directory):
  expert_hard_dist.csv     — label proportions per expert (hard routing)
  expert_soft_affinity.csv — weighted label affinity per expert (soft gating)
  expert_doc_samples.csv   — top doc per expert with labels
  expert_summary.txt       — human-readable summary

Usage:
  python analyse_expert_labels.py                      # all runs
  python analyse_expert_labels.py --run <run_dir>      # single run
  python analyse_expert_labels.py --dataset reuters    # filter by dataset
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Dataset config ────────────────────────────────────────────────────────────

DATASETS = {
    "reuters_10": {
        "csv": "data/reuters_10.csv",
        "text_col": "text",
        "label_start": 2,
    },
    "mpst": {
        "csv": "data/mpst_8_labels_final.csv",
        "text_col": "plot",
        "label_start": 2,
    },
}

RESULTS_DATASET_MAP = {
    "reuters_10": "reuters_10",
    "mpst": "mpst",
}


def detect_dataset(run_dir: Path) -> str | None:
    for key in DATASETS:
        if key in run_dir.name:
            return key
    return None


def load_dataset(dataset_key: str, base_dir: Path) -> pd.DataFrame:
    cfg = DATASETS[dataset_key]
    path = base_dir / cfg["csv"]
    return pd.read_csv(path)


def get_label_cols(df: pd.DataFrame, dataset_key: str) -> list[str]:
    start = DATASETS[dataset_key]["label_start"]
    return df.columns[start:].tolist()


def analyse_run(run_dir: Path, base_dir: Path) -> bool:
    artifacts = run_dir / "artifacts"
    gate_path = artifacts / "gate_weights.npy"
    doc_topic_path = artifacts / "doc_topic.npy"

    if not gate_path.exists():
        print(f"  [skip] no gate_weights.npy: {run_dir.name}")
        return False

    dataset_key = detect_dataset(run_dir)
    if dataset_key is None:
        print(f"  [skip] unknown dataset: {run_dir.name}")
        return False

    print(f"\n{'='*70}")
    print(f"Run: {run_dir.name}")
    print(f"Dataset: {dataset_key}")

    # Load
    gate = np.load(gate_path)          # [N, E]
    doc_topic = np.load(doc_topic_path) if doc_topic_path.exists() else None
    df = load_dataset(dataset_key, base_dir)
    label_cols = get_label_cols(df, dataset_key)
    text_col = DATASETS[dataset_key]["text_col"]

    N, E = gate.shape
    L = len(label_cols)

    if len(df) != N:
        print(f"  [warn] doc count mismatch: gate={N}, csv={len(df)}. Truncating to min.")
        n = min(N, len(df))
        gate = gate[:n]
        df = df.iloc[:n].reset_index(drop=True)
        N = n

    labels = df[label_cols].values.astype(float)  # [N, L]

    # ── Hard assignment ───────────────────────────────────────────────────────
    expert_assign = gate.argmax(axis=1)  # [N]

    hard_rows = []
    for e in range(E):
        mask = expert_assign == e
        n_e = mask.sum()
        if n_e == 0:
            row = {"expert": e, "n_docs": 0, "pct_corpus": 0.0}
            for lc in label_cols:
                row[lc] = float("nan")
            row["dominant_label"] = "—"
            row["top2_label"] = "—"
            row["label_entropy"] = float("nan")
        else:
            label_dist = labels[mask].mean(axis=0)  # [L]
            sorted_idx = label_dist.argsort()[::-1]
            dominant = label_cols[sorted_idx[0]]
            top2 = label_cols[sorted_idx[1]] if L > 1 else "—"
            p = label_dist / (label_dist.sum() + 1e-12)
            entropy = float(-np.sum(p * np.log(p + 1e-12)))

            row = {
                "expert": e,
                "n_docs": int(n_e),
                "pct_corpus": round(100 * n_e / N, 2),
            }
            for i, lc in enumerate(label_cols):
                row[lc] = round(float(label_dist[i]), 4)
            row["dominant_label"] = dominant
            row["top2_label"] = top2
            row["label_entropy"] = round(entropy, 4)
        hard_rows.append(row)

    hard_df = pd.DataFrame(hard_rows)

    # ── Soft affinity ─────────────────────────────────────────────────────────
    gate_mass = gate.sum(axis=0) + 1e-12   # [E]
    soft_affinity = (gate.T @ labels) / gate_mass[:, None]  # [E, L]

    soft_rows = []
    for e in range(E):
        row = {"expert": e, "gate_mass": round(float(gate_mass[e]), 4)}
        for i, lc in enumerate(label_cols):
            row[lc] = round(float(soft_affinity[e, i]), 4)
        sorted_idx = soft_affinity[e].argsort()[::-1]
        row["dominant_label"] = label_cols[sorted_idx[0]]
        row["top2_label"] = label_cols[sorted_idx[1]] if L > 1 else "—"
        soft_rows.append(row)

    soft_df = pd.DataFrame(soft_rows)

    # ── Mean gating entropy per expert ───────────────────────────────────────
    gate_entropy = -(gate * np.log(gate + 1e-12)).sum(axis=1).mean()

    # ── Expert utilization ───────────────────────────────────────────────────
    utilization = [(expert_assign == e).mean() for e in range(E)]

    # ── Sample docs: top-3 per expert (highest gate[:,e]) ────────────────────
    sample_rows = []
    for e in range(E):
        top_idx = gate[:, e].argsort()[::-1][:3]
        for rank, idx in enumerate(top_idx):
            text = str(df.iloc[idx][text_col])
            text_snippet = text.replace("\n", " ").strip()[:200]
            row_labels = {lc: int(df.iloc[idx][lc]) for lc in label_cols}
            active_labels = [lc for lc, v in row_labels.items() if v == 1]
            sample_rows.append({
                "expert": e,
                "rank_in_expert": rank + 1,
                "doc_idx": int(idx),
                "gate_weight": round(float(gate[idx, e]), 4),
                "active_labels": "|".join(active_labels) if active_labels else "none",
                "text_snippet": text_snippet,
            })

    sample_df = pd.DataFrame(sample_rows)

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    hard_df.to_csv(run_dir / "expert_hard_dist.csv", index=False)
    soft_df.to_csv(run_dir / "expert_soft_affinity.csv", index=False)
    sample_df.to_csv(run_dir / "expert_doc_samples.csv", index=False)

    # ── Human-readable summary ────────────────────────────────────────────────
    lines = []
    lines.append(f"Expert-Label Analysis: {run_dir.name}")
    lines.append(f"Dataset: {dataset_key}  |  N={N}  |  E={E}  |  Labels={L}")
    lines.append(f"Mean gating entropy: {gate_entropy:.4f}")
    lines.append("")
    lines.append("── Hard Assignment (argmax routing) ─────────────────────")
    lines.append(hard_df[["expert", "n_docs", "pct_corpus", "dominant_label", "top2_label", "label_entropy"]].to_string(index=False))
    lines.append("")
    lines.append("── Label proportions per expert (hard) ──────────────────")
    lines.append(hard_df[["expert"] + label_cols].to_string(index=False))
    lines.append("")
    lines.append("── Soft Affinity (weighted by gate mass) ────────────────")
    lines.append(soft_df[["expert", "gate_mass", "dominant_label", "top2_label"] + label_cols].to_string(index=False))
    lines.append("")
    lines.append("── Expert Utilization (fraction of docs routed here) ────")
    for e, u in enumerate(utilization):
        bar = "█" * int(u * 40)
        lines.append(f"  Expert {e}: {u*100:5.1f}%  {bar}")
    lines.append("")
    lines.append("── Top-3 Doc Samples per Expert ─────────────────────────")
    for e in range(E):
        lines.append(f"\n  Expert {e}:")
        sub = sample_df[sample_df["expert"] == e]
        for _, r in sub.iterrows():
            lines.append(f"    [{r['rank_in_expert']}] gate={r['gate_weight']:.4f}  labels=[{r['active_labels']}]")
            lines.append(f"        {r['text_snippet'][:180]}")

    summary = "\n".join(lines)
    (run_dir / "expert_summary.txt").write_text(summary)

    # Print summary
    print(summary)
    print(f"\n  Saved: expert_hard_dist.csv, expert_soft_affinity.csv, expert_doc_samples.csv, expert_summary.txt")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default=None,
                        help="Single run directory path")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Filter by dataset key (reuters_10, mpst)")
    args = parser.parse_args()

    base_dir = Path(__file__).parent

    if args.run:
        run_dir = Path(args.run)
        analyse_run(run_dir, base_dir)
        return

    # Discover all results dirs with gate_weights
    gate_files = sorted(base_dir.glob("results_*/*/artifacts/gate_weights.npy"))
    run_dirs = [g.parent.parent for g in gate_files]

    if args.dataset:
        run_dirs = [r for r in run_dirs if args.dataset in r.name]

    print(f"Found {len(run_dirs)} runs with gate_weights.npy")
    ok = 0
    for rd in run_dirs:
        if analyse_run(rd, base_dir):
            ok += 1

    print(f"\n{'='*70}")
    print(f"Done. Analysed {ok}/{len(run_dirs)} runs.")


if __name__ == "__main__":
    main()
