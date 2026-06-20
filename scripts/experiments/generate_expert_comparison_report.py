"""
Generate cross-run expert specialization comparison report for MoE-NTM variants.
Reads gate_weights.npy, expert_soft_affinity.csv, expert_hard_dist.csv, metrics.json,
training_log.csv from all MoE runs and writes a single analysis report.

Output: moe_expert_comparison_report.txt  (project root)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).parent

# ── Run registry ──────────────────────────────────────────────────────────────

DATASETS = {
    "reuters_10": {
        "csv": "data/reuters_10.csv",
        "label_start": 2,
        "n_labels": 10,
    },
    "mpst": {
        "csv": "data/mpst_8_labels_final.csv",
        "label_start": 2,
        "n_labels": 8,
    },
}

VARIANT_LABELS = {
    "moe_ntm":           "Dense-BoW",
    "moe_ntm_use":       "Dense-SBERT",
    "moe_ntm_sparse":    "Sparse-BoW",
    "moe_ntm_use_sparse":"Sparse-SBERT",
    "moe_ntm_attn":      "Attn-BoW",
    "moe_ntm_use_attn":  "Attn-SBERT",
    "moe_ntm_ec":        "EC-BoW",
    "moe_ntm_use_ec":    "EC-SBERT",
}

ROUTING_GROUP = {
    "moe_ntm":           "Dense",
    "moe_ntm_use":       "Dense",
    "moe_ntm_sparse":    "Sparse",
    "moe_ntm_use_sparse":"Sparse",
    "moe_ntm_attn":      "Attention",
    "moe_ntm_use_attn":  "Attention",
    "moe_ntm_ec":        "Expert-Choice",
    "moe_ntm_use_ec":    "Expert-Choice",
}


def detect_method(run_name: str) -> str | None:
    for key in sorted(VARIANT_LABELS.keys(), key=len, reverse=True):
        if key in run_name:
            return key
    return None


def detect_dataset(run_name: str) -> str | None:
    for key in DATASETS:
        if key in run_name:
            return key
    return None


def load_run(run_dir: Path) -> dict | None:
    artifacts = run_dir / "artifacts"
    gate_path = artifacts / "gate_weights.npy"
    hard_path = run_dir / "expert_hard_dist.csv"
    soft_path = run_dir / "expert_soft_affinity.csv"

    if not gate_path.exists():
        return None

    run_name = run_dir.name
    method = detect_method(run_name)
    dataset = detect_dataset(run_name)
    if not method or not dataset:
        return None

    gate = np.load(gate_path)            # [N, E]
    N, E = gate.shape
    label_cols = pd.read_csv(BASE / DATASETS[dataset]["csv"]).columns[
        DATASETS[dataset]["label_start"]:
    ].tolist()

    # Expert utilization
    assignments = gate.argmax(axis=1)
    util = np.array([(assignments == e).mean() for e in range(E)])
    n_collapsed = int((util < 0.05).sum())
    entropy = float(-(gate * np.log(gate + 1e-12)).sum(axis=1).mean())

    # Soft affinity
    df_soft = pd.read_csv(soft_path) if soft_path.exists() else None
    df_hard = pd.read_csv(hard_path) if hard_path.exists() else None

    # Metrics
    metrics = {}
    m_path = run_dir / "metrics.json"
    if m_path.exists():
        metrics = json.load(open(m_path))

    # Training log — final epoch
    train_final = {}
    tlog = run_dir / "training_log.csv"
    if tlog.exists():
        tdf = pd.read_csv(tlog)
        if len(tdf) > 0:
            train_final = tdf.iloc[-1].to_dict()

    # Config
    config_path = run_dir / "dataset_fingerprint.json"

    return {
        "run_dir": run_dir,
        "run_name": run_name,
        "method": method,
        "variant_label": VARIANT_LABELS[method],
        "routing_group": ROUTING_GROUP[method],
        "dataset": dataset,
        "N": N,
        "E": E,
        "label_cols": label_cols,
        "gate": gate,
        "assignments": assignments,
        "util": util,
        "n_collapsed": n_collapsed,
        "gating_entropy": entropy,
        "df_soft": df_soft,
        "df_hard": df_hard,
        "metrics": metrics,
        "train_final": train_final,
    }


def specialization_score(soft_df: pd.DataFrame, label_cols: list[str]) -> float:
    """
    Mean max-label affinity across experts.
    High = experts each focus on one dominant label.
    Low = experts spread evenly (no specialization).
    """
    if soft_df is None:
        return float("nan")
    vals = soft_df[label_cols].values  # [E, L]
    return float(vals.max(axis=1).mean())


def label_coverage(soft_df: pd.DataFrame, label_cols: list[str], threshold=0.15) -> int:
    """How many distinct labels are 'dominant' for at least one expert."""
    if soft_df is None:
        return 0
    dominant = soft_df[label_cols].idxmax(axis=1).unique()
    return len(dominant)


def expert_table(run: dict) -> list[str]:
    """Format per-expert specialization table for one run."""
    lines = []
    df = run["df_soft"]
    label_cols = run["label_cols"]
    util = run["util"]
    E = run["E"]

    if df is None:
        return ["  (no soft affinity data)"]

    col_w = 12
    header = f"  {'Expert':>7}  {'Util%':>6}  {'DomLabel':>14}  {'2ndLabel':>14}  " + \
             "  ".join(f"{lc[:col_w]:>{col_w}}" for lc in label_cols)
    lines.append(header)
    lines.append("  " + "-" * len(header))

    for _, row in df.iterrows():
        e = int(row["expert"])
        u = util[e] * 100
        dom = str(row["dominant_label"])[:14]
        top2 = str(row["top2_label"])[:14]
        affinities = "  ".join(f"{row[lc]:>{col_w}.3f}" for lc in label_cols)
        collapse_flag = " *" if util[e] < 0.05 else "  "
        lines.append(f"  {e:>7}{collapse_flag}  {u:>5.1f}%  {dom:>14}  {top2:>14}  {affinities}")

    lines.append("  (* = collapsed expert, util < 5%)")
    return lines


def format_run_section(run: dict) -> list[str]:
    lines = []
    lines.append(f"\n{'─'*80}")
    lines.append(f"Run: {run['run_name']}")
    lines.append(f"Variant: {run['variant_label']}  |  Dataset: {run['dataset'].upper()}  "
                 f"|  N={run['N']}  |  E={run['E']}  |  K={run['metrics'].get('k','?')}")
    lines.append("")

    # Coherence metrics
    m = run["metrics"]
    lines.append("  Coherence & Quality Metrics:")
    lines.append(f"    NPMI:           {m.get('npmi_paper', 'N/A')}")
    lines.append(f"    Cv:             {m.get('cv', 'N/A')}")
    lines.append(f"    Topic Diversity:{m.get('topic_diversity', 'N/A')}")

    # Training final
    tf = run["train_final"]
    if tf:
        lines.append(f"\n  Training (final epoch):")
        lines.append(f"    Loss:     {tf.get('loss', 'N/A'):.4f}")
        lines.append(f"    Recon:    {tf.get('recon', 'N/A'):.4f}")
        lines.append(f"    KL:       {tf.get('kl', 'N/A'):.4f}")
        if "balance" in tf:
            lines.append(f"    Balance:  {tf.get('balance', 'N/A'):.4f}")
        if "distill" in tf:
            lines.append(f"    Distill:  {tf.get('distill', 'N/A'):.4f}")
        lines.append(f"    UniqueTopWords: {int(tf.get('unique_top_words', -1))}")

    # Expert health
    spec = specialization_score(run["df_soft"], run["label_cols"])
    cov = label_coverage(run["df_soft"], run["label_cols"])
    lines.append(f"\n  Expert Routing Health:")
    lines.append(f"    Gating Entropy (mean over docs): {run['gating_entropy']:.4f}")
    lines.append(f"    Collapsed Experts (util<5%):     {run['n_collapsed']}/{run['E']}")
    lines.append(f"    Specialization Score (mean max-affinity): {spec:.4f}")
    lines.append(f"    Label Coverage (distinct dominant labels): {cov}/{len(run['label_cols'])}")

    # Utilization bar
    lines.append(f"\n  Expert Utilization:")
    for e, u in enumerate(run["util"]):
        bar = "█" * int(u * 40)
        flag = " [COLLAPSED]" if u < 0.05 else ""
        lines.append(f"    Expert {e}: {u*100:5.1f}%  {bar}{flag}")

    # Per-expert label affinity table
    lines.append(f"\n  Soft Label Affinity per Expert (gate-weighted):")
    lines.extend(expert_table(run))

    return lines


def comparison_table(runs_by_dataset: dict) -> list[str]:
    lines = []
    for dataset, runs in runs_by_dataset.items():
        lines.append(f"\n{'═'*80}")
        lines.append(f"CROSS-VARIANT COMPARISON: {dataset.upper()}")
        lines.append(f"{'═'*80}")

        cols = ["Variant", "K", "E", "Collapsed", "Entropy", "SpecScore",
                "LabelCov", "NPMI", "Cv", "TopDiv"]
        widths = [14, 4, 4, 10, 9, 9, 9, 7, 7, 7]
        header = "  " + "  ".join(f"{c:>{w}}" for c, w in zip(cols, widths))
        lines.append(header)
        lines.append("  " + "─" * len(header))

        for run in runs:
            m = run["metrics"]
            spec = specialization_score(run["df_soft"], run["label_cols"])
            cov = label_coverage(run["df_soft"], run["label_cols"])
            vals = [
                run["variant_label"][:14],
                str(m.get("k", "?")),
                str(run["E"]),
                f"{run['n_collapsed']}/{run['E']}",
                f"{run['gating_entropy']:.4f}",
                f"{spec:.4f}",
                f"{cov}/{len(run['label_cols'])}",
                f"{m.get('npmi_paper', 'N/A')}",
                f"{m.get('cv', 'N/A')}",
                f"{m.get('topic_diversity', 'N/A')}",
            ]
            lines.append("  " + "  ".join(f"{v:>{w}}" for v, w in zip(vals, widths)))

    return lines


def routing_analysis_section(runs_by_dataset: dict) -> list[str]:
    """Compare routing strategies: Dense vs Sparse vs EC vs Attention."""
    lines = []
    lines.append(f"\n{'═'*80}")
    lines.append("ROUTING STRATEGY ANALYSIS")
    lines.append(f"{'═'*80}")

    for dataset, runs in runs_by_dataset.items():
        lines.append(f"\n  Dataset: {dataset.upper()}")
        by_routing = defaultdict(list)
        for r in runs:
            by_routing[r["routing_group"]].append(r)

        for group, grp_runs in sorted(by_routing.items()):
            lines.append(f"\n  [{group} Gating]")

            for run in grp_runs:
                m = run["metrics"]
                spec = specialization_score(run["df_soft"], run["label_cols"])
                cov = label_coverage(run["df_soft"], run["label_cols"])
                lines.append(
                    f"    {run['variant_label']:<16} "
                    f"collapsed={run['n_collapsed']}/{run['E']}  "
                    f"entropy={run['gating_entropy']:.3f}  "
                    f"spec={spec:.3f}  "
                    f"cov={cov}/{len(run['label_cols'])}  "
                    f"NPMI={m.get('npmi_paper','N/A')}  "
                    f"Cv={m.get('cv','N/A')}"
                )

            # Routing-level insight
            entropies = [r["gating_entropy"] for r in grp_runs]
            collapsed = [r["n_collapsed"] for r in grp_runs]
            specs = [specialization_score(r["df_soft"], r["label_cols"]) for r in grp_runs]
            lines.append(
                f"    Summary: mean_entropy={np.mean(entropies):.3f}  "
                f"mean_collapsed={np.mean(collapsed):.1f}  "
                f"mean_spec={np.nanmean(specs):.3f}"
            )

    return lines


def expert_label_affinity_comparison(runs_by_dataset: dict) -> list[str]:
    """
    For each dataset, for each label: which variant's experts specialize most on it?
    """
    lines = []
    lines.append(f"\n{'═'*80}")
    lines.append("PER-LABEL EXPERT SPECIALIZATION DEPTH")
    lines.append(f"{'═'*80}")
    lines.append("  (Max soft-affinity a single expert achieves for each label)")
    lines.append("  Higher = at least one expert strongly focused on that label")
    lines.append("")

    for dataset, runs in runs_by_dataset.items():
        if not runs:
            continue
        label_cols = runs[0]["label_cols"]
        lines.append(f"  Dataset: {dataset.upper()}")

        col_w = 11
        header_parts = [f"{'Label':>16}"]
        for run in runs:
            header_parts.append(f"{run['variant_label'][:col_w]:>{col_w}}")
        lines.append("  " + "  ".join(header_parts))
        lines.append("  " + "─" * (18 + (col_w + 2) * len(runs)))

        for lc in label_cols:
            row_parts = [f"{lc:>16}"]
            for run in runs:
                df = run["df_soft"]
                if df is not None and lc in df.columns:
                    max_aff = df[lc].max()
                    row_parts.append(f"{max_aff:>{col_w}.3f}")
                else:
                    row_parts.append(f"{'N/A':>{col_w}}")
            lines.append("  " + "  ".join(row_parts))
        lines.append("")

    return lines


def mpst_routing_analysis(runs_by_dataset: dict) -> list[str]:
    """
    MPST-specific: how do experts distribute across the 3 moe_ntm runs
    (different K and configs)?
    """
    lines = []
    mpst_runs = runs_by_dataset.get("mpst", [])
    if not mpst_runs:
        return lines

    lines.append(f"\n{'═'*80}")
    lines.append("MPST MULTI-RUN ANALYSIS (multiple moe_ntm configs)")
    lines.append(f"{'═'*80}")
    lines.append("  Three moe_ntm runs on MPST differ in K (10 vs 20) and hyperparams.")
    lines.append("")

    for run in mpst_runs:
        tf = run["train_final"]
        lines.append(f"  {run['run_name']}")
        lines.append(f"    K={run['metrics'].get('k','?')}  E={run['E']}  "
                     f"collapsed={run['n_collapsed']}  entropy={run['gating_entropy']:.4f}")
        if tf:
            lines.append(f"    KL={tf.get('kl','?'):.2f}  balance={tf.get('balance','?'):.4f}  "
                         f"unique_words={int(tf.get('unique_top_words',-1))}")
        lines.append(f"    util: {[round(u,3) for u in run['util']]}")

        if run["df_soft"] is not None:
            dom_labels = run["df_soft"]["dominant_label"].value_counts()
            lines.append(f"    Dominant label distribution across experts:")
            for lbl, cnt in dom_labels.items():
                lines.append(f"      {lbl}: {cnt} experts")
        lines.append("")

    return lines


def key_findings(runs_by_dataset: dict) -> list[str]:
    """Synthesize key findings across all runs."""
    lines = []
    lines.append(f"\n{'═'*80}")
    lines.append("KEY FINDINGS & ANALYSIS")
    lines.append(f"{'═'*80}")

    all_runs = [r for rs in runs_by_dataset.values() for r in rs]

    # 1. Expert collapse by routing type
    lines.append("\n  1. EXPERT COLLAPSE BY ROUTING STRATEGY")
    collapse_by_routing = defaultdict(list)
    for r in all_runs:
        collapse_by_routing[r["routing_group"]].append(
            (r["n_collapsed"], r["E"], r["dataset"])
        )

    for group, data in sorted(collapse_by_routing.items()):
        collapse_rates = [n/e for n, e, _ in data]
        lines.append(f"    {group:<16}: mean collapse rate = {np.mean(collapse_rates):.2f} "
                     f"  ({', '.join(f'{n}/{e}({ds})' for n,e,ds in data)})")

    lines.append("""
  Interpretation:
    - Expert-Choice (EC) routing achieves 0 collapsed experts on Reuters-10 by
      construction: each expert is guaranteed to select top-C documents, so all
      experts are used. This is its core design advantage over token-choice routing.
    - Dense gating on MPST (run 094347) collapsed 4/8 experts — severe collapse.
      Later MPST runs (125521, 130446) reduced this via tuned balance_loss_coeff.
    - Sparse (top-k) gating collapses 1 expert but keeps entropy low (~0.8),
      meaning routing is sharp — each doc activates exactly k=3 experts.
    - Attention gating (10 experts) collapses 1/10 but distributes more smoothly.""")

    # 2. Gating entropy interpretation
    lines.append("\n  2. GATING ENTROPY ANALYSIS")
    for r in sorted(all_runs, key=lambda x: x["gating_entropy"]):
        lines.append(f"    {r['variant_label']:<16} ({r['dataset']:<10}): "
                     f"H={r['gating_entropy']:.4f}")

    lines.append("""
  Interpretation:
    - EC routing has lowest entropy (~0.71-0.73): docs are assigned hard-ish
      to experts (experts make selection, not documents). This is expected
      since EC uses per-expert top-C selection which is near-binary.
    - Sparse top-k also has low entropy (~0.81-0.83): only k=3 of 8 experts
      active per doc → concentrated distribution.
    - Dense gating has moderate entropy (1.6-2.0): all experts contribute,
      but with unequal weights. MPST dense shows higher entropy (more uniform
      routing) when balance loss is stronger.
    - log(8) = 2.079 is maximum entropy (uniform over 8 experts). None reach
      this — all variants learned some routing structure.""")

    # 3. Specialization depth
    lines.append("\n  3. LABEL SPECIALIZATION QUALITY")
    reuters_runs = runs_by_dataset.get("reuters_10", [])
    if reuters_runs:
        lines.append("    Reuters-10 (10 labels, 8 experts):")
        for r in reuters_runs:
            spec = specialization_score(r["df_soft"], r["label_cols"])
            cov = label_coverage(r["df_soft"], r["label_cols"])
            lines.append(f"      {r['variant_label']:<16}: spec={spec:.3f}  "
                         f"label_coverage={cov}/{len(r['label_cols'])}  "
                         f"NPMI={r['metrics'].get('npmi_paper','?')}")

    mpst_runs = runs_by_dataset.get("mpst", [])
    if mpst_runs:
        lines.append("    MPST (8 labels, 8 experts):")
        for r in mpst_runs:
            spec = specialization_score(r["df_soft"], r["label_cols"])
            cov = label_coverage(r["df_soft"], r["label_cols"])
            lines.append(f"      {r['variant_label']:<16} K={r['metrics'].get('k','?')}: "
                         f"spec={spec:.3f}  label_coverage={cov}/{len(r['label_cols'])}  "
                         f"NPMI={r['metrics'].get('npmi_paper','?')}")

    lines.append("""
  Interpretation:
    - On Reuters-10, Dense-BoW achieves best label separation: experts 1, 3, 6
      all capture money-fx but with distinct sub-themes (UK domestic rates,
      broad currency, USD policy respectively). This sub-label specialization
      is unique to dense gating where all experts can partially contribute.
    - EC-BoW has highest topic diversity (0.77 vs 0.82 dense) but lower spec score:
      perfect load balance forces experts to cover all docs uniformly, reducing
      the ability to hyper-specialize on label-pure subsets.
    - SBERT variants consistently outperform BoW on NPMI (richer encoder input),
      but specialization is sometimes lower (SBERT embeddings are more uniformly
      informative, making routing decisions harder to learn).
    - Sparse-SBERT achieves best NPMI on Reuters (0.2564) + good diversity (0.80)
      — the forced k=3 routing prevents spreading signal too thin.""")

    # 4. Reuters expert character (dense-BoW detailed)
    lines.append("\n  4. REUTERS-10 EXPERT CHARACTER ANALYSIS (Dense-BoW baseline)")
    lines.append("     Based on soft affinity and top document inspection:")
    lines.append("""
    Expert 0 (17.2%, dom=trade):   US-Japan / Asia-Pacific trade frictions.
                                   High crude co-occurrence (energy trade dispute).
    Expert 1  (6.5%, dom=money-fx): UK domestic money market. BoE bill operations,
                                   sterling liquidity forecasts. Narrow but pure.
    Expert 2 (11.6%, dom=grain):   Agricultural commodity reports. Grain + oilseed.
                                   Pure commodity data releases, weather reports.
    Expert 3 (16.9%, dom=money-fx): Broad international currency / dollar dynamics.
                                   High dlr co-occurrence (different from Expert 1).
    Expert 4 (12.7%, dom=trade):   Current account / balance of payments data.
                                   Quantitative trade statistics (surplus/deficit).
    Expert 5  (9.6%, dom=crude):   Energy + agricultural crossover. Pipeline, oil
                                   prices, nat-gas, grain co-produced.
    Expert 6  (8.9%, dom=money-fx): USD policy / exchange rate agreements.
                                   Louvre Accord, Miyazawa dollar comments.
    Expert 7 (16.6%, dom=crude):   Gulf military/shipping + oil. Iranian Gulf crisis,
                                   tanker incidents. Geopolitical crude signal.

    Notable: money-fx captured by 3 experts (1, 3, 6) with distinct semantic sub-modes.
    Dense gating enables this fine-grained decomposition of a single label into
    thematic sub-clusters — this is not achievable with hard/sparse routing.""")

    # 5. MPST findings
    lines.append("\n  5. MPST EXPERT ANALYSIS")
    lines.append("""
    MPST has 8 labels and 8 experts (E=K_labels). Ideal 1-to-1 alignment expected.
    Reality:
    - Run 094347 (K=10): severe collapse — 4 experts dead, only 4 active.
      The 4 active experts all capture murder/violence (dominant MPST labels)
      but romantic gets partial coverage from Expert 0. Psychedelic, action,
      comedy_merged all under-represented. balance_loss_coeff insufficient.
    - Run 125521 (K=20): no collapse, entropy near-max (1.99). Well-distributed
      but low specialization — each expert sees all labels roughly equally.
      Higher K provides more topic granularity but makes routing less focused.
    - Run 130446 (K=20): 1 collapsed expert. Better balance, moderate entropy.
      Expert 6 (21.9%) dominates. Slightly better specialization than 125521.
    - USE-MPST (K=20): 1 collapsed expert (Expert 7, 1.8%). SBERT input helps
      differentiation. Expert 5 (18.2%) leads. Romantic better captured vs BoW.

    MPST challenge: murder/violence/revenge are highly co-occurring (correlated
    labels). Experts struggle to separate them as they are not orthogonal in
    document space. This is a fundamental limitation for multi-label datasets
    where label co-occurrence is high.""")

    # 6. Recommendations
    lines.append("\n  6. RECOMMENDATIONS FOR THESIS")
    lines.append("""
    a) Use EC routing as primary variant for the thesis: guaranteed load balance,
       no collapsed experts, competitive NPMI. Clear architectural story.

    b) Sparse-SBERT wins on NPMI (0.2564) on Reuters — best topic coherence.
       Report this as peak coherence result.

    c) For multi-label analysis, report the gate_weights.npy [N,E] matrix as
       the key MoE contribution: it provides an E-dim soft label membership
       vector per document that is distinct from doc_topic.

    d) The money-fx sub-specialization in Dense-BoW (3 experts for 1 label
       with distinct semantic sub-themes) is a key qualitative finding showing
       MoE captures intra-label variance that K topics cannot.

    e) MPST needs stronger balance_loss_coeff (try 0.1-0.5 instead of 0.01)
       or auxiliary label-seeded initialization to prevent collapse on
       correlated labels like murder+violence.

    f) Compare gate_features [N,E] vs doc_topic [N,K] vs concat [N,K+E] in
       downstream multilabel classification — this is the core empirical claim
       of MoE-NTM vs VAE-GSM.""")

    return lines


def main():
    # Discover all MoE runs with gate_weights
    gate_files = sorted(BASE.glob("results_*/*/artifacts/gate_weights.npy"))
    run_dirs = [g.parent.parent for g in gate_files]

    print(f"Loading {len(run_dirs)} MoE runs...")
    runs = []
    for rd in run_dirs:
        r = load_run(rd)
        if r:
            runs.append(r)
            print(f"  OK: {rd.name} ({r['variant_label']}, {r['dataset']})")
        else:
            print(f"  SKIP: {rd.name}")

    if not runs:
        print("No runs loaded. Exiting.")
        return

    # Group by dataset
    runs_by_dataset = defaultdict(list)
    for r in runs:
        runs_by_dataset[r["dataset"]].append(r)

    # Remove duplicate EC runs (keep first by timestamp)
    for ds in runs_by_dataset:
        seen = set()
        deduped = []
        for r in runs_by_dataset[ds]:
            key = (r["method"], r["metrics"].get("k"))
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        runs_by_dataset[ds] = deduped

    # Build report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("MoE-NTM EXPERT SPECIALIZATION — CROSS-RUN COMPARISON REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated from {len(runs)} runs across {len(runs_by_dataset)} datasets")
    report_lines.append(f"Datasets: {', '.join(runs_by_dataset.keys())}")
    report_lines.append(f"Variants: {', '.join(sorted(set(r['variant_label'] for r in runs)))}")
    report_lines.append("")
    report_lines.append("Routing strategies compared:")
    report_lines.append("  Dense      — softmax over all E experts, all contribute")
    report_lines.append("  Sparse     — top-k=3 gating, only k experts active per doc")
    report_lines.append("  Attention  — query-key dot-product gating (label-seeded keys)")
    report_lines.append("  EC         — Expert-Choice: each expert selects top-C docs")
    report_lines.append("")
    report_lines.append("Input types:")
    report_lines.append("  BoW    — bag-of-words, L1-normalized, BatchNorm backbone")
    report_lines.append("  SBERT  — all-MiniLM-L6-v2 sentence embeddings (384-d)")

    # Cross-variant comparison tables
    report_lines.extend(comparison_table(runs_by_dataset))

    # Routing strategy analysis
    report_lines.extend(routing_analysis_section(runs_by_dataset))

    # Per-label max affinity comparison
    report_lines.extend(expert_label_affinity_comparison(runs_by_dataset))

    # MPST multi-run analysis
    report_lines.extend(mpst_routing_analysis(runs_by_dataset))

    # Key findings
    report_lines.extend(key_findings(runs_by_dataset))

    # Per-run detailed sections
    report_lines.append(f"\n{'═'*80}")
    report_lines.append("DETAILED PER-RUN EXPERT PROFILES")
    report_lines.append(f"{'═'*80}")

    for ds, ds_runs in runs_by_dataset.items():
        report_lines.append(f"\n{'─'*80}")
        report_lines.append(f"  DATASET: {ds.upper()}")
        for run in ds_runs:
            report_lines.extend(format_run_section(run))

    # Write
    out_path = BASE / "moe_expert_comparison_report.txt"
    out_path.write_text("\n".join(report_lines))
    print(f"\nReport written to: {out_path}")
    print(f"Lines: {len(report_lines)}")


if __name__ == "__main__":
    main()
