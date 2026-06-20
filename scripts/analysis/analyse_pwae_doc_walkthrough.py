"""
PWAE-NTM: Document Forward-Pass Walkthrough Visualizer

For the canonical best run, produces:
  1. Per-label figure: gate bar + dominant-expert attention bar
  2. Combined grid figure: all labels in one panel

Outputs saved to analysis_attention/walkthrough/.

Edit MODEL_PATH and RUN_DIR at top before running.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from collections import defaultdict

# ── Config ─────────────────────────────────────────────────────────────────────
RUN_DIR = "results_reuters_10_pwae/20260602_004032_pwae_ntm_reuters_10"

LABEL_COLS = ["interest", "money-fx", "trade", "bop", "crude",
              "ship", "nat-gas", "grain", "oilseed", "dlr"]

# Corpus-level expert semantic labels (from mean_attn_per_expert, top words)
EXPERT_LABELS = {
    0: "commodity\nprices",
    1: "grain+energy\n+forex",
    2: "numeric\n(dormant)",
    3: "maritime\nfinance",
    4: "geography\n(countries)",
    5: "gulf\ngeography",
    6: "infra\nexport",
    7: "quantities",
    8: "temporal\npolicy",
    9: "trade\nactions",
}

# Colour palette
COL_DOMINANT = "#e06c1e"   # orange — dominant expert
COL_OTHER    = "#aec6cf"   # muted blue — other experts
COL_ATTN     = "#4a90d9"   # blue — attention bars
COL_ZERO     = "#d9d9d9"   # light grey — near-zero gate

OUTPUT_DIR = Path(RUN_DIR) / "analysis_attention" / "walkthrough"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading correct_docs_attention.json ...")
with open(Path(RUN_DIR) / "analysis_attention" / "correct_docs_attention.json") as f:
    docs = json.load(f)

K = 10
expert_keys = [f"E{k}" for k in range(K)]


# ── Pick best representative doc per label ────────────────────────────────────
def pick_best(label, docs, prefer_single=True):
    """Highest max-gate doc for this label; prefer single-label docs."""
    candidates = [d for d in docs if label in d["true_labels"]]
    if not candidates:
        return None
    if prefer_single:
        single = [d for d in candidates if len(d["true_labels"]) == 1]
        pool = single if single else candidates
    else:
        pool = candidates
    return max(pool, key=lambda d: max(d["gate_weights"].values()))


best = {lbl: pick_best(lbl, docs) for lbl in LABEL_COLS}


# ── Helpers ────────────────────────────────────────────────────────────────────
def wrap_text(text, max_chars=90):
    """Wrap text snippet to max_chars per line, max 2 lines."""
    text = text.replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rfind(" ")
    line1 = text[:cut] if cut > 0 else text[:max_chars]
    rest = text[cut+1:]
    if len(rest) > max_chars:
        rest = rest[:max_chars - 3] + "…"
    return line1 + "\n" + rest


def gate_arrays(doc):
    gw = doc["gate_weights"]
    vals = np.array([gw[f"E{k}"] for k in range(K)])
    return vals


def attn_arrays(doc, expert_idx, n=5):
    key = f"E{expert_idx}"
    attn = doc["expert_top5_attn"].get(key, {})
    words = list(attn.keys())[:n]
    effs  = [attn[w]["eff"] for w in words]
    alphas = [attn[w]["alpha"] for w in words]
    return words, effs, alphas


# ── Single-label figure ────────────────────────────────────────────────────────
def make_doc_figure(doc, label, save_path):
    dom = doc["dom_expert"]
    gate_vals = gate_arrays(doc)
    max_gate  = gate_vals.max()

    words, effs, alphas = attn_arrays(doc, dom, n=5)

    fig = plt.figure(figsize=(11, 4.5))
    fig.patch.set_facecolor("#fafafa")

    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.42,
                           left=0.06, right=0.97, top=0.82, bottom=0.18)

    # ── Left: gate bar chart ──────────────────────────────────────────────────
    ax_gate = fig.add_subplot(gs[0])
    colors = [COL_DOMINANT if k == dom else
              (COL_ZERO if gate_vals[k] < 0.02 else COL_OTHER)
              for k in range(K)]
    x = np.arange(K)
    bars = ax_gate.bar(x, gate_vals, color=colors, edgecolor="white",
                       linewidth=0.8, zorder=3)

    # annotate dominant bar
    ax_gate.text(dom, gate_vals[dom] + 0.015,
                 f"{gate_vals[dom]:.2f}",
                 ha="center", va="bottom", fontsize=9,
                 fontweight="bold", color=COL_DOMINANT)

    ax_gate.set_xticks(x)
    ax_gate.set_xticklabels([f"E{k}" for k in range(K)], fontsize=8)
    ax_gate.set_ylim(0, min(1.0, max_gate + 0.18))
    ax_gate.set_ylabel("Gate weight  $g_k$", fontsize=9)
    ax_gate.set_xlabel("Expert  k", fontsize=9)
    ax_gate.set_title("Expert gate distribution", fontsize=10, fontweight="bold", pad=6)
    ax_gate.axhline(1/K, color="#888", linestyle="--", linewidth=0.8,
                    label=f"uniform (1/K={1/K:.2f})", zorder=2)
    ax_gate.legend(fontsize=7.5, loc="upper right")
    ax_gate.tick_params(axis="y", labelsize=8)
    ax_gate.set_facecolor("white")
    ax_gate.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)

    # expert semantic label below dominant bar
    sem = EXPERT_LABELS.get(dom, "")
    ax_gate.text(dom, -0.12, sem, ha="center", va="top",
                 fontsize=7, color=COL_DOMINANT, transform=ax_gate.get_xaxis_transform())

    # ── Right: attention bar chart for dominant expert ────────────────────────
    ax_attn = fig.add_subplot(gs[1])
    yw = np.arange(len(words))
    effs_arr = np.array(effs)
    h_bars = ax_attn.barh(yw, effs_arr, color=COL_ATTN, edgecolor="white",
                          linewidth=0.8, zorder=3)

    for i, (w, e) in enumerate(zip(words, effs_arr)):
        ax_attn.text(e + 0.01, i, f"{e:.3f}", va="center", fontsize=8.5)

    ax_attn.set_yticks(yw)
    ax_attn.set_yticklabels(words, fontsize=10, fontweight="bold")
    ax_attn.set_xlabel("Effective attention  (α × idf, normalised)", fontsize=9)
    ax_attn.set_title(f"E{dom} — top attention words", fontsize=10,
                      fontweight="bold", color=COL_DOMINANT, pad=6)
    ax_attn.invert_yaxis()
    ax_attn.set_xlim(0, min(1.0, max(effs_arr) + 0.15))
    ax_attn.set_facecolor("white")
    ax_attn.grid(axis="x", linestyle=":", alpha=0.5, zorder=0)

    # ── Title ─────────────────────────────────────────────────────────────────
    snippet = wrap_text(doc["text_snippet"], max_chars=100)
    true_str = ", ".join(doc["true_labels"])
    pred_str = ", ".join(doc["pred_labels"])
    topic_str = f"T{doc['dom_topic']}"
    match_sym = "✓" if set(doc["true_labels"]) == set(doc["pred_labels"]) else "≈"

    fig.suptitle(
        f"Label: {label.upper()}   |   Dom. expert: E{dom} → topic {topic_str}   "
        f"|   pred: [{pred_str}] {match_sym}\n\"{snippet}\"",
        fontsize=9, y=0.97, ha="left", x=0.03,
        fontfamily="monospace", color="#333"
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved: {save_path.name}")


# ── Combined grid figure ───────────────────────────────────────────────────────
def make_combined_figure(best, save_path):
    """
    10-label grid: each row = one label.
    Left column: gate bars. Right column: dominant-expert attention bars.
    """
    n = len(LABEL_COLS)
    fig = plt.figure(figsize=(14, 3.2 * n))
    fig.patch.set_facecolor("#fafafa")

    outer = gridspec.GridSpec(n, 1, figure=fig, hspace=0.55,
                              left=0.04, right=0.98, top=0.97, bottom=0.02)

    for row_i, lbl in enumerate(LABEL_COLS):
        doc = best[lbl]
        if doc is None:
            continue

        dom = doc["dom_expert"]
        gate_vals = gate_arrays(doc)
        words, effs, _ = attn_arrays(doc, dom, n=5)
        effs_arr = np.array(effs)

        inner = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[row_i], wspace=0.38, width_ratios=[1.3, 1]
        )

        # Gate
        ax_g = fig.add_subplot(inner[0])
        colors = [COL_DOMINANT if k == dom else
                  (COL_ZERO if gate_vals[k] < 0.02 else COL_OTHER)
                  for k in range(K)]
        ax_g.bar(np.arange(K), gate_vals, color=colors,
                 edgecolor="white", linewidth=0.6, zorder=3)
        ax_g.axhline(1/K, color="#888", linestyle="--", linewidth=0.7, zorder=2)
        ax_g.set_xticks(np.arange(K))
        ax_g.set_xticklabels([f"E{k}" for k in range(K)], fontsize=7)
        ax_g.set_ylim(0, min(1.0, gate_vals.max() + 0.18))
        ax_g.set_ylabel("$g_k$", fontsize=8)
        ax_g.tick_params(axis="y", labelsize=7)
        ax_g.set_facecolor("white")
        ax_g.grid(axis="y", linestyle=":", alpha=0.45, zorder=0)

        snippet_short = doc["text_snippet"].replace("\n", " ")[:75]
        true_str = ", ".join(doc["true_labels"])
        ax_g.set_title(
            f"{lbl.upper()}  |  E{dom}→T{doc['dom_topic']}  |  g={gate_vals[dom]:.2f}\n"
            f"\"{snippet_short}…\"",
            fontsize=7.5, fontweight="bold", pad=3,
            fontfamily="monospace", loc="left"
        )

        # Attention
        ax_a = fig.add_subplot(inner[1])
        yw = np.arange(len(words))
        ax_a.barh(yw, effs_arr, color=COL_ATTN,
                  edgecolor="white", linewidth=0.6, zorder=3)
        for i, (w, e) in enumerate(zip(words, effs_arr)):
            ax_a.text(e + 0.01, i, f"{e:.3f}", va="center", fontsize=7.5)
        ax_a.set_yticks(yw)
        ax_a.set_yticklabels(words, fontsize=9, fontweight="bold")
        ax_a.set_xlabel("α × idf", fontsize=7.5)
        ax_a.set_title(f"E{dom}: {EXPERT_LABELS.get(dom,'').replace(chr(10),' ')}",
                       fontsize=8, color=COL_DOMINANT, pad=3, fontweight="bold")
        ax_a.invert_yaxis()
        ax_a.set_xlim(0, min(1.0, max(effs_arr) + 0.18))
        ax_a.set_facecolor("white")
        ax_a.grid(axis="x", linestyle=":", alpha=0.45, zorder=0)

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved: {save_path.name}")


# ── Expert corpus-level attention heatmap ─────────────────────────────────────
def make_expert_heatmap(run_dir, save_path, top_n=25,
                        exclude_words=("mar", "apr")):
    """
    Heatmap: experts × top-N vocab words.
    Rows = experts, columns = union of top-N attention words per expert.
    Cell = mean α_k(word) (corpus level).
    exclude_words: tokens present across all experts (date noise) — filtered out.
    """
    with open(Path(run_dir) / "analysis_attention" / "attention_analysis.json") as f:
        analysis = json.load(f)

    exclude = set(exclude_words)

    # union of top-N words per expert, skip excluded
    all_words = []
    for k in range(K):
        top = analysis["top_attn_words_per_expert"][f"E{k}"]["words"][:top_n]
        all_words.extend([w for w in top if w not in exclude])
    # deduplicate preserving order
    seen = set()
    union_words = []
    for w in all_words:
        if w not in seen:
            seen.add(w)
            union_words.append(w)

    # need vocab to get indices
    # mean_attn rows=experts, cols=vocab; we need word→idx
    # use attention_analysis top words + weights to reconstruct mapping
    # (we don't have id2word here without the model, so use analysis data directly)
    # Build [K, len(union_words)] from per-expert top-word weights
    word2experts = defaultdict(dict)
    for k in range(K):
        edata = analysis["top_attn_words_per_expert"][f"E{k}"]
        for w, wt in zip(edata["words"], edata["weights"]):
            word2experts[w][k] = wt

    # matrix: rows=experts, cols=union_words
    mat = np.zeros((K, len(union_words)))
    for j, w in enumerate(union_words):
        for k, wt in word2experts.get(w, {}).items():
            mat[k, j] = wt

    fig, ax = plt.subplots(figsize=(max(14, len(union_words) * 0.38), 4.5))
    fig.patch.set_facecolor("#fafafa")

    im = ax.imshow(mat, aspect="auto", cmap="YlOrBr", vmin=0, vmax=1.0)
    ax.set_yticks(np.arange(K))
    ax.set_yticklabels(
        [f"E{k}: {EXPERT_LABELS[k].replace(chr(10), ' ')}" for k in range(K)],
        fontsize=9
    )
    ax.set_xticks(np.arange(len(union_words)))
    ax.set_xticklabels(union_words, rotation=60, ha="right", fontsize=8)
    ax.set_title(
        f"Expert attention vocabulary heatmap  "
        f"(top-{top_n} words per expert, mean α×idf weight)\n"
        f"Pairwise Jaccard = 0.032 — experts attend to nearly disjoint vocabulary",
        fontsize=10, pad=8
    )

    plt.colorbar(im, ax=ax, label="mean α × idf weight", shrink=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved: {save_path.name}")


# ── Gate routing by label summary ─────────────────────────────────────────────
def make_gate_by_label(docs, save_path):
    """
    Stacked bar: for each label, mean gate weight per expert across all docs with that label.
    Shows which experts dominate each label.
    """
    label_gates = defaultdict(list)
    for doc in docs:
        gw = doc["gate_weights"]
        gate_vec = np.array([gw[f"E{k}"] for k in range(K)])
        for lbl in doc["true_labels"]:
            label_gates[lbl].append(gate_vec)

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("#fafafa")

    x = np.arange(len(LABEL_COLS))
    bar_w = 0.65

    # colour per expert: use a qualitative colormap
    cmap = plt.get_cmap("tab10")
    bottoms = np.zeros(len(LABEL_COLS))

    for k in range(K):
        means = []
        for lbl in LABEL_COLS:
            vecs = label_gates.get(lbl, [])
            means.append(np.mean([v[k] for v in vecs]) if vecs else 0.0)
        means = np.array(means)
        ax.bar(x, means, bar_w, bottom=bottoms,
               color=cmap(k), label=f"E{k}: {EXPERT_LABELS[k].replace(chr(10),' ')}",
               edgecolor="white", linewidth=0.5)
        bottoms += means

    ax.set_xticks(x)
    ax.set_xticklabels([l.upper() for l in LABEL_COLS], fontsize=9)
    ax.set_ylabel("Mean gate weight", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="#aaa", linestyle="--", linewidth=0.7)
    ax.set_title(
        "Mean expert gate distribution per Reuters-10 label\n"
        "(correctly classified test docs, RF seed=42)",
        fontsize=11, pad=8
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.22, 1.0),
              fontsize=8, framealpha=0.9, title="Expert", title_fontsize=8)
    ax.set_facecolor("white")
    ax.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)
    fig.subplots_adjust(right=0.78)

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved: {save_path.name}")


# ── Run ────────────────────────────────────────────────────────────────────────
print(f"\nOutput dir: {OUTPUT_DIR}\n")

# 1. Per-label individual figures
print("Generating per-label walkthrough figures ...")
for lbl in LABEL_COLS:
    doc = best[lbl]
    if doc is None:
        print(f"  {lbl}: no correctly classified docs — skipping")
        continue
    out = OUTPUT_DIR / f"walkthrough_{lbl.replace('-','_')}.png"
    make_doc_figure(doc, lbl, out)

# 2. Combined grid
print("\nGenerating combined grid figure ...")
make_combined_figure(best, OUTPUT_DIR / "walkthrough_all_labels.png")

# 3. Expert heatmap
print("\nGenerating expert attention heatmap ...")
make_expert_heatmap(RUN_DIR, OUTPUT_DIR / "expert_attn_heatmap.png", top_n=20)

# 4. Gate by label stacked bar
print("\nGenerating gate-by-label routing figure ...")
make_gate_by_label(docs, OUTPUT_DIR / "gate_routing_by_label.png")

print("\nDone. Files:")
for p in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {p}")
