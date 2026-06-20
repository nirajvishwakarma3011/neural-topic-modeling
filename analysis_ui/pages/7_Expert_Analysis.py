"""
Expert Analysis — MoE-NTM expert word associations and cross-run comparison.

Tabs:
  Single Run  — pick one MoE run, explore each expert's words (3 approaches),
                utilization, topic affinity heatmap, label affinity
  Cross-Run   — compare all MoE runs on the same dataset side-by-side:
                specialization scores, label coverage, per-label affinity,
                expert word comparison across variants
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from loader import REPO_ROOT, list_runs, _load_json

st.set_page_config(page_title="Expert Analysis", layout="wide")
st.title("MoE-NTM — Expert Analysis")

# ── Helpers ───────────────────────────────────────────────────────────────────

VARIANT_LABELS = {
    "moe_ntm_use_sparse": "Sparse-SBERT",
    "moe_ntm_use_attn":   "Attn-SBERT",
    "moe_ntm_use_ec":     "EC-SBERT",
    "moe_ntm_use":        "Dense-SBERT",
    "moe_ntm_sparse":     "Sparse-BoW",
    "moe_ntm_attn":       "Attn-BoW",
    "moe_ntm_ec":         "EC-BoW",
    "moe_ntm":            "Dense-BoW",
}

ROUTING_COLOR = {
    "Dense":         "#4C78A8",
    "Sparse":        "#F58518",
    "Attention":     "#54A24B",
    "Expert-Choice": "#E45756",
}

def variant_label(run_name: str) -> str:
    for key in VARIANT_LABELS:
        if key in run_name:
            return VARIANT_LABELS[key]
    return run_name

def routing_group(lbl: str) -> str:
    if "Dense" in lbl:   return "Dense"
    if "Sparse" in lbl:  return "Sparse"
    if "Attn" in lbl:    return "Attention"
    if "EC" in lbl:      return "Expert-Choice"
    return "Other"


@st.cache_data(show_spinner=False)
def list_moe_runs() -> list[dict]:
    """Return all runs that have gate_weights.npy + expert_words.json."""
    runs = []
    for g in sorted(REPO_ROOT.glob("results_*/*/artifacts/gate_weights.npy")):
        run_dir = g.parent.parent
        ew_path = run_dir / "artifacts" / "expert_words.json"
        fp_path = run_dir / "dataset_fingerprint.json"
        if not ew_path.exists() or not fp_path.exists():
            continue
        fp = json.loads(fp_path.read_text())
        m_path = run_dir / "metrics.json"
        metrics = json.loads(m_path.read_text()) if m_path.exists() else {}
        vlabel = variant_label(run_dir.name)
        runs.append({
            "run_dir": run_dir,
            "run_name": run_dir.name,
            "dataset": fp.get("dataset_name", "unknown"),
            "results_root": run_dir.parent.name,
            "variant_label": vlabel,
            "routing": routing_group(vlabel),
            "K": fp.get("k", metrics.get("k", "?")),
            "E": None,  # filled from expert_words
            "metrics": metrics,
            "fp": fp,
        })
    return runs


@st.cache_data(show_spinner=False)
def load_expert_words(run_dir_str: str) -> dict:
    return json.loads((Path(run_dir_str) / "artifacts" / "expert_words.json").read_text())


@st.cache_data(show_spinner=False)
def load_label_affinity(run_dir_str: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    rd = Path(run_dir_str)
    hard = soft = None
    hp = rd / "expert_hard_dist.csv"
    sp = rd / "expert_soft_affinity.csv"
    if hp.exists():
        hard = pd.read_csv(hp)
    if sp.exists():
        soft = pd.read_csv(sp)
    return hard, soft


@st.cache_data(show_spinner=False)
def load_gate_weights(run_dir_str: str) -> np.ndarray:
    return np.load(Path(run_dir_str) / "artifacts" / "gate_weights.npy")


def word_bar(words: list[str], scores: list[float], title: str, color: str = "#4C78A8") -> go.Figure:
    if not words:
        fig = go.Figure()
        fig.add_annotation(text="No words (collapsed expert)", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5)
        fig.update_layout(height=250)
        return fig
    n = min(15, len(words))
    fig = go.Figure(go.Bar(
        x=scores[:n][::-1],
        y=words[:n][::-1],
        orientation="h",
        marker_color=color,
    ))
    fig.update_layout(
        title=title, height=max(250, n * 22),
        margin=dict(l=10, r=10, t=35, b=10),
        xaxis_title="Score", yaxis_title="",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def utilization_bar(util: list[float], highlight_e: int | None = None) -> go.Figure:
    E = len(util)
    colors = []
    for e, u in enumerate(util):
        if u < 0.05:
            colors.append("#d62728")   # red = collapsed
        elif e == highlight_e:
            colors.append("#ff7f0e")   # orange = selected
        else:
            colors.append("#1f77b4")
    fig = go.Figure(go.Bar(
        x=list(range(E)), y=[u * 100 for u in util],
        marker_color=colors,
        text=[f"{u*100:.1f}%" for u in util],
        textposition="outside",
    ))
    fig.add_hline(y=5, line_dash="dot", line_color="red",
                  annotation_text="5% collapse threshold", annotation_position="top right")
    fig.update_layout(
        height=280, margin=dict(l=10, r=10, t=15, b=30),
        xaxis=dict(title="Expert", tickvals=list(range(E))),
        yaxis=dict(title="% docs routed here"),
        showlegend=False,
    )
    return fig


def topic_heatmap(topic_weights_per_expert: list[list[float]], K: int) -> go.Figure:
    E = len(topic_weights_per_expert)
    z = np.array([tw if tw else [0.0] * K for tw in topic_weights_per_expert])
    fig = px.imshow(
        z, aspect="auto", color_continuous_scale="Blues",
        labels=dict(x="Topic", y="Expert", color="Mean Weight"),
        x=[f"T{k}" for k in range(K)],
        y=[f"E{e}" for e in range(E)],
    )
    fig.update_layout(height=max(200, E * 35 + 60),
                      margin=dict(l=10, r=10, t=30, b=30),
                      title="Expert → Topic Affinity (mean doc_topic)")
    return fig


# ── Sidebar: MoE run filter ───────────────────────────────────────────────────

all_runs = list_moe_runs()
if not all_runs:
    st.error("No MoE runs found. Run `python analysis_ui/precompute_expert_words.py` first.")
    st.stop()

datasets = sorted(set(r["dataset"] for r in all_runs))
sel_dataset = st.sidebar.selectbox("Dataset", datasets, key="exp_dataset")
ds_runs = [r for r in all_runs if r["dataset"] == sel_dataset]

# Enrich E from expert_words
for r in ds_runs:
    ew = load_expert_words(str(r["run_dir"]))
    r["E"] = ew["E"]

tab_single, tab_compare = st.tabs(["Single Run Deep Dive", "Cross-Run Comparison"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Run Deep Dive
# ══════════════════════════════════════════════════════════════════════════════

with tab_single:
    run_labels_s = [f"{r['variant_label']}  (K={r['K']}, E={r['E']})  — {r['run_name'][:30]}"
                    for r in ds_runs]
    sel_idx = st.selectbox("Select run", range(len(ds_runs)),
                           format_func=lambda i: run_labels_s[i],
                           key="single_run_idx")
    run = ds_runs[sel_idx]
    ew = load_expert_words(str(run["run_dir"]))
    hard_df, soft_df = load_label_affinity(str(run["run_dir"]))
    gate = load_gate_weights(str(run["run_dir"]))
    E = ew["E"]
    util = ew["expert_utilization"]

    # Top metrics strip
    m = run["metrics"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Variant",  run["variant_label"])
    c2.metric("K topics", str(run["K"]))
    c3.metric("E experts", str(E))
    c4.metric("NPMI",     str(m.get("npmi_paper", "N/A")))
    c5.metric("Cv",       str(m.get("cv", "N/A")))

    n_collapsed = sum(u < 0.05 for u in util)
    if n_collapsed > 0:
        st.warning(f"{n_collapsed}/{E} experts collapsed (utilization < 5%). Shown in red.")

    st.subheader("Expert Utilization")
    sel_expert = st.selectbox("Highlight expert", range(E),
                              format_func=lambda e: f"Expert {e}  ({util[e]*100:.1f}%)",
                              key="single_expert")
    st.plotly_chart(utilization_bar(util, highlight_e=sel_expert), use_container_width=True)

    # Expert-topic heatmap (Approach 3 topic_weights)
    if ew["approach3_topic_affinity"]:
        K_val = ew["K"]
        tw_per_expert = [x.get("topic_weights", []) for x in ew["approach3_topic_affinity"]]
        st.subheader("Expert → Topic Affinity (Approach 3)")
        st.plotly_chart(topic_heatmap(tw_per_expert, K_val), use_container_width=True)
        st.caption(
            "Each row = one expert. Colour = average topic proportion of docs routed to that expert. "
            "Bright diagonal-like pattern = experts specialize on distinct topics. "
            "Uniform rows = experts not topic-specialized."
        )

    # Label affinity heatmap
    if soft_df is not None:
        meta_cols = {"expert", "gate_mass", "dominant_label", "top2_label"}
        label_cols = [c for c in soft_df.columns if c not in meta_cols]
        if label_cols:
            st.subheader("Expert → Label Affinity (soft, gate-weighted)")
            z = soft_df[label_cols].values
            fig_la = px.imshow(
                z, aspect="auto", color_continuous_scale="RdBu_r",
                labels=dict(x="Label", y="Expert", color="Weighted affinity"),
                x=label_cols,
                y=[f"E{e}" for e in range(len(z))],
                zmin=0, zmax=min(1.0, float(z.max()) * 1.1),
            )
            fig_la.update_layout(height=max(200, len(z) * 35 + 60),
                                  margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_la, use_container_width=True)

    # Per-expert word inspection
    st.subheader(f"Expert {sel_expert} — Word Associations")

    e_data = {
        "ap1": next((x for x in ew["approach1_senclu"] if x["expert"] == sel_expert), None),
        "ap2": next((x for x in ew["approach2_decoder"] if x["expert"] == sel_expert), None),
        "ap3": next((x for x in ew["approach3_topic_affinity"] if x["expert"] == sel_expert), None),
    }

    color = "#d62728" if util[sel_expert] < 0.05 else "#4C78A8"
    at1, at2, at3 = st.tabs([
        "Approach 1: SenClu Frequency",
        "Approach 2: Decoder Decomposition",
        "Approach 3: Topic Affinity",
    ])

    with at1:
        st.caption(
            "**SenClu-style:** Words distinctive to this expert via damped frequency × excess routing probability. "
            "`score = √max(n(w,e) − n_min, 0) × (p(e|w) − 1/E)`.  "
            "High score = word appears often in this expert's docs AND rarely in others."
        )
        ap1 = e_data["ap1"]
        if ap1 and ap1["words"]:
            st.plotly_chart(
                word_bar(ap1["words"], ap1["scores"],
                         f"Expert {sel_expert} — SenClu top words", color),
                use_container_width=True
            )
            with st.expander("Raw scores"):
                st.dataframe(pd.DataFrame({"word": ap1["words"], "score": ap1["scores"]}))
        else:
            st.info("No distinctive words (expert collapsed or too few docs).")

    with at2:
        st.caption(
            "**Decoder decomposition:** Mean topic proportions t̄_e of docs routed to this expert "
            "are decoded through β (topic-word matrix) to get expert-specific word distribution. "
            "Shows what vocabulary the expert's topic mix produces."
        )
        ap2 = e_data["ap2"]
        if ap2 and ap2["words"]:
            st.plotly_chart(
                word_bar(ap2["words"], ap2["scores"],
                         f"Expert {sel_expert} — Decoder top words", color),
                use_container_width=True
            )
            if ap2.get("topic_mix"):
                K_val = ew["K"]
                mix_df = pd.DataFrame({
                    "topic": [f"T{k}" for k in range(K_val)],
                    "weight": ap2["topic_mix"],
                }).sort_values("weight", ascending=False)
                st.caption("Topic mix (t̄_e):")
                st.bar_chart(mix_df.set_index("topic")["weight"])
        else:
            st.info("No decoder data (no topic_word_prob available).")

    with at3:
        st.caption(
            "**Topic affinity:** Identifies the top-3 topics this expert handles (by mean t̄_e), "
            "then derives words from those topics' β rows weighted by t̄_e. "
            "Readable as: 'this expert specializes on topics K and reads their words'."
        )
        ap3 = e_data["ap3"]
        if ap3 and ap3.get("top_topic_ids"):
            st.markdown(f"**Top topics:** {ap3['top_topic_ids']}")
            cols_t = st.columns(min(3, len(ap3["top_topic_ids"])))
            for ci, (tid, tw) in enumerate(zip(ap3["top_topic_ids"], ap3["top_topic_words"])):
                with cols_t[ci % 3]:
                    weight = ap3["topic_weights"][tid] if ap3.get("topic_weights") else "?"
                    st.markdown(f"**Topic {tid}** (weight={weight:.3f})")
                    st.write(", ".join(tw[:10]))

            st.plotly_chart(
                word_bar(ap3["weighted_words"], ap3["weighted_scores"],
                         f"Expert {sel_expert} — Affinity-weighted words", color),
                use_container_width=True
            )
        else:
            st.info("No topic affinity data.")

    # Show top docs for selected expert
    st.subheader(f"Expert {sel_expert} — Top Documents")
    sample_path = run["run_dir"] / "expert_doc_samples.csv"
    if sample_path.exists():
        samples = pd.read_csv(sample_path)
        sub = samples[samples["expert"] == sel_expert].reset_index(drop=True)
        if len(sub) == 0:
            st.info("No sample docs (expert collapsed).")
        else:
            for _, row in sub.iterrows():
                with st.expander(
                    f"Rank {int(row['rank_in_expert'])} | gate={row['gate_weight']:.4f} | "
                    f"labels=[{row['active_labels']}]"
                ):
                    st.write(row["text_snippet"])
    else:
        st.info("Run `analyse_expert_labels.py` to generate doc samples.")

    # All experts overview: top words side by side
    st.subheader("All Experts — Top Words Overview")
    approach_choice = st.radio(
        "Approach for overview",
        ["SenClu (Approach 1)", "Decoder (Approach 2)", "Topic Affinity (Approach 3)"],
        horizontal=True, key="overview_approach",
    )
    ap_key = {
        "SenClu (Approach 1)": "approach1_senclu",
        "Decoder (Approach 2)": "approach2_decoder",
        "Topic Affinity (Approach 3)": "approach3_topic_affinity",
    }[approach_choice]
    ap_word_field = {
        "approach1_senclu": "words",
        "approach2_decoder": "words",
        "approach3_topic_affinity": "weighted_words",
    }[ap_key]
    ap_score_field = {
        "approach1_senclu": "scores",
        "approach2_decoder": "scores",
        "approach3_topic_affinity": "weighted_scores",
    }[ap_key]

    experts_data = ew[ap_key]
    n_cols = min(4, E)
    rows = [list(range(i, min(i + n_cols, E))) for i in range(0, E, n_cols)]
    for row_experts in rows:
        cols = st.columns(len(row_experts))
        for ci, e in enumerate(row_experts):
            with cols[ci]:
                ex = next((x for x in experts_data if x["expert"] == e), None)
                u = util[e]
                badge = "🔴" if u < 0.05 else "🟢"
                st.markdown(f"**{badge} Expert {e}** ({u*100:.1f}%)")
                if soft_df is not None:
                    row_s = soft_df[soft_df["expert"] == e]
                    if len(row_s) > 0:
                        dom = row_s["dominant_label"].values[0]
                        top2 = row_s["top2_label"].values[0]
                        st.caption(f"Label: {dom} / {top2}")
                if ex and ex.get(ap_word_field):
                    words = ex[ap_word_field]
                    scores = ex.get(ap_score_field, [1.0] * len(words))
                    st.dataframe(
                        pd.DataFrame({"word": words[:10], "score": [round(s, 5) for s in scores[:10]]}),
                        use_container_width=True, hide_index=True,
                    )
                else:
                    st.caption("*(collapsed)*")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Cross-Run Comparison
# ══════════════════════════════════════════════════════════════════════════════

with tab_compare:
    st.markdown(f"**Dataset:** `{sel_dataset}` — {len(ds_runs)} MoE runs available")

    # ── Metrics comparison table ──────────────────────────────────────────────
    st.subheader("Run Metrics Comparison")
    metric_rows = []
    for r in ds_runs:
        ew_r = load_expert_words(str(r["run_dir"]))
        util_r = ew_r["expert_utilization"]
        n_collapsed = sum(u < 0.05 for u in util_r)
        _g = load_gate_weights(str(r["run_dir"]))
        gate_ent = float(-(_g * np.log(_g + 1e-12)).sum(axis=1).mean())
        # Specialization = mean max soft-affinity
        _, soft_r = load_label_affinity(str(r["run_dir"]))
        spec = float("nan")
        cov = "N/A"
        if soft_r is not None:
            meta_cols = {"expert", "gate_mass", "dominant_label", "top2_label"}
            lc = [c for c in soft_r.columns if c not in meta_cols]
            if lc:
                spec = round(float(soft_r[lc].max(axis=1).mean()), 4)
                cov = f"{len(soft_r['dominant_label'].unique())}/{len(lc)}"
        m_r = r["metrics"]
        metric_rows.append({
            "Variant": r["variant_label"],
            "Routing": r["routing"],
            "K": r["K"],
            "E": r["E"],
            "Collapsed": f"{n_collapsed}/{r['E']}",
            "GateEntropy": round(gate_ent, 4),
            "SpecScore": spec,
            "LabelCov": cov,
            "NPMI": m_r.get("npmi_paper", "N/A"),
            "Cv": m_r.get("cv", "N/A"),
            "TopDiv": m_r.get("topic_diversity", "N/A"),
        })

    metric_df = pd.DataFrame(metric_rows)
    st.dataframe(metric_df, use_container_width=True, hide_index=True)

    # ── Gating entropy comparison bar ────────────────────────────────────────
    st.subheader("Gating Entropy by Variant")
    st.caption(
        "Lower entropy = sharper routing (Sparse/EC). Higher = more diffuse (Dense/Attn). "
        "log(8) = 2.079 is the maximum (uniform over 8 experts)."
    )
    ent_df = metric_df[["Variant", "Routing", "GateEntropy"]].dropna()
    fig_ent = px.bar(
        ent_df, x="Variant", y="GateEntropy", color="Routing",
        color_discrete_map=ROUTING_COLOR,
        text="GateEntropy",
    )
    fig_ent.add_hline(y=2.079, line_dash="dot", annotation_text="log(8) uniform max")
    fig_ent.update_layout(height=300, margin=dict(t=20, b=20))
    st.plotly_chart(fig_ent, use_container_width=True)

    # ── Specialization score comparison ──────────────────────────────────────
    spec_df = metric_df[["Variant", "Routing", "SpecScore"]].dropna(subset=["SpecScore"])
    if not spec_df.empty:
        st.subheader("Specialization Score by Variant")
        st.caption("Mean max-label soft-affinity per expert. Higher = experts focus on distinct labels.")
        fig_spec = px.bar(
            spec_df, x="Variant", y="SpecScore", color="Routing",
            color_discrete_map=ROUTING_COLOR, text="SpecScore",
        )
        fig_spec.update_layout(height=300, margin=dict(t=20, b=20))
        st.plotly_chart(fig_spec, use_container_width=True)

    # ── Per-label max-affinity cross-run heatmap ──────────────────────────────
    st.subheader("Per-Label Max Affinity — Cross-Run Heatmap")
    st.caption(
        "For each label: maximum soft-affinity achieved by any single expert in each run. "
        "Higher = at least one expert strongly specializes on that label."
    )
    label_aff_rows = []
    label_cols_global = None
    for r in ds_runs:
        _, soft_r = load_label_affinity(str(r["run_dir"]))
        if soft_r is None:
            continue
        meta_cols = {"expert", "gate_mass", "dominant_label", "top2_label"}
        lc = [c for c in soft_r.columns if c not in meta_cols]
        if not lc:
            continue
        label_cols_global = lc
        max_aff = soft_r[lc].max(axis=0)
        row = {"Variant": r["variant_label"]}
        row.update(max_aff.to_dict())
        label_aff_rows.append(row)

    if label_aff_rows and label_cols_global:
        la_df = pd.DataFrame(label_aff_rows).set_index("Variant")
        fig_la_cross = px.imshow(
            la_df.values.T,
            x=la_df.index.tolist(),
            y=la_df.columns.tolist(),
            color_continuous_scale="Blues", aspect="auto",
            labels=dict(x="Variant", y="Label", color="Max Affinity"),
        )
        fig_la_cross.update_layout(
            height=max(250, len(label_cols_global) * 30 + 80),
            margin=dict(l=10, r=10, t=10, b=40),
        )
        st.plotly_chart(fig_la_cross, use_container_width=True)

    # ── Expert utilization heatmap (all runs) ─────────────────────────────────
    st.subheader("Expert Utilization — All Runs")
    st.caption("Fraction of docs routed to each expert per run. Red = collapsed (<5%).")
    util_rows = []
    max_E = max(r["E"] for r in ds_runs if r["E"])
    for r in ds_runs:
        ew_r = load_expert_words(str(r["run_dir"]))
        util_r = ew_r["expert_utilization"]
        row = {"Variant": r["variant_label"]}
        for e in range(max_E):
            row[f"E{e}"] = util_r[e] if e < len(util_r) else float("nan")
        util_rows.append(row)
    util_df = pd.DataFrame(util_rows).set_index("Variant")
    fig_util = px.imshow(
        util_df.values,
        x=util_df.columns.tolist(),
        y=util_df.index.tolist(),
        color_continuous_scale="RdYlGn", zmin=0, zmax=0.3,
        aspect="auto",
        labels=dict(x="Expert", y="Variant", color="Fraction"),
    )
    fig_util.update_layout(
        height=max(200, len(util_rows) * 35 + 80),
        margin=dict(l=10, r=10, t=10, b=30),
    )
    st.plotly_chart(fig_util, use_container_width=True)
    st.caption("Green = well-used. Red = collapsed. Compare routing strategies: EC is most uniform.")

    # ── Cross-run word comparison for a specific expert slot ─────────────────
    st.subheader("Expert Word Comparison Across Runs")
    st.caption(
        "Select an expert slot (0–7) and approach. "
        "Shows top words for that expert across all variants side-by-side."
    )
    col_exp, col_ap = st.columns(2)
    with col_exp:
        cmp_expert = st.number_input("Expert slot", min_value=0, max_value=max_E - 1,
                                     value=0, step=1, key="cmp_expert")
    with col_ap:
        cmp_approach = st.radio(
            "Approach",
            ["SenClu (1)", "Decoder (2)", "Topic Affinity (3)"],
            horizontal=True, key="cmp_approach",
        )

    ap_key_cmp = {
        "SenClu (1)": ("approach1_senclu", "words", "scores"),
        "Decoder (2)": ("approach2_decoder", "words", "scores"),
        "Topic Affinity (3)": ("approach3_topic_affinity", "weighted_words", "weighted_scores"),
    }[cmp_approach]
    ap_k, ap_w, ap_s = ap_key_cmp

    n_cols_cmp = min(4, len(ds_runs))
    run_chunks = [ds_runs[i:i+n_cols_cmp] for i in range(0, len(ds_runs), n_cols_cmp)]
    for chunk in run_chunks:
        cols = st.columns(len(chunk))
        for ci, r in enumerate(chunk):
            with cols[ci]:
                ew_r = load_expert_words(str(r["run_dir"]))
                util_r = ew_r["expert_utilization"]
                u = util_r[cmp_expert] if cmp_expert < len(util_r) else 0.0
                badge = "🔴" if u < 0.05 else "🟢"
                st.markdown(f"**{badge} {r['variant_label']}**")
                st.caption(f"util={u*100:.1f}%")

                # Get label affinity for this expert
                _, soft_r = load_label_affinity(str(r["run_dir"]))
                if soft_r is not None:
                    row_s = soft_r[soft_r["expert"] == cmp_expert]
                    if len(row_s) > 0:
                        dom = row_s["dominant_label"].values[0]
                        st.caption(f"Dominant label: **{dom}**")

                ex_list = ew_r.get(ap_k, [])
                ex = next((x for x in ex_list if x["expert"] == cmp_expert), None)
                if ex and ex.get(ap_w):
                    words = ex[ap_w][:10]
                    scores = ex.get(ap_s, [1.0] * len(words))[:10]
                    st.dataframe(
                        pd.DataFrame({
                            "word": words,
                            "score": [round(float(s), 5) for s in scores],
                        }),
                        use_container_width=True, hide_index=True,
                    )
                else:
                    st.caption("*(collapsed or no data)*")

    # ── NPMI vs SpecScore scatter ─────────────────────────────────────────────
    st.subheader("NPMI vs Specialization Score")
    st.caption("Tradeoff: higher specialization doesn't always mean better topic coherence.")
    scatter_rows = []
    for r in ds_runs:
        ew_r = load_expert_words(str(r["run_dir"]))
        util_r = ew_r["expert_utilization"]
        _, soft_r = load_label_affinity(str(r["run_dir"]))
        spec = float("nan")
        if soft_r is not None:
            meta_cols = {"expert", "gate_mass", "dominant_label", "top2_label"}
            lc = [c for c in soft_r.columns if c not in meta_cols]
            if lc:
                spec = float(soft_r[lc].max(axis=1).mean())
        npmi_val = r["metrics"].get("npmi_paper")
        if npmi_val is not None and not np.isnan(spec):
            scatter_rows.append({
                "Variant": r["variant_label"],
                "Routing": r["routing"],
                "NPMI": float(npmi_val),
                "SpecScore": spec,
                "Collapsed": sum(u < 0.05 for u in util_r),
            })
    if scatter_rows:
        sc_df = pd.DataFrame(scatter_rows)
        fig_sc = px.scatter(
            sc_df, x="SpecScore", y="NPMI", color="Routing",
            text="Variant", size_max=12,
            color_discrete_map=ROUTING_COLOR,
            hover_data=["Collapsed"],
        )
        fig_sc.update_traces(textposition="top center", marker_size=10)
        fig_sc.update_layout(height=380, margin=dict(t=20, b=20))
        st.plotly_chart(fig_sc, use_container_width=True)

    # ── Collapsed expert count comparison ────────────────────────────────────
    st.subheader("Collapsed Experts per Variant")
    collapse_rows = []
    for r in ds_runs:
        ew_r = load_expert_words(str(r["run_dir"]))
        util_r = ew_r["expert_utilization"]
        collapse_rows.append({
            "Variant": r["variant_label"],
            "Routing": r["routing"],
            "Collapsed": sum(u < 0.05 for u in util_r),
            "Total": r["E"],
        })
    coll_df = pd.DataFrame(collapse_rows)
    fig_coll = px.bar(
        coll_df, x="Variant", y="Collapsed", color="Routing",
        color_discrete_map=ROUTING_COLOR,
        text="Collapsed",
        title="Number of collapsed experts (util < 5%)",
    )
    fig_coll.update_layout(height=280, margin=dict(t=35, b=20))
    st.plotly_chart(fig_coll, use_container_width=True)
