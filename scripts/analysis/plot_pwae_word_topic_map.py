"""
Visualise PWAE word-attention -> expert -> topic -> topic-word findings.

Reads ONLY the saved word_topic_map outputs (no model reload) and writes PNGs to
  <RUN_DIR>/analysis_attention/word_topic_map/figures/

Figures:
  fig1_expert_topic_heatmap.png  routing matrix mean theta_k [K x T] (collapse + orphan)
  fig2_attn_vs_betarank.png      attended-word alpha vs its beta-rank in mapped topic
  fig3_rank_gap.png              mean beta-rank own-topic vs best-other, per expert
  fig4_top10_overlap.png         attended top-N hits in topic top-10 + jaccard
  fig5_doc_chain.png             one example doc: gate + dom-expert words (alpha, beta-rank)
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

RUN_DIR = "results_20news_5class_pwae/20260606_201027_pwae_ntm_20news_5class"
WTM     = Path(RUN_DIR) / "analysis_attention" / "word_topic_map"
FIG_DIR = WTM / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
V_TOTAL = 2000   # vocab size, for rank colour scale

# ── load saved outputs ────────────────────────────────────────────────────────
etm   = pd.read_csv(WTM / "expert_topic_map.csv")
wtr   = json.load(open(WTM / "word_topic_rank.json"))
ov    = pd.read_csv(WTM / "attn_top10_overlap.csv")
detail = json.load(open(WTM / "detailed_examples.json"))
K = len(etm)
T = sum(c.startswith("theta_t") for c in etm.columns)

# short topic labels from top-3 beta words of each topic
topic_label = {}
for _, r in etm.iterrows():
    topic_label[int(r["mapped_topic"])] = " ".join(r["topic_top10"].split()[:3])

# ── FIG 1: expert -> topic routing heatmap ────────────────────────────────────
M = etm[[f"theta_t{t}" for t in range(T)]].values  # (K, T)
fig, ax = plt.subplots(figsize=(7.5, 5.2))
im = ax.imshow(M, cmap="viridis", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(T))
ax.set_xticklabels([f"t{t}\n{topic_label.get(t,'(orphan)') if t in topic_label else '(orphan)'}"
                    for t in range(T)], fontsize=8)
ax.set_yticks(range(K)); ax.set_yticklabels([f"E{k}" for k in range(K)])
ax.set_xlabel("Topic (decoder beta)"); ax.set_ylabel("Expert")
ax.set_title("Expert -> Topic routing  (mean $\\theta_k$ across corpus)")
for k in range(K):
    t_star = int(M[k].argmax())
    ax.text(t_star, k, "*", ha="center", va="center", color="red", fontsize=18, fontweight="bold")
for k in range(K):
    for t in range(T):
        ax.text(t, k, f"{M[k,t]:.2f}", ha="center", va="center",
                color="white" if M[k,t] < 0.6 else "black", fontsize=7)
# annotate orphan topics (no expert argmax)
claimed = set(int(M[k].argmax()) for k in range(K))
orphan  = [t for t in range(T) if t not in claimed]
fig.colorbar(im, ax=ax, label="mean $\\theta_k$ mass")
note = f"red * = expert's dominant topic   |   collapse: experts sharing a topic   |   orphan topic(s): {orphan}"
fig.text(0.5, 0.01, note, ha="center", fontsize=7.5, color="dimgray")
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(FIG_DIR / "fig1_expert_topic_heatmap.png", dpi=150)
plt.close(fig)

# ── FIG 2: attended-word alpha vs beta-rank in mapped topic ───────────────────
fig, axes = plt.subplots(1, K, figsize=(3.0*K, 3.6), sharey=True)
if K == 1: axes = [axes]
for k in range(K):
    ax = axes[k]
    e = wtr[f"expert{k}"]; t_star = e["mapped_topic"]
    ranks  = np.array([w["rank_in_mapped_topic"] for w in e["attended_words"]])
    alphas = np.array([w["mean_alpha"] for w in e["attended_words"]])
    in10   = np.array([w["in_topic_top10"] for w in e["attended_words"]])
    words  = [w["word"] for w in e["attended_words"]]
    ax.scatter(ranks[~in10], alphas[~in10], c="steelblue", s=28, label="not in top-10")
    ax.scatter(ranks[in10],  alphas[in10],  c="crimson",  s=44, marker="*", label="in topic top-10")
    # label the 4 highest-alpha words
    for i in np.argsort(-alphas)[:4]:
        ax.annotate(words[i], (ranks[i], alphas[i]), fontsize=6,
                    xytext=(3,3), textcoords="offset points")
    ax.set_xscale("log")
    ax.axvline(10, color="green", ls="--", lw=0.8)   # top-10 boundary
    ax.set_title(f"E{k} -> t{t_star} [{topic_label.get(t_star,'')}]", fontsize=8)
    ax.set_xlabel("beta-rank in mapped topic\n(1=top word, log)", fontsize=7)
    if k == 0:
        ax.set_ylabel("mean attention $\\alpha$")
        ax.legend(fontsize=6, loc="upper right")
fig.suptitle("Attention vs interpretation: do high-$\\alpha$ words rank high in the mapped topic?",
             fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(FIG_DIR / "fig2_attn_vs_betarank.png", dpi=150)
plt.close(fig)

# ── FIG 3: rank-gap per expert (own vs best-other) ────────────────────────────
own   = [wtr[f"expert{k}"]["mean_rank_in_mapped_topic"]    for k in range(K)]
other = [wtr[f"expert{k}"]["mean_best_rank_other_topic"]   for k in range(K)]
tstar = [wtr[f"expert{k}"]["mapped_topic"]                 for k in range(K)]
x = np.arange(K); w = 0.38
fig, ax = plt.subplots(figsize=(8, 4.5))
b1 = ax.bar(x - w/2, own,   w, label="rank in mapped topic", color="crimson")
b2 = ax.bar(x + w/2, other, w, label="best rank in any OTHER topic", color="steelblue")
ax.set_xticks(x); ax.set_xticklabels([f"E{k}\n->t{tstar[k]}" for k in range(K)])
ax.set_ylabel("mean beta-rank of top-15 attended words\n(LOWER = more central to topic)")
ax.set_title("Rank-gap: attended words' centrality in mapped vs other topics")
ax.legend(fontsize=8)
for i in range(K):
    gap = other[i] - own[i]
    col = "green" if gap > 0 else "darkred"
    ax.text(i, max(own[i], other[i]) + 30, f"gap {gap:+.0f}", ha="center", fontsize=8, color=col)
fig.text(0.5, 0.005, "gap>0 (green): attention reads the mapped topic's signature vocab   |   "
         "gap<0 (red): attended words are more central elsewhere (routing != interpretation)",
         ha="center", fontsize=7, color="dimgray")
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(FIG_DIR / "fig3_rank_gap.png", dpi=150)
plt.close(fig)

# ── FIG 4: top-10 overlap + jaccard ───────────────────────────────────────────
hit_col = [c for c in ov.columns if c.startswith("top10_hits")][0]
jac_col = [c for c in ov.columns if c.startswith("jaccard")][0]
hits = ov[hit_col].values; jac = ov[jac_col].values
labels = [f"E{r.expert}->t{r.mapped_topic}" for r in ov.itertuples()]
fig, ax1 = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(ov))
bars = ax1.bar(x, hits, color="mediumseagreen", width=0.55)
ax1.set_ylabel("topic top-10 words found in top-50 attended", color="seagreen")
ax1.set_ylim(0, 10); ax1.set_xticks(x); ax1.set_xticklabels(labels)
ax1.set_title("Attention ∩ topic interpretation vocabulary")
for r in ov.itertuples():
    hw = getattr(r, "top10_hit_words")
    if isinstance(hw, str) and hw.strip():
        ax1.text(r.Index, getattr(r, hit_col)+0.15, hw, ha="center", fontsize=6, color="darkgreen")
ax2 = ax1.twinx()
ax2.plot(x, jac, "o-", color="indigo", label="Jaccard(top-50 attn, top-50 beta)")
ax2.set_ylabel("Jaccard", color="indigo"); ax2.set_ylim(0, max(0.2, jac.max()*1.3))
ax2.legend(fontsize=7, loc="upper right")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig4_top10_overlap.png", dpi=150)
plt.close(fig)

# ── FIG 5: single-doc chain (pick motorcycle example, else first) ─────────────
ex = next((d for d in detail if d["true_label"] == "rec.motorcycles"), detail[0])
ke = ex["dom_expert"]; t_star = ex["dom_expert_mapped_topic"]
gate = ex["gate"]
dom_words = next(e for e in ex["experts"] if e["expert"] == ke)["top_attended_words"]

fig = plt.figure(figsize=(11, 4.8))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.6, 0.05], wspace=0.35)

# panel A: gate over experts
axg = fig.add_subplot(gs[0, 0])
gx = list(gate.keys()); gv = list(gate.values())
cols = ["crimson" if g == f"E{ke}" else "lightgray" for g in gx]
axg.bar(gx, gv, color=cols)
axg.set_title(f"Gate (doc {ex['doc_idx']})\ntrue={ex['true_label']}", fontsize=9)
axg.set_ylabel("gate weight"); axg.set_ylim(0, 1)
axg.text(0, 0.9, f"dominant E{ke}->t{t_star}", fontsize=8, color="crimson")

# panel B: dom-expert attended words, alpha bars colored by beta-rank in mapped topic
axw = fig.add_subplot(gs[0, 1])
words = [w["word"] for w in dom_words][::-1]
alphas = [w["alpha"] for w in dom_words][::-1]
rmap  = [w["rank_in_dom_topic"] for w in dom_words][::-1]
in10  = [w["in_dom_topic_top10"] for w in dom_words][::-1]
y = np.arange(len(words))
norm = LogNorm(vmin=1, vmax=V_TOTAL)
colors = plt.cm.RdYlGn_r(norm(rmap))
bars = axw.barh(y, alphas, color=colors)
axw.set_yticks(y); axw.set_yticklabels(words, fontsize=9)
axw.set_xlabel("attention $\\alpha$")
axw.set_title(f"E{ke} attended words -> topic {t_star} [{topic_label.get(t_star,'')}]\n"
              f"bar colour = word's beta-rank in topic {t_star} (green=top word)", fontsize=8)
for i in range(len(words)):
    tag = "  *top10" if in10[i] else ""
    axw.text(alphas[i] + 0.01, y[i], f"rank {rmap[i]}{tag}", va="center", fontsize=7)
axw.set_xlim(0, max(alphas)*1.35)
# colorbar
cax = fig.add_subplot(gs[0, 2])
sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=norm); sm.set_array([])
fig.colorbar(sm, cax=cax, label="beta-rank (log)")

fig.suptitle("Per-doc chain: word -> attention -> dominant expert -> mapped topic -> topic-word rank",
             fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(FIG_DIR / "fig5_doc_chain.png", dpi=150)
plt.close(fig)

print(f"Saved figures -> {FIG_DIR}/")
for f in sorted(FIG_DIR.glob("*.png")):
    print(f"  {f.name}")
