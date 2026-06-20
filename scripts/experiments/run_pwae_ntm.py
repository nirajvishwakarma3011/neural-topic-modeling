"""
PWAE-NTM: Full training, diagnostics, and evaluation pipeline on Reuters-10.
Phases 5-9.

Usage:
    python run_pwae_ntm.py            # run1 (initial config)
    python run_pwae_ntm.py --run2     # run2 (post-fix config)
    python run_pwae_ntm.py --eval     # Phase 8 eval on best run
"""
from __future__ import annotations
import argparse, ast, json, math, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, label_ranking_average_precision_score
from sklearn.model_selection import train_test_split

ROOT = Path('/raid/home/nirajv/small_text')
sys.path.insert(0, str(ROOT))

from src.models.pwae_ntm_model import PWAENTMModel, PWAENet, compute_loss
from src.utils.word_embeddings import get_vocab_embeddings

# ─── Dataset ─────────────────────────────────────────────────────────────────
DATA_CSV   = ROOT / 'data' / 'reuters_10.csv'
LABELS     = ['interest','money-fx','trade','bop','crude','ship',
              'nat-gas','grain','oilseed','dlr']
SEED       = 42

# ─── Run configs ─────────────────────────────────────────────────────────────
RUN1 = dict(
    K=10, T=10, H=384, L=128, encoder_hidden=256,
    tau_multiplier=2.0,
    num_epochs=150, batch_size=200, lr=0.002, weight_decay=1e-5,
    free_bits=0.5, beta_kl_max=1.0, kl_warmup_fraction=0.3,
    lambda_ent=0.1, lambda_gate=0.05, lambda_div=0.1,
    load_balance=False,
    max_vocab=2000, min_df=2,
    word_emb_cache="cache/reuters10_vocab_embeddings_minilm.npy",
    embedding_model="all-MiniLM-L6-v2",
    random_state=SEED,
)

# Run2 params are filled in after Phase 6 diagnosis (starts as copy of run1)
RUN2 = dict(RUN1)  # will be updated after diagnostics

# Run3: corrected KL scaling — free_bits is nats-per-expert (not per-dim),
# beta_kl_max scaled down so KL ~= recon in magnitude.
RUN3 = dict(RUN1,
    free_bits=0.0,
    beta_kl_max=0.01,
    kl_warmup_fraction=0.4,
    lr=0.001,
    num_epochs=200,
    random_state=SEED,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────
def load_data():
    import re, csv
    def clean(t):
        t = t.lower()
        t = re.sub(r'[^a-z0-9\s]', ' ', t)
        return ' '.join(t.split())

    df = pd.read_csv(DATA_CSV)
    docs = [clean(str(t)) for t in df['text'].tolist()]
    Y = df[LABELS].values.astype(np.int32)
    return docs, Y


def make_split(docs, Y, test_ratio=0.2, seed=42):
    from collections import Counter
    idx = np.arange(len(docs))
    label_keys = ["".join(str(x) for x in row) for row in Y]
    key_counts = Counter(label_keys)
    rare_keys = {k for k, v in key_counts.items() if v < 2}
    if rare_keys:
        rare_mask = np.array([label_keys[i] in rare_keys for i in range(len(docs))])
        common_idx = idx[~rare_mask]
        rare_idx = idx[rare_mask]
        common_keys = [label_keys[i] for i in common_idx]
        common_train, common_test = train_test_split(
            common_idx, test_size=test_ratio, stratify=common_keys, random_state=seed)
        tr = np.sort(np.concatenate([common_train, rare_idx]))
        te = np.sort(common_test)
    else:
        tr, te = train_test_split(idx, test_size=test_ratio,
                                  stratify=label_keys, random_state=seed)
        tr, te = np.sort(tr), np.sort(te)
    return tr, te


# ─── Phase 5: Train ──────────────────────────────────────────────────────────
def train_run(run_params: dict, log_suffix: str, ckpt_suffix: str) -> PWAENTMModel:
    log_dir = ROOT / 'logs'
    ckpt_path = ROOT / 'checkpoints' / f'pwae_ntm_reuters10_{ckpt_suffix}.pt'
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    docs, Y = load_data()
    print(f"[Phase 5] Training {log_suffix} — {len(docs)} docs, {len(LABELS)} labels")

    model = PWAENTMModel(
        **run_params,
        log_dir=str(log_dir / log_suffix),
        checkpoint_path=str(ckpt_path),
    )
    # Redirect log CSV inside logs/ with suffix in filename
    import os
    os.makedirs(str(log_dir / log_suffix), exist_ok=True)

    model.fit(docs)
    model_path = ROOT / 'models' / f'pwae_ntm_reuters10_{ckpt_suffix}'
    model.save(str(model_path))
    print(f"[Phase 5] Model saved to {model_path}")
    return model, docs, Y


# ─── Phase 6: Diagnose ───────────────────────────────────────────────────────
def diagnose(log_suffix: str, model: PWAENTMModel) -> dict:
    log_dir = ROOT / 'logs' / log_suffix
    log_csv = log_dir / 'training_log.csv'

    print(f"\n{'='*60}")
    print(f"PHASE 6: Diagnostic checks — {log_suffix}")
    print(f"{'='*60}")

    log = pd.read_csv(log_csv)
    K = model.K
    L = model.L
    free_bits = model.free_bits
    checks = {}

    # Check 1 — KL collapse (use actual_kl_raw: real encoder KL without free_bits floor)
    # With free_bits applied as nats-per-expert, kl_raw can still be dominated by the floor.
    # actual_kl_raw measures whether the encoder is learning real posterior structure.
    actual_col = 'actual_kl_raw' if 'actual_kl_raw' in log.columns else 'kl_raw'
    final_kl_raw = log[actual_col].iloc[-1]
    # Expect encoder to use at least 1.0 nats per expert on average (K total minimum)
    threshold = K * 1.0
    c1 = final_kl_raw >= threshold
    checks['kl_collapse'] = c1
    checks['_final_kl_raw'] = final_kl_raw
    print(f"  Check 1 KL collapse:      {'PASS' if c1 else 'FAIL'}"
          f"  actual_kl={final_kl_raw:.2f}, threshold={threshold:.1f} ({K} experts × 1 nat)")

    # Check 2 — Expert load imbalance
    raw_load = ast.literal_eval(log['expert_load'].iloc[-1])
    min_load = min(raw_load)
    max_load = max(raw_load)
    argmin_e = raw_load.index(min_load)
    argmax_e = raw_load.index(max_load)
    c2a = min_load >= 0.02
    c2b = max_load <= 0.60
    c2 = c2a and c2b
    checks['expert_load'] = c2
    print(f"  Check 2 Expert load:      {'PASS' if c2 else 'FAIL'}"
          f"  min=E{argmin_e}:{min_load:.3f}  max=E{argmax_e}:{max_load:.3f}")

    # Check 3 — Attention entropy
    raw_ae = ast.literal_eval(log['attn_entropy_per_expert'].iloc[-1])
    min_ae = min(raw_ae)
    argmin_ae = raw_ae.index(min_ae)
    c3 = min_ae >= 0.5
    checks['attn_entropy'] = c3
    print(f"  Check 3 Attn entropy:     {'PASS' if c3 else 'FAIL'}"
          f"  min=E{argmin_ae}:{min_ae:.3f}")

    # Check 4 — Query vector collapse
    device = model.device
    queries = torch.stack([exp.query for exp in model.net.experts])  # (K, H)
    q_norm = F.normalize(queries.detach(), dim=-1)
    eye_mask = torch.eye(K, dtype=torch.bool, device=queries.device)
    off_diag_sim = (q_norm @ q_norm.T)[~eye_mask].mean().item()
    c4 = off_diag_sim <= 0.8
    checks['query_diversity'] = c4
    print(f"  Check 4 Query diversity:  {'PASS' if c4 else 'FAIL'}"
          f"  off_diag_sim={off_diag_sim:.3f}")

    # Check 5 — Reconstruction plateau
    recon_epoch10 = log['recon_loss'].iloc[9] if len(log) >= 10 else log['recon_loss'].iloc[0]
    recon_final   = log['recon_loss'].iloc[-1]
    c5 = recon_final < 0.95 * recon_epoch10
    checks['recon_plateau'] = c5
    print(f"  Check 5 Recon plateau:    {'PASS' if c5 else 'FAIL'}"
          f"  epoch10={recon_epoch10:.3f} → final={recon_final:.3f}")

    # Check 6 — Gate entropy
    final_gate_H = log['gate_entropy'].iloc[-1]
    max_gate_H   = math.log(K)
    c6 = final_gate_H >= 0.5 * max_gate_H
    checks['gate_entropy'] = c6
    tag = 'PASS' if c6 else 'WARN'
    print(f"  Check 6 Gate entropy:     {tag}"
          f"  gate_H={final_gate_H:.3f}, half_max={0.5*max_gate_H:.3f}")

    # Summary
    n_pass = sum(1 for v in checks.values() if v)
    print(f"\n  {n_pass}/6 checks passed")
    checks['_raw_load'] = raw_load
    checks['_final_kl_raw'] = final_kl_raw
    checks['_off_diag_sim'] = off_diag_sim
    checks['_recon_epoch10'] = recon_epoch10
    checks['_recon_final'] = recon_final
    checks['_gate_H'] = final_gate_H
    checks['_attn_ent'] = raw_ae
    return checks


# ─── Phase 7: Build fix config ───────────────────────────────────────────────
def build_run2_config(checks: dict, run1_params: dict) -> dict:
    run2 = dict(run1_params)
    fixes_applied = []

    if not checks['kl_collapse']:
        run2['free_bits'] = 1.0
        run2['beta_kl_max'] = 0.5
        run2['kl_warmup_fraction'] = 0.5
        fixes_applied.append("Fix C1: free_bits=1.0, beta_kl_max=0.5, kl_warmup=0.5")

    if not checks['expert_load']:
        run2['lambda_gate'] = 0.2
        run2['load_balance'] = True
        fixes_applied.append("Fix C2: lambda_gate=0.2, load_balance=True")

    if not checks['attn_entropy']:
        run2['tau_multiplier'] = 4.0
        run2['lambda_ent'] = 0.3
        fixes_applied.append("Fix C3: tau_multiplier=4.0, lambda_ent=0.3")

    if not checks['query_diversity']:
        run2['lambda_div'] = 0.5
        fixes_applied.append("Fix C4: lambda_div=0.5")

    if not checks['recon_plateau']:
        run2['lr'] = 0.001
        fixes_applied.append("Fix C5: lr=0.001")

    if fixes_applied:
        print("\n[Phase 7] Fixes applied for run2:")
        for f in fixes_applied:
            print(f"  {f}")
    else:
        print("\n[Phase 7] All checks passed — run2 uses same config as run1.")

    return run2


# ─── Phase 8: Evaluation ─────────────────────────────────────────────────────
def evaluate(model: PWAENTMModel, docs, Y, run_label: str):
    print(f"\n{'='*60}")
    print(f"PHASE 8: Evaluation — {run_label}")
    print(f"{'='*60}")

    train_idx, test_idx = make_split(docs, Y, test_ratio=0.2, seed=SEED)

    # Get theta for all docs
    all_theta = np.array(model.transform(docs))  # (N, T)
    theta_train = all_theta[train_idx]
    theta_test  = all_theta[test_idx]
    y_train = Y[train_idx]
    y_test  = Y[test_idx]

    # 8a — Linear probe
    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=SEED)
    )
    clf.fit(theta_train, y_train)
    y_pred      = clf.predict(theta_test)
    y_pred_prob = clf.predict_proba(theta_test)

    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_test, y_pred, average='micro', zero_division=0)
    lrap     = label_ranking_average_precision_score(y_test, y_pred_prob)

    print(f"\n  8a — Classification (OVR LogReg on theta):")
    print(f"  Macro-F1: {macro_f1:.4f}")
    print(f"  Micro-F1: {micro_f1:.4f}")
    print(f"  LRAP:     {lrap:.4f}")

    per_label_f1 = {lbl: round(float(f1_score(y_test[:, i], y_pred[:, i], zero_division=0)), 4)
                    for i, lbl in enumerate(LABELS)}
    print("  Per-label F1:")
    for lbl, f in per_label_f1.items():
        print(f"    {lbl:<15s}: {f:.4f}")

    # 8b — Top vocab per expert
    print(f"\n  8b — Expert vocabulary (attention-based vs beta-based):")
    vocab = list(model.vectorizer.get_feature_names_out())
    V = len(vocab)
    K = model.K
    T = model.T
    device = model.device
    model.net.eval()

    # Build BoW for all docs
    bow_full = model._build_bow(docs, fit=False)
    bow_t = torch.tensor(bow_full, dtype=torch.float32)

    # Accumulate mean attention per vocab word per expert
    attn_acc   = [np.zeros(V) for _ in range(K)]
    attn_count = [np.zeros(V) for _ in range(K)]

    batch_size = 256
    with torch.no_grad():
        E = model.net.word_emb.weight  # (V, H)
        for start in range(0, len(docs), batch_size):
            batch = bow_t[start:start+batch_size].to(device)
            mask = (batch > 0)
            M = int(mask.sum(dim=1).max().item())
            M = max(M, 1)
            topk_vals, topk_idx = torch.topk(mask.long(), k=M, dim=1)
            nz_valid = topk_vals.bool()

            for k, expert in enumerate(model.net.experts):
                c_k, theta_k, mu_k, ls_k, alpha_k = expert(batch, E, V)
                # alpha_k: (B, M), topk_idx: (B, M)
                b_sz = batch.shape[0]
                for b in range(b_sz):
                    valid_m = nz_valid[b].cpu().numpy()
                    idx_b   = topk_idx[b].cpu().numpy()[valid_m]
                    alp_b   = alpha_k[b].cpu().numpy()[valid_m]
                    attn_acc[k][idx_b]   += alp_b
                    attn_count[k][idx_b] += 1

    with torch.no_grad():
        beta = model.net.get_beta().cpu().numpy()  # (T, V)

    for k in range(K):
        mean_attn = attn_acc[k] / np.clip(attn_count[k], 1, None)
        top15_attn = [vocab[i] for i in np.argsort(-mean_attn)[:15]]
        top15_beta = [vocab[i] for i in np.argsort(-beta[k])[:15]]
        overlap = len(set(top15_attn) & set(top15_beta))
        print(f"  Expert {k}:")
        print(f"    Attn: {top15_attn}")
        print(f"    Beta: {top15_beta}")
        print(f"    Overlap: {overlap}/15")

    # 8c — Expert-to-class alignment
    print(f"\n  8c — Expert-to-class alignment:")
    # Get gate weights for all docs
    bow_all = model._build_bow(docs, fit=False)
    all_theta2, all_gates = model._encode_all(bow_all)  # (N,T), (N,K)
    g_test = all_gates[test_idx]

    unique_assignments = set()
    assignment_table = {}
    for c, lbl in enumerate(LABELS):
        doc_mask = y_test[:, c] == 1
        if doc_mask.sum() == 0:
            continue
        mean_g = g_test[doc_mask].mean(axis=0)  # (K,)
        dom_expert = int(mean_g.argmax())
        unique_assignments.add(dom_expert)
        assignment_table[lbl] = (dom_expert, float(mean_g[dom_expert]))
        print(f"    {lbl:<20s} → Expert {dom_expert:2d}  (gate={mean_g[dom_expert]:.3f})")

    print(f"\n  Unique expert assignments: {len(unique_assignments)} / {len(LABELS)}")
    collisions = len(LABELS) - len(unique_assignments)
    if collisions > 0:
        print(f"  WARNING: {collisions} class(es) share an expert assignment.")

    # 8d — Gate entropy distribution on test set
    print(f"\n  8d — Gate entropy distribution (test set):")
    g_H = -(g_test * np.log(g_test + 1e-10)).sum(axis=1)
    print(f"  mean={g_H.mean():.3f}  median={np.median(g_H):.3f}  "
          f"max_possible={math.log(K):.3f}")

    return {
        'macro_f1': macro_f1, 'micro_f1': micro_f1, 'lrap': lrap,
        'per_label_f1': per_label_f1,
        'unique_expert_assignments': len(unique_assignments),
        'assignment_table': assignment_table,
        'gate_H_mean': float(g_H.mean()),
        'gate_H_median': float(np.median(g_H)),
    }


# ─── Phase 9: Summary ────────────────────────────────────────────────────────
def print_summary(run_label: str, checks: dict, eval_results: dict, K: int):
    print(f"\n{'='*60}")
    print(f"PWAE-NTM on Reuters-10 — Final Results")
    print(f"{'='*60}")
    print(f"Run used: {run_label}")
    print()
    print("Diagnostic checks:")
    print(f"  Check 1 KL collapse:     {'PASS' if checks['kl_collapse'] else 'FAIL'}"
          f"  (kl_raw={checks['_final_kl_raw']:.2f})")
    rl = checks['_raw_load']
    print(f"  Check 2 Expert load:     {'PASS' if checks['expert_load'] else 'FAIL'}"
          f"  (min={min(rl):.3f}, max={max(rl):.3f})")
    print(f"  Check 3 Attn entropy:    {'PASS' if checks['attn_entropy'] else 'FAIL'}"
          f"  (min_expert={min(checks['_attn_ent']):.2f})")
    print(f"  Check 4 Query diversity: {'PASS' if checks['query_diversity'] else 'FAIL'}"
          f"  (off_diag_sim={checks['_off_diag_sim']:.3f})")
    print(f"  Check 5 Recon plateau:   {'PASS' if checks['recon_plateau'] else 'FAIL'}"
          f"  ({checks['_recon_epoch10']:.3f} → {checks['_recon_final']:.3f})")
    print(f"  Check 6 Gate entropy:    {'PASS' if checks['gate_entropy'] else 'WARN'}"
          f"  ({checks['_gate_H']:.3f})")
    print()
    print("Classification:")
    print(f"  Macro-F1: {eval_results['macro_f1']:.4f}")
    print(f"  Micro-F1: {eval_results['micro_f1']:.4f}")
    print(f"  LRAP:     {eval_results['lrap']:.4f}")
    print()
    print("Expert-class alignment:")
    print(f"  Unique expert assignments: {eval_results['unique_expert_assignments']} / {K}")
    for lbl, (expert, weight) in eval_results['assignment_table'].items():
        print(f"    {lbl:<20s} → Expert {expert:2d}  (gate={weight:.3f})")
    print()
    print(f"Gate entropy: mean={eval_results['gate_H_mean']:.3f}  "
          f"median={eval_results['gate_H_median']:.3f}  "
          f"max_possible={math.log(K):.3f}")
    print(f"{'='*60}")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run2",    action="store_true", help="run2 (post-fix config)")
    parser.add_argument("--run3",    action="store_true", help="run3 (KL-scaling fix)")
    parser.add_argument("--eval",    action="store_true", help="Phase 8 eval on best run")
    parser.add_argument("--skip-train", action="store_true", help="Skip training, load saved model")
    args = parser.parse_args()

    docs, Y = load_data()

    if args.eval:
        # Load best available run
        model_path = ROOT / 'models' / 'pwae_ntm_reuters10_run2'
        if not (model_path.parent / 'pwae_ntm_reuters10_run2_meta.json').exists():
            model_path = ROOT / 'models' / 'pwae_ntm_reuters10_run1'
        run_label = model_path.name
        model = PWAENTMModel(**RUN1)
        model.load(str(model_path))
        eval_results = evaluate(model, docs, Y, run_label)
        sys.exit(0)

    # ── Fast-path: run3 (KL-scaling fix) ─────────────────────────────────────
    if args.run3:
        model_run3, docs, Y = train_run(RUN3, "run3", "run3")
        checks_run3 = diagnose("run3", model_run3)
        eval_results = evaluate(model_run3, docs, Y, "run3")
        print_summary("run3", checks_run3, eval_results, K=model_run3.K)
        sys.exit(0)

    # ── Phase 5: run1 ─────────────────────────────────────────────────────────
    run1_ckpt = ROOT / 'models' / 'pwae_ntm_reuters10_run1_meta.json'
    if args.skip_train and run1_ckpt.exists():
        print("[Phase 5] Loading existing run1 model")
        model_run1 = PWAENTMModel(**RUN1)
        model_run1.fit(docs)  # need fit to build vectorizer for later eval
    else:
        model_run1, docs, Y = train_run(RUN1, "run1", "run1")

    # ── Phase 6: diagnose run1 ────────────────────────────────────────────────
    checks_run1 = diagnose("run1", model_run1)

    # ── Decide best run so far ────────────────────────────────────────────────
    best_model = model_run1
    best_label = "run1"
    best_checks = checks_run1

    if args.run2 or not all(checks_run1.get(k) for k in
                             ['kl_collapse','expert_load','attn_entropy',
                              'query_diversity','recon_plateau','gate_entropy']):
        # ── Phase 7: build run2 config ────────────────────────────────────────
        run2_params = build_run2_config(checks_run1, RUN1)

        # ── Phase 5 (run2): train with fixed config ───────────────────────────
        model_run2, docs, Y = train_run(run2_params, "run2", "run2")

        # ── Phase 6 (run2): diagnose ──────────────────────────────────────────
        checks_run2 = diagnose("run2", model_run2)

        # Choose best
        n1 = sum(1 for k,v in checks_run1.items() if not k.startswith('_') and v)
        n2 = sum(1 for k,v in checks_run2.items() if not k.startswith('_') and v)
        if n2 >= n1:
            best_model, best_label, best_checks = model_run2, "run2", checks_run2
            print(f"\n[Phase 7] run2 passes more checks ({n2} vs {n1}) → using run2")
        else:
            print(f"\n[Phase 7] run1 still better ({n1} vs {n2}) → keeping run1")
    else:
        print("\n[Phase 7] All checks passed in run1 — no retraining needed.")

    # ── Phase 8: Evaluate ─────────────────────────────────────────────────────
    eval_results = evaluate(best_model, docs, Y, best_label)

    # ── Phase 9: Summary ──────────────────────────────────────────────────────
    print_summary(best_label, best_checks, eval_results, K=best_model.K)
