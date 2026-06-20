"""
Routing Specialization Analysis for MoE-NTM on Reuters-10.
All data hard-coded from:
  - moe_expert_comparison_report.txt  (SpecScore, LabelCov, Collapsed, per-expert affinities)
  - reuters_10_experiment_report_v2.txt Sec 3 (NPMI, CV) and Sec 5a (RF_macro_F1)
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings, sys, io

warnings.filterwarnings('ignore')

lines = []

def pr(s=''):
    lines.append(s)
    print(s)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Model-level data table
# ─────────────────────────────────────────────────────────────────────────────
# Sources:
#   SpecScore, LabelCov, Collapsed  → comparison report REUTERS_10 section
#   NPMI, CV                        → v2 report Section 3
#   RF_macro_F1                     → v2 report Section 5a (aggregate metrics table)
#   moe_ntm_ec: run 20260503_221646 (NPMI=0.2503); duplicate run 20260503_221723 (NPMI=N/A) excluded
#   Collapsed counts: from "Collapsed X/Y" in comparison report REUTERS_10 table
#   Attn models have E=10; Collapsed=1/10=1

model_rows = [
    # Model               SpecScore  LabelCov  NPMI    CV      RF_F1   Collapsed  IsVAE
    ('vae_gsm',           np.nan,    np.nan,   0.214,  0.469,  0.604,  np.nan,    True),
    ('vae_gsm_use',       np.nan,    np.nan,   0.222,  0.477,  0.691,  np.nan,    True),
    ('moe_ntm',           0.3795,    4,        0.216,  0.409,  0.636,  0,         False),
    ('moe_ntm_use',       0.4375,    4,        0.250,  0.458,  0.686,  2,         False),
    ('moe_ntm_sparse',    0.5063,    5,        0.224,  0.443,  0.650,  1,         False),
    ('moe_ntm_use_sparse',0.5228,    4,        0.256,  0.472,  0.690,  1,         False),
    ('moe_ntm_attn',      0.4186,    4,        0.209,  0.444,  0.620,  1,         False),
    ('moe_ntm_use_attn',  0.4088,    4,        0.213,  0.473,  0.610,  1,         False),
    ('moe_ntm_ec',        0.5298,    5,        0.250,  0.460,  0.724,  0,         False),
    ('moe_ntm_use_ec',    0.5132,    6,        0.218,  0.457,  0.638,  0,         False),
]

cols = ['Model','SpecScore','LabelCov','NPMI','CV','RF_macro_F1','Collapsed','IsVAE']
df = pd.DataFrame(model_rows, columns=cols)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Expert-level data
# ─────────────────────────────────────────────────────────────────────────────
# Labels order: interest, money-fx, trade, bop, crude, ship, nat-gas, grain, oilseed, dlr
# Collapsed experts (util < 5%) excluded as specified in task.
# Sources: comparison report "DETAILED PER-RUN EXPERT PROFILES" → REUTERS_10 section

LABELS = ['interest','money-fx','trade','bop','crude','ship','nat-gas','grain','oilseed','dlr']

def entropy(aff_vec):
    a = np.array(aff_vec, dtype=float)
    a = a / a.sum()
    a = a[a > 0]
    return float(-np.sum(a * np.log(a)))

expert_records = []

def add_experts(model, experts):
    for (idx, util, dom, aff) in experts:
        mx = max(aff)
        h  = entropy(aff)
        expert_records.append({
            'Model': model, 'Expert': idx, 'Util_pct': util,
            'DomLabel': dom, 'MaxAffinity': mx, 'LabelEntropy': h,
        })

# moe_ntm (Dense-BoW) — 0 collapsed, all 8 experts included
add_experts('moe_ntm', [
    (0, 17.2, 'trade',    [0.105,0.186,0.296,0.041,0.225,0.070,0.042,0.143,0.046,0.047]),
    (1,  6.5, 'money-fx', [0.434,0.485,0.101,0.053,0.067,0.025,0.013,0.068,0.022,0.068]),
    (2, 11.6, 'grain',    [0.142,0.152,0.051,0.016,0.179,0.117,0.026,0.415,0.124,0.033]),
    (3, 16.9, 'money-fx', [0.216,0.362,0.170,0.024,0.090,0.100,0.015,0.152,0.046,0.106]),
    (4, 12.7, 'trade',    [0.212,0.200,0.224,0.108,0.224,0.039,0.058,0.118,0.040,0.029]),
    (5,  9.6, 'grain',    [0.031,0.045,0.054,0.006,0.366,0.111,0.073,0.378,0.103,0.010]),
    (6,  8.9, 'money-fx', [0.250,0.585,0.157,0.012,0.048,0.046,0.006,0.049,0.014,0.177]),
    (7, 16.6, 'crude',    [0.056,0.133,0.182,0.011,0.290,0.207,0.040,0.210,0.055,0.037]),
])

# moe_ntm_use (Dense-SBERT) — E4 (3.6%) and E7 (4.9%) collapsed; 6 experts included
add_experts('moe_ntm_use', [
    (0, 16.2, 'money-fx', [0.361,0.474,0.143,0.061,0.047,0.022,0.010,0.071,0.022,0.129]),
    (1, 14.3, 'money-fx', [0.317,0.469,0.193,0.081,0.020,0.016,0.003,0.125,0.036,0.097]),
    (2, 16.5, 'trade',    [0.121,0.192,0.288,0.018,0.213,0.090,0.037,0.140,0.044,0.045]),
    (3, 15.1, 'crude',    [0.038,0.041,0.065,0.007,0.382,0.150,0.098,0.312,0.098,0.008]),
    (5, 18.0, 'grain',    [0.107,0.134,0.170,0.039,0.147,0.122,0.034,0.341,0.093,0.031]),
    (6, 11.4, 'crude',    [0.084,0.088,0.091,0.005,0.534,0.236,0.062,0.068,0.024,0.020]),
])

# moe_ntm_sparse (Sparse-BoW) — E6 (0.0%) collapsed; 7 experts included
add_experts('moe_ntm_sparse', [
    (0, 18.7, 'trade',    [0.079,0.402,0.432,0.019,0.038,0.028,0.003,0.082,0.026,0.123]),
    (1,  6.5, 'grain',    [0.029,0.029,0.021,0.004,0.422,0.041,0.058,0.439,0.129,0.004]),
    (2, 14.3, 'interest', [0.294,0.244,0.084,0.055,0.100,0.172,0.037,0.224,0.084,0.015]),
    (3, 14.2, 'grain',    [0.005,0.009,0.100,0.003,0.039,0.030,0.004,0.762,0.191,0.001]),
    (4, 20.7, 'money-fx', [0.417,0.601,0.116,0.074,0.003,0.004,0.000,0.004,0.001,0.163]),
    (5,  9.1, 'crude',    [0.053,0.071,0.272,0.068,0.424,0.065,0.094,0.105,0.033,0.014]),
    (7, 16.6, 'crude',    [0.062,0.019,0.092,0.000,0.555,0.316,0.087,0.042,0.015,0.000]),
])

# moe_ntm_use_sparse (Sparse-SBERT) — E2 (0.0%) collapsed; 7 experts included
add_experts('moe_ntm_use_sparse', [
    (0, 10.7, 'grain',    [0.008,0.032,0.376,0.068,0.103,0.045,0.023,0.407,0.094,0.007]),
    (1, 18.3, 'interest', [0.561,0.507,0.034,0.025,0.002,0.003,0.000,0.107,0.032,0.080]),
    (3, 19.5, 'crude',    [0.005,0.001,0.011,0.001,0.575,0.261,0.134,0.139,0.055,0.000]),
    (4, 13.7, 'money-fx', [0.325,0.477,0.231,0.146,0.013,0.007,0.002,0.016,0.005,0.159]),
    (5, 13.3, 'grain',    [0.064,0.087,0.014,0.002,0.040,0.072,0.008,0.731,0.211,0.000]),
    (6, 11.1, 'crude',    [0.043,0.080,0.216,0.003,0.533,0.230,0.051,0.017,0.013,0.012]),
    (7, 13.5, 'money-fx', [0.097,0.543,0.368,0.006,0.019,0.015,0.001,0.022,0.013,0.171]),
])

# moe_ntm_attn (Attn-BoW, E=10) — E4 (2.4%) collapsed; 9 experts included
add_experts('moe_ntm_attn', [
    (0, 13.8, 'grain',    [0.049,0.093,0.208,0.014,0.237,0.195,0.030,0.274,0.077,0.023]),
    (1,  9.9, 'crude',    [0.101,0.125,0.116,0.013,0.402,0.173,0.068,0.137,0.042,0.029]),
    (2, 14.9, 'crude',    [0.201,0.206,0.130,0.056,0.244,0.084,0.062,0.176,0.057,0.035]),
    (3, 11.0, 'money-fx', [0.233,0.516,0.212,0.047,0.047,0.024,0.007,0.066,0.020,0.158]),
    (5, 10.4, 'crude',    [0.199,0.199,0.079,0.016,0.393,0.104,0.078,0.081,0.026,0.047]),
    (6,  9.1, 'money-fx', [0.400,0.498,0.121,0.082,0.037,0.017,0.009,0.088,0.028,0.091]),
    (7,  5.8, 'money-fx', [0.323,0.597,0.086,0.013,0.073,0.025,0.012,0.029,0.010,0.190]),
    (8, 10.6, 'grain',    [0.050,0.077,0.150,0.032,0.102,0.107,0.016,0.517,0.143,0.014]),
    (9, 12.1, 'trade',    [0.070,0.211,0.367,0.035,0.061,0.060,0.007,0.263,0.074,0.059]),
])

# moe_ntm_use_attn (Attn-SBERT, E=10) — E2 (4.4%) collapsed; 9 experts included
add_experts('moe_ntm_use_attn', [
    (0, 13.7, 'trade',    [0.124,0.184,0.277,0.052,0.242,0.057,0.050,0.138,0.047,0.045]),
    (1, 14.4, 'money-fx', [0.297,0.358,0.112,0.023,0.106,0.091,0.022,0.123,0.038,0.108]),
    (3,  8.2, 'crude',    [0.045,0.124,0.235,0.008,0.329,0.242,0.038,0.127,0.042,0.035]),
    (4,  5.9, 'money-fx', [0.404,0.502,0.142,0.081,0.051,0.017,0.010,0.047,0.015,0.094]),
    (5, 13.8, 'grain',    [0.082,0.084,0.071,0.012,0.227,0.172,0.050,0.386,0.106,0.017]),
    (6, 10.5, 'trade',    [0.099,0.355,0.383,0.022,0.126,0.040,0.018,0.056,0.021,0.104]),
    (7,  6.3, 'grain',    [0.066,0.076,0.059,0.017,0.187,0.081,0.038,0.528,0.153,0.009]),
    (8, 14.7, 'crude',    [0.023,0.032,0.076,0.007,0.413,0.142,0.073,0.320,0.084,0.007]),
    (9,  8.0, 'money-fx', [0.279,0.330,0.204,0.114,0.124,0.031,0.028,0.120,0.039,0.033]),
])

# moe_ntm_ec (EC-BoW) — 0 collapsed, all 8 experts included
add_experts('moe_ntm_ec', [
    (0, 11.1, 'crude',    [0.001,0.003,0.020,0.004,0.621,0.284,0.136,0.095,0.035,0.000]),
    (1, 13.3, 'crude',    [0.068,0.068,0.270,0.144,0.449,0.046,0.131,0.016,0.006,0.005]),
    (2, 13.4, 'grain',    [0.045,0.181,0.051,0.015,0.212,0.022,0.008,0.417,0.164,0.071]),
    (3, 10.5, 'money-fx', [0.449,0.622,0.086,0.061,0.008,0.003,0.002,0.020,0.010,0.208]),
    (4, 11.4, 'money-fx', [0.131,0.562,0.305,0.038,0.065,0.008,0.008,0.013,0.001,0.160]),
    (5, 12.5, 'grain',    [0.001,0.002,0.086,0.001,0.160,0.328,0.003,0.536,0.110,0.000]),
    (6, 16.0, 'interest', [0.514,0.396,0.031,0.005,0.013,0.032,0.000,0.198,0.069,0.017]),
    (7, 11.8, 'trade',    [0.027,0.125,0.518,0.005,0.063,0.071,0.003,0.257,0.056,0.036]),
])

# moe_ntm_use_ec (EC-SBERT) — 0 collapsed, all 8 experts included
add_experts('moe_ntm_use_ec', [
    (0,  9.9, 'interest', [0.592,0.592,0.025,0.019,0.004,0.002,0.001,0.003,0.002,0.170]),
    (1, 15.9, 'crude',    [0.158,0.203,0.088,0.015,0.391,0.162,0.046,0.050,0.020,0.023]),
    (2,  9.3, 'money-fx', [0.426,0.686,0.046,0.007,0.002,0.001,0.000,0.002,0.001,0.249]),
    (3, 12.1, 'trade',    [0.127,0.144,0.362,0.191,0.041,0.027,0.009,0.276,0.095,0.026]),
    (4, 12.2, 'ship',     [0.036,0.266,0.282,0.003,0.228,0.305,0.010,0.049,0.017,0.072]),
    (5, 12.2, 'grain',    [0.025,0.040,0.067,0.040,0.072,0.098,0.019,0.661,0.194,0.000]),
    (6, 15.3, 'crude',    [0.091,0.161,0.008,0.003,0.645,0.106,0.167,0.030,0.007,0.001]),
    (7, 13.2, 'grain',    [0.015,0.055,0.408,0.004,0.017,0.030,0.003,0.463,0.117,0.014]),
])

df_exp = pd.DataFrame(expert_records)

# ─────────────────────────────────────────────────────────────────────────────
# Helper: Spearman correlation with interpretation
# ─────────────────────────────────────────────────────────────────────────────
def spearman_result(x, y, xname, yname, n_label=None):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    rho, pval = stats.spearmanr(x, y)
    sig = '**' if pval < 0.01 else ('*' if pval < 0.05 else 'ns')
    label = n_label if n_label else f'n={n}'
    interp = ('positive' if rho > 0 else 'negative') + ', ' + \
             ('significant' if pval < 0.05 else 'non-significant')
    return rho, pval, n, sig, interp

# =============================================================================
# REPORT OUTPUT
# =============================================================================
pr("=" * 78)
pr("ROUTING SPECIALIZATION ANALYSIS: MoE-NTM on Reuters-10")
pr("Dataset : Reuters-10 (reuters_10.csv)  N=2,929  Labels=10  Seed=42")
pr("Date    : 2026-05-10")
pr("Sources : moe_expert_comparison_report.txt,")
pr("          reuters_10_experiment_report_v2.txt (Sections 3 & 5a)")
pr("=" * 78)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 TABLE
# ─────────────────────────────────────────────────────────────────────────────
pr()
pr("─" * 78)
pr("SECTION 1: MODEL-LEVEL DATA TABLE (n=10)")
pr("─" * 78)
pr("SpecScore = mean max soft-affinity across experts (N/A for VAE baselines)")
pr("LabelCov  = distinct dominant labels covered by experts (N/A for VAE baselines)")
pr("Collapsed = count of experts with util < 5% (N/A for VAE baselines)")
pr("RF_macro_F1 = Random Forest macro-F1 on doc_topic [N×K] features")
pr()

hdr = f"{'Model':<24} {'SpecScore':>9} {'LabelCov':>9} {'NPMI':>6} {'CV':>6} {'RF_F1':>7} {'Collapsed':>10}"
pr(hdr)
pr("─" * 78)
for _, r in df.iterrows():
    sp  = f"{r.SpecScore:.4f}" if not np.isnan(r.SpecScore) else "N/A"
    lc  = f"{int(r.LabelCov)}/10"  if not np.isnan(r.LabelCov) else "N/A"
    col = f"{int(r.Collapsed)}"    if not np.isnan(r.Collapsed) else "N/A"
    pr(f"{r.Model:<24} {sp:>9} {lc:>9} {r.NPMI:>6.3f} {r.CV:>6.3f} {r.RF_macro_F1:>7.3f} {col:>10}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Spearman correlations — MoE models only (n=8)
# ─────────────────────────────────────────────────────────────────────────────
pr()
pr("─" * 78)
pr("SECTION 2: SPEARMAN RANK CORRELATIONS — MoE MODELS ONLY (n=8)")
pr("─" * 78)
pr("Exclude VAE baselines; they have no routing layer (SpecScore, LabelCov, Collapsed = N/A).")
pr()

moe = df[~df.IsVAE].copy()

corrs_moe = [
    ('SpecScore',   'RF_macro_F1', 'Routing specialization → classification'),
    ('NPMI',        'RF_macro_F1', 'Topic coherence (NPMI)  → classification'),
    ('CV',          'RF_macro_F1', 'Topic coherence (CV)    → classification'),
    ('LabelCov',    'RF_macro_F1', 'Label coverage          → classification'),
    ('Collapsed',   'RF_macro_F1', 'Expert collapse         → classification (expect −)'),
]

pr(f"{'Predictor':<14} {'Target':<14} {'rho':>7} {'p-value':>9} {'n':>4} {'sig':>4}  Interpretation")
pr("─" * 78)
saved_rho = {}
for xn, yn, desc in corrs_moe:
    rho, pval, n, sig, interp = spearman_result(
        moe[xn].values, moe[yn].values, xn, yn)
    saved_rho[xn] = (rho, pval, n, sig, interp)
    pr(f"{xn:<14} {yn:<14} {rho:>7.4f} {pval:>9.4f} {n:>4} {sig:>4}  {interp}")

pr()
pr("Significance: ** p<0.01, * p<0.05, ns = not significant")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Coherence correlations including VAE baselines (n=10)
# ─────────────────────────────────────────────────────────────────────────────
pr()
pr("─" * 78)
pr("SECTION 3: COHERENCE CORRELATIONS — ALL MODELS INCLUDING VAE BASELINES (n=10)")
pr("─" * 78)
pr("Tests whether topic coherence predicts classification across all model families.")
pr()

pr(f"{'Predictor':<14} {'Target':<14} {'rho':>7} {'p-value':>9} {'n':>4} {'sig':>4}  Interpretation")
pr("─" * 78)
for xn in ['NPMI', 'CV']:
    rho, pval, n, sig, interp = spearman_result(
        df[xn].values, df['RF_macro_F1'].values, xn, 'RF_macro_F1')
    saved_rho[f'{xn}_all'] = (rho, pval, n, sig, interp)
    pr(f"{xn:<14} {'RF_macro_F1':<14} {rho:>7.4f} {pval:>9.4f} {n:>4} {sig:>4}  {interp}")

pr()
pr("Significance: ** p<0.01, * p<0.05, ns = not significant")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Expert-level analysis
# ─────────────────────────────────────────────────────────────────────────────
pr()
pr("─" * 78)
pr("SECTION 4: EXPERT-LEVEL ANALYSIS TABLE (non-collapsed experts only)")
pr("─" * 78)
pr(f"Total non-collapsed experts: {len(df_exp)} across {df_exp.Model.nunique()} models")
pr()

# Print table
pr(f"{'Model':<24} {'Exp':>4} {'Util%':>6} {'DomLabel':<12} {'MaxAff':>7} {'H(label)':>9}")
pr("─" * 78)
for _, r in df_exp.iterrows():
    pr(f"{r.Model:<24} {r.Expert:>4} {r.Util_pct:>6.1f} {r.DomLabel:<12} {r.MaxAffinity:>7.4f} {r.LabelEntropy:>9.4f}")

pr()
pr("Spearman correlations at expert level:")
pr()

pr(f"{'Predictor':<20} {'Target':<20} {'rho':>7} {'p-value':>9} {'n':>4} {'sig':>4}  Interpretation")
pr("─" * 78)

# Correlation 1: MaxAffinity vs Utilization
rho1, pval1, n1, sig1, interp1 = spearman_result(
    df_exp['MaxAffinity'].values, df_exp['Util_pct'].values,
    'MaxAffinity', 'Util_pct')
saved_rho['exp_aff_util'] = (rho1, pval1, n1)
pr(f"{'MaxAffinity':<20} {'Util_pct':<20} {rho1:>7.4f} {pval1:>9.4f} {n1:>4} {sig1:>4}  {interp1}")

# Correlation 2: LabelEntropy vs MaxAffinity (expect negative)
rho2, pval2, n2, sig2, interp2 = spearman_result(
    df_exp['LabelEntropy'].values, df_exp['MaxAffinity'].values,
    'LabelEntropy', 'MaxAffinity')
saved_rho['exp_ent_aff'] = (rho2, pval2, n2)
pr(f"{'LabelEntropy':<20} {'MaxAffinity':<20} {rho2:>7.4f} {pval2:>9.4f} {n2:>4} {sig2:>4}  {interp2}")

pr()
pr("Note: LabelEntropy vs MaxAffinity expected negative — purer experts (lower entropy)")
pr("      should have higher max affinity for a single label.")
pr()
pr("Significance: ** p<0.01, * p<0.05, ns = not significant")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Head-to-head comparison
# ─────────────────────────────────────────────────────────────────────────────
pr()
pr("─" * 78)
pr("SECTION 5: HEAD-TO-HEAD — SpecScore vs NPMI vs CV as predictors of RF macro-F1")
pr("─" * 78)
pr("Scope: MoE models only (n=8, VAE baselines excluded)")
pr()

rho_spec, pval_spec, n_spec, sig_spec, _ = saved_rho['SpecScore']
rho_npmi, pval_npmi, n_npmi, sig_npmi, _ = saved_rho['NPMI']
rho_cv,   pval_cv,   n_cv,   sig_cv,   _ = saved_rho['CV']

pr(f"  SpecScore vs RF_macro_F1 : rho = {rho_spec:+.4f}  p = {pval_spec:.4f}  ({sig_spec})")
pr(f"  NPMI      vs RF_macro_F1 : rho = {rho_npmi:+.4f}  p = {pval_npmi:.4f}  ({sig_npmi})")
pr(f"  CV        vs RF_macro_F1 : rho = {rho_cv:+.4f}  p = {pval_cv:.4f}  ({sig_cv})")
pr()

# Determine stronger predictor
def stronger(rho1, rho2, name1, name2):
    if abs(rho1) > abs(rho2):
        return f"{name1} is the stronger predictor (|rho|={abs(rho1):.4f} vs {abs(rho2):.4f}, Δ={abs(rho1)-abs(rho2):.4f})"
    else:
        return f"{name2} is the stronger predictor (|rho|={abs(rho2):.4f} vs {abs(rho1):.4f}, Δ={abs(rho2)-abs(rho1):.4f})"

pr("SpecScore vs NPMI:")
pr("  " + stronger(rho_spec, rho_npmi, "SpecScore", "NPMI"))
pr()
pr("SpecScore vs CV:")
pr("  " + stronger(rho_spec, rho_cv, "SpecScore", "CV"))
pr()

# Full n=10 coherence rhos for comparison
rho_npmi_all, pval_npmi_all, n_npmi_all, sig_npmi_all, _ = saved_rho['NPMI_all']
rho_cv_all,   pval_cv_all,   n_cv_all,   sig_cv_all,   _ = saved_rho['CV_all']
pr("For reference — coherence correlations when VAE baselines included (n=10):")
pr(f"  NPMI vs RF_macro_F1 (n=10): rho = {rho_npmi_all:+.4f}  p = {pval_npmi_all:.4f}  ({sig_npmi_all})")
pr(f"  CV   vs RF_macro_F1 (n=10): rho = {rho_cv_all:+.4f}  p = {pval_cv_all:.4f}  ({sig_cv_all})")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY SECTION
# ─────────────────────────────────────────────────────────────────────────────
pr()
pr("=" * 78)
pr("SUMMARY AND THESIS INTERPRETATION")
pr("=" * 78)
pr()
pr("(a) Does routing specialization predict classification better than topic coherence?")
pr()

if abs(rho_spec) > abs(rho_npmi):
    verb_npmi = "yes — SpecScore outperforms NPMI"
else:
    verb_npmi = "no — NPMI outperforms SpecScore"
if abs(rho_spec) > abs(rho_cv):
    verb_cv = "yes — SpecScore outperforms CV"
else:
    verb_cv = "no — CV outperforms SpecScore"

pr(f"    vs NPMI: {verb_npmi}")
pr(f"             SpecScore rho={rho_spec:+.4f} ({sig_spec}), NPMI rho={rho_npmi:+.4f} ({sig_npmi}), n=8")
pr(f"    vs CV:   {verb_cv}")
pr(f"             SpecScore rho={rho_spec:+.4f} ({sig_spec}), CV rho={rho_cv:+.4f} ({sig_cv}), n=8")
pr()
pr("(b) Strength and significance of each relationship (MoE models, n=8):")
pr()
summary_rows = [
    ("SpecScore → RF_macro_F1",  rho_spec,     pval_spec,  8,  sig_spec),
    ("NPMI → RF_macro_F1",       rho_npmi,     pval_npmi,  8,  sig_npmi),
    ("CV → RF_macro_F1",         rho_cv,       pval_cv,    8,  sig_cv),
    ("LabelCov → RF_macro_F1",   *saved_rho['LabelCov'][:2], 8,  saved_rho['LabelCov'][3]),
    ("Collapsed → RF_macro_F1",  *saved_rho['Collapsed'][:2], 8, saved_rho['Collapsed'][3]),
    ("NPMI → RF_macro_F1 (n=10)",rho_npmi_all, pval_npmi_all, 10, sig_npmi_all),
    ("CV → RF_macro_F1 (n=10)",  rho_cv_all,   pval_cv_all,   10, sig_cv_all),
    ("Expert MaxAff → Util",     *saved_rho['exp_aff_util'][:2], n1, sig1),
    ("Expert H(label) → MaxAff", *saved_rho['exp_ent_aff'][:2], n2, sig2),
]
pr(f"  {'Relationship':<35} {'rho':>7} {'p-value':>9} {'n':>4} {'sig':>4}")
pr("  " + "─" * 62)
for row in summary_rows:
    pr(f"  {row[0]:<35} {row[1]:>7.4f} {row[2]:>9.4f} {row[3]:>4} {row[4]:>4}")
pr()

pr("(c) Caveats and limitations:")
pr()
pr("  1. Sample size: n=8 MoE models. Spearman rho is sensitive to ties and")
pr("     outliers at small n. Results should be interpreted as indicative, not")
pr("     definitive. Replication across additional datasets required.")
pr()
pr("  2. Single dataset and seed: all results from Reuters-10 (N=2,929), seed=42.")
pr("     Ranking order among models may differ on MPST or other datasets.")
pr()
pr("  3. Single metric conflation: RF macro-F1 is the sole downstream measure.")
pr("     micro-F1, subset accuracy, and per-label F1 tell a different story")
pr("     (moe_ntm_ec's macro gain is driven primarily by nat-gas and bop tail labels).")
pr()
pr("  4. SpecScore confound: SpecScore is the mean max soft-affinity across experts,")
pr("     which conflates routing sharpness with label alignment. Models with sharp")
pr("     routing (EC, Sparse) will mechanically score higher on SpecScore regardless")
pr("     of whether specialization is semantically meaningful.")
pr()
pr("  5. Expert-level n: 62 non-collapsed experts across 8 models. Experts are not")
pr("     independent — within-model correlations inflate effective sample size.")
pr()
pr("─" * 78)
pr("THESIS STATEMENT (if rho_spec > rho_npmi and SpecScore is significant):")
pr()
if abs(rho_spec) > abs(rho_npmi) and pval_spec < 0.05:
    pr("  Across 8 MoE-NTM routing variants on Reuters-10, expert routing specialization")
    pr(f"  (SpecScore, rho={rho_spec:.3f}, p={pval_spec:.3f}) is a stronger predictor of downstream")
    pr(f"  multilabel classification performance (RF macro-F1) than topic coherence")
    pr(f"  (NPMI, rho={rho_npmi:.3f}, p={pval_npmi:.3f}). Routing variants that achieve higher")
    pr("  mean max soft-affinity between experts and gold labels tend to also achieve")
    pr("  better classification, suggesting that expert specialization quality is more")
    pr("  diagnostic of downstream utility than word-level topic coherence.")
elif abs(rho_spec) > abs(rho_npmi):
    pr("  SpecScore shows higher |rho| than NPMI but neither is statistically")
    pr(f"  significant at p<0.05 (SpecScore: rho={rho_spec:.3f}, p={pval_spec:.3f}; n=8).")
    pr("  The direction supports the specialization hypothesis, but the effect size")
    pr("  should be treated as preliminary given the small sample.")
else:
    pr(f"  SpecScore (rho={rho_spec:.3f}) does not outperform NPMI (rho={rho_npmi:.3f}) as a")
    pr("  predictor of RF macro-F1 at n=8. No strong evidence that routing specialization")
    pr("  is a better predictor than topic coherence on this dataset.")
pr()
pr("─" * 78)
pr("End of routing_specialization_analysis.txt")
pr("─" * 78)

# ─────────────────────────────────────────────────────────────────────────────
# Write to file
# ─────────────────────────────────────────────────────────────────────────────
out_path = '/raid/home/nirajv/small_text/routing_specialization_analysis.txt'
with open(out_path, 'w') as f:
    f.write('\n'.join(lines) + '\n')
print(f"\n[Saved to {out_path}]")
