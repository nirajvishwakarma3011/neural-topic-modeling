"""
Multi-seed, multi-dataset experiment runner for MoE-NTM thesis.

Runs 100 experiments:
  Tier 1: 5 models × 3 datasets × 5 seeds = 75 runs
  Tier 2: 5 models × 1 dataset (reuters_10) × 5 seeds = 25 runs

After each run: runs classify_multi.py and computes MoE diagnostics.
Logs all metrics to experiment_results_log.csv.
"""

import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GPU = "2"  # Tesla V100, most free at experiment start

# Local bin/ dir with java wrapper that reroutes Palmetto to /raid paths
LOCAL_BIN = str(Path(__file__).parent / "bin")

SEEDS = [42, 123, 456, 789, 1024]

DATASETS = {
    "reuters_10": {
        "data_cfg": "data_config/reuters_10.json",
        "results_dir": "results_reuters_10",
        "label_csv": "data/reuters_10.csv",
        "label_start_col": "interest",
        "n_docs": 2929,
    },
    "googlenews_10": {
        "data_cfg": "data_config/googlenews_10_mistral.json",
        "results_dir": "results_googlenews_10",
        "label_csv": "data/googlenewst_10_binary_labels.csv",
        "label_start_col": "China",
        "n_docs": 2414,
    },
    "20news_10": {
        "data_cfg": "data_config/20news_10.json",
        "results_dir": "results_20news_10",
        "label_csv": "data/20news_10_filtered.csv",
        "label_start_col": "comp.windows.x",
        "n_docs": 9571,
    },
}

BASE_CONFIGS = {
    "vae_gsm":           "model_config/vae_gsm_k10.json",
    "vae_gsm_use":       "model_config/vae_gsm_use_k10.json",
    "moe_ntm_ec":        "model_config/moe_ntm_ec_k10.json",
    "moe_ntm_use_ec":    "model_config/moe_ntm_use_ec_k10.json",
    "moe_ntm_use_sparse":"model_config/moe_ntm_use_sparse_k10.json",
    "moe_ntm":           "model_config/moe_ntm_k10.json",
    "moe_ntm_use":       "model_config/moe_ntm_use_k10.json",
    "moe_ntm_sparse":    "model_config/moe_ntm_sparse_k10.json",
    "moe_ntm_attn":      "model_config/moe_ntm_attn_k10.json",
    "moe_ntm_use_attn":  "model_config/moe_ntm_use_attn_k10.json",
}

TIER1_MODELS = ["vae_gsm", "vae_gsm_use", "moe_ntm_ec", "moe_ntm_use_ec", "moe_ntm_use_sparse"]
TIER2_MODELS = ["moe_ntm", "moe_ntm_use", "moe_ntm_sparse", "moe_ntm_attn", "moe_ntm_use_attn"]

LOG_CSV = "experiment_results_log.csv"
TMP_CFG_DIR = Path("tmp_configs_experiment")

LOG_HEADER = [
    "dataset", "model", "seed",
    "npmi", "cv", "topic_div",
    "rf_macro_f1", "rf_micro_f1", "hamming", "subset_acc",
    "num_collapsed", "spec_score", "label_cov", "gating_entropy",
    "per_label_f1_json",
    "run_dir", "wall_time_min", "status",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent


def log_print(msg: str):
    print(msg, flush=True)


def make_seed_config(model: str, seed: int) -> Path:
    """Create per-seed config with random_state=seed."""
    TMP_CFG_DIR.mkdir(parents=True, exist_ok=True)
    base_path = BASE_DIR / BASE_CONFIGS[model]
    cfg = json.loads(base_path.read_text())
    if "params" in cfg:
        cfg["params"]["random_state"] = seed
    out_path = TMP_CFG_DIR / f"{model}_seed{seed}.json"
    out_path.write_text(json.dumps(cfg, indent=2))
    return out_path


def run_main(dataset_cfg: str, method_cfg: str, seed: int, results_dir: str) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable, "main.py",
        "--dataset_cfg", dataset_cfg,
        "--method_cfg", str(method_cfg),
        "--seed", str(seed),
        "--results_dir", results_dir,
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = GPU
    env["PATH"] = LOCAL_BIN + ":" + env.get("PATH", "")
    return subprocess.run(
        cmd, capture_output=True, text=True, env=env, cwd=str(BASE_DIR)
    )


def find_latest_run(results_dir: str, method: str, dataset_name: str) -> Path | None:
    """Find most recently created run dir matching *_{method}_{dataset_name}."""
    res_path = BASE_DIR / results_dir
    if not res_path.exists():
        return None
    pattern = f"*_{method}_{dataset_name}"
    dirs = sorted(res_path.glob(pattern))
    return dirs[-1] if dirs else None


def run_classify(theta_path: Path, label_csv: str, label_start_col: str,
                 out_dir: Path, seed: int = 42, test_ratio: float = 0.2) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable, "classify_multi.py",
        "--theta", str(theta_path),
        "--csv", label_csv,
        "--label_start_col", label_start_col,
        "--seed", str(seed),
        "--test_ratio", str(test_ratio),
        "--text_col", "text",
        "--out_dir", str(out_dir),
    ]
    return subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(BASE_DIR)
    )


def load_metrics_json(run_dir: Path) -> dict:
    p = run_dir / "metrics.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def load_classify_json(run_dir: Path) -> dict:
    p = run_dir / "classification_multilabel_report.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def compute_moe_diagnostics(run_dir: Path) -> dict:
    """Compute num_collapsed, gating_entropy, spec_score from gate artifacts."""
    art_dir = run_dir / "artifacts"
    result = {
        "num_collapsed": None,
        "gating_entropy": None,
        "spec_score": None,
        "label_cov": None,
    }

    # EC models save distilled_gate; dense/sparse save gate_weights
    gate = None
    for fname in ("distilled_gate.npy", "gate_weights.npy"):
        p = art_dir / fname
        if p.exists():
            gate = np.load(str(p)).astype(np.float64)
            break

    if gate is None:
        return result  # VAE-GSM models have no gate

    N, E = gate.shape

    # Utilization: fraction of docs where each expert has highest gate weight
    top_expert = gate.argmax(axis=1)
    util = np.array([(top_expert == e).mean() for e in range(E)])

    # Collapsed: experts with utilization < 5%
    result["num_collapsed"] = int((util < 0.05).sum())

    # Gating entropy: mean -sum_e p_e log(p_e)
    eps = 1e-10
    entropy = -(gate * np.log(gate + eps)).sum(axis=1)
    result["gating_entropy"] = round(float(entropy.mean()), 4)

    # SpecScore: mean of max gate weight for documents routed to each expert
    # (= how pure each expert's routing is on average)
    spec_scores = []
    for e in range(E):
        mask = top_expert == e
        if mask.sum() > 0:
            spec_scores.append(float(gate[mask, e].mean()))
    result["spec_score"] = round(float(np.mean(spec_scores)), 4) if spec_scores else None

    return result


def extract_rf_metrics(classify_json: dict) -> dict:
    """Extract RF macro-F1, micro-F1, hamming, subset_acc, per_label from classify report."""
    out = {
        "rf_macro_f1": None, "rf_micro_f1": None,
        "hamming": None, "subset_acc": None,
        "per_label_f1_json": "{}",
    }
    if not classify_json:
        return out
    rf = classify_json.get("classifiers", {}).get("RF", {})
    if not rf:
        return out
    out["rf_macro_f1"] = rf.get("macro_f1")
    out["rf_micro_f1"] = rf.get("micro_f1")
    out["hamming"] = rf.get("hamming_loss")
    out["subset_acc"] = rf.get("subset_accuracy")
    per_label = {k: v["f1"] for k, v in rf.get("per_label", {}).items()}
    out["per_label_f1_json"] = json.dumps(per_label)
    return out


def append_log(row: dict):
    log_path = BASE_DIR / LOG_CSV
    write_header = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_HEADER)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in LOG_HEADER})


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_one(dataset: str, model: str, seed: int, run_num: int, total: int):
    ds = DATASETS[dataset]
    t0 = time.time()

    log_print(f"\n{'='*70}")
    log_print(f"Starting run {run_num}/{total}: dataset={dataset} model={model} seed={seed}")
    log_print(f"{'='*70}")

    # Create per-seed config
    try:
        cfg_path = make_seed_config(model, seed)
    except Exception as e:
        log_print(f"  ERROR creating config: {e}")
        append_log({"dataset": dataset, "model": model, "seed": seed,
                    "status": f"CONFIG_ERROR: {e}"})
        return

    # Ensure results dir exists
    results_dir = ds["results_dir"]
    (BASE_DIR / results_dir).mkdir(parents=True, exist_ok=True)

    # Run main.py
    log_print(f"  Running main.py ...")
    result = run_main(ds["data_cfg"], cfg_path, seed, results_dir)

    if result.returncode != 0:
        log_print(f"  FAILED (returncode={result.returncode})")
        log_print(f"  STDERR: {result.stderr[-2000:]}")
        wall = round((time.time() - t0) / 60, 2)
        append_log({"dataset": dataset, "model": model, "seed": seed,
                    "wall_time_min": wall,
                    "status": f"TRAINING_FAILED: {result.stderr[-300:]}"})
        return

    # Find the run directory (most recently created)
    # model name in config is the "name" field (e.g., "moe_ntm_ec")
    cfg = json.loads(Path(cfg_path).read_text())
    method_name = cfg["name"]
    dataset_name = json.loads(Path(ds["data_cfg"]).read_text())["name"]

    run_dir = find_latest_run(results_dir, method_name, dataset_name)
    if run_dir is None:
        log_print(f"  ERROR: could not find run dir after training")
        append_log({"dataset": dataset, "model": model, "seed": seed,
                    "status": "NO_RUN_DIR"})
        return

    log_print(f"  Run dir: {run_dir.name}")

    # Load training metrics
    metrics = load_metrics_json(run_dir)
    npmi = metrics.get("npmi_paper")
    cv = metrics.get("cv")
    topic_div = metrics.get("topic_diversity")

    # Verify doc_topic shape
    theta_path = run_dir / "artifacts" / "doc_topic.npy"
    if theta_path.exists():
        theta = np.load(str(theta_path))
        n_docs_actual = theta.shape[0]
        n_docs_expected = ds["n_docs"]
        if n_docs_actual != n_docs_expected:
            log_print(f"  WARNING: theta has {n_docs_actual} rows, expected {n_docs_expected}")
    else:
        log_print(f"  WARNING: doc_topic.npy not found")
        theta = None
        n_docs_actual = 0

    # Run classify_multi.py if theta exists
    rf_metrics = {"rf_macro_f1": None, "rf_micro_f1": None,
                  "hamming": None, "subset_acc": None, "per_label_f1_json": "{}"}
    if theta_path.exists() and n_docs_actual == ds["n_docs"]:
        log_print(f"  Running classify_multi.py ...")
        cl_result = run_classify(theta_path, ds["label_csv"], ds["label_start_col"],
                                  run_dir, seed=42, test_ratio=0.2)
        if cl_result.returncode != 0:
            log_print(f"  classify FAILED: {cl_result.stderr[-500:]}")
        else:
            classify_json = load_classify_json(run_dir)
            rf_metrics = extract_rf_metrics(classify_json)
            macro_f1 = rf_metrics.get("rf_macro_f1")
            log_print(f"  RF macro-F1: {macro_f1:.4f}" if macro_f1 else "  RF macro-F1: N/A")
    elif n_docs_actual != ds["n_docs"]:
        log_print(f"  Skipping classify: row count mismatch ({n_docs_actual} vs {ds['n_docs']})")

    # Compute MoE diagnostics
    moe_diag = compute_moe_diagnostics(run_dir)
    log_print(f"  NPMI: {npmi}  CV: {cv}  TD: {topic_div}")
    log_print(f"  Collapsed: {moe_diag['num_collapsed']}  "
              f"Entropy: {moe_diag['gating_entropy']}  "
              f"SpecScore: {moe_diag['spec_score']}")

    wall = round((time.time() - t0) / 60, 2)
    log_print(f"  Completed in {wall:.1f} min")

    # Append to log
    row = {
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "npmi": npmi,
        "cv": cv,
        "topic_div": topic_div,
        **rf_metrics,
        **moe_diag,
        "run_dir": str(run_dir),
        "wall_time_min": wall,
        "status": "OK",
    }
    append_log(row)


def build_experiment_list():
    """Build ordered list of (dataset, model, seed) tuples."""
    runs = []

    # Phase 1: Reuters-10, Tier 1, all seeds
    for model in TIER1_MODELS:
        for seed in SEEDS:
            runs.append(("reuters_10", model, seed))

    # Phase 2: 20news_10, Tier 1, all seeds
    for model in TIER1_MODELS:
        for seed in SEEDS:
            runs.append(("20news_10", model, seed))

    # Phase 3: googlenews_10, Tier 1, all seeds
    for model in TIER1_MODELS:
        for seed in SEEDS:
            runs.append(("googlenews_10", model, seed))

    # Phase 4: Reuters-10, Tier 2, all seeds
    for model in TIER2_MODELS:
        for seed in SEEDS:
            runs.append(("reuters_10", model, seed))

    return runs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    runs = build_experiment_list()
    total = len(runs)

    log_print(f"Starting {total} experiments on GPU {GPU}")
    log_print(f"Log file: {LOG_CSV}")
    log_print(f"Phases:")
    log_print(f"  Phase 1: Reuters-10, Tier-1 (25 runs)")
    log_print(f"  Phase 2: 20news_10, Tier-1 (25 runs)")
    log_print(f"  Phase 3: googlenews_10, Tier-1 (25 runs)")
    log_print(f"  Phase 4: Reuters-10, Tier-2 (25 runs)")

    total_t0 = time.time()

    for i, (dataset, model, seed) in enumerate(runs, 1):
        try:
            run_one(dataset, model, seed, i, total)
        except Exception as e:
            log_print(f"  UNHANDLED ERROR in run {i}: {e}")
            import traceback
            traceback.print_exc()
            append_log({"dataset": dataset, "model": model, "seed": seed,
                        "status": f"UNHANDLED_ERROR: {e}"})

    total_min = (time.time() - total_t0) / 60
    log_print(f"\nAll {total} experiments completed in {total_min:.1f} min")
    log_print(f"Results logged to: {LOG_CSV}")
