"""
Evaluate topic model θ as features for multi-label classification.

Each document can belong to multiple categories simultaneously
(e.g. Computer Science AND Mathematics). Labels are binary columns
in the CSV (1 = belongs, 0 = does not).

Usage:
    python analysis_ui/classify_multilabel.py \
        --theta artifacts/doc_topic.npy \
        --csv   data/arxiv_abstracts.csv \
        --label_cols "Computer Science" "Physics" "Mathematics" "Statistics" \
                     "Quantitative Biology" "Quantitative Finance"

    # If label columns are all columns after a known cutoff:
    python analysis_ui/classify_multilabel.py \
        --theta artifacts/doc_topic.npy \
        --csv   data/arxiv_abstracts.csv \
        --label_start_col "Computer Science"

    # Custom split:
    python analysis_ui/classify_multilabel.py \
        --theta artifacts/doc_topic.npy \
        --csv   data/arxiv_abstracts.csv \
        --label_start_col "Computer Science" \
        --test_ratio 0.2 --seed 42

Output:
    classification_multilabel_report.json  — machine-readable, per-label + macro
    classification_multilabel_report.txt   — human-readable
    stdout: summary tables
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_data(theta_path: str, csv_path: str,
              label_cols: list[str] | None,
              label_start_col: str | None,
              text_col: str = "ABSTRACT") -> tuple[np.ndarray, np.ndarray, list[str], list[str] | None]:
    """
    Returns: (theta, Y, label_names, texts_or_None)
      theta: [N, K] float
      Y:     [N, C] binary int
      label_names: list of C label column names
      texts: list of N strings (for error inspection), or None
    """
    theta = np.load(theta_path)
    df = pd.read_csv(csv_path)

    # Determine label columns
    if label_cols:
        label_names = label_cols
    elif label_start_col:
        col_list = list(df.columns)
        start_idx = col_list.index(label_start_col)
        label_names = col_list[start_idx:]
    else:
        raise ValueError("Provide either --label_cols or --label_start_col")

    # Validate
    for lc in label_names:
        if lc not in df.columns:
            raise ValueError(f"Label column '{lc}' not found in CSV. "
                             f"Available: {list(df.columns)}")

    Y = df[label_names].values.astype(np.int32)
    texts = list(df[text_col].astype(str)) if text_col in df.columns else None

    # Row alignment check
    if theta.shape[0] != Y.shape[0]:
        raise ValueError(
            f"Row count mismatch: theta has {theta.shape[0]} rows, "
            f"CSV has {Y.shape[0]} rows. They must be row-aligned."
        )

    print(f"Loaded: theta {theta.shape}, labels {Y.shape} ({len(label_names)} categories)")
    print(f"Label columns: {label_names}")

    # Label distribution
    print("\nLabel distribution:")
    for i, name in enumerate(label_names):
        pos = int(Y[:, i].sum())
        print(f"  {name:<25s}  {pos:>6d} / {Y.shape[0]}  ({100*pos/Y.shape[0]:.1f}%)")

    # Multi-label stats
    labels_per_doc = Y.sum(axis=1)
    print(f"\nLabels per doc: mean={labels_per_doc.mean():.2f}  "
          f"median={np.median(labels_per_doc):.1f}  "
          f"max={labels_per_doc.max()}  "
          f"single-label={int((labels_per_doc == 1).sum())} "
          f"({100*(labels_per_doc == 1).sum()/len(labels_per_doc):.1f}%)")

    return theta, Y, label_names, texts


def split_data(theta: np.ndarray, Y: np.ndarray,
               test_ratio: float, seed: int):
    """
    Stratified multi-label split. Uses iterative stratification if available,
    falls back to random split with label-frequency-based validation.
    """
    from sklearn.model_selection import train_test_split

    n = theta.shape[0]
    idx = np.arange(n)

    # For multi-label, stratify on the most frequent label to get roughly
    # balanced splits. Not perfect, but good enough without extra dependencies.
    # Use the label combination as a string key for stratification.
    label_keys = ["".join(str(x) for x in row) for row in Y]

    # If too many unique combinations for stratification, fall back to random
    unique_keys = set(label_keys)
    if len(unique_keys) > n * 0.5:
        print("  (too many label combinations for stratification, using random split)")
        train_idx, test_idx = train_test_split(
            idx, test_size=test_ratio, random_state=seed
        )
    else:
        # Filter out label combos with < 2 samples (can't stratify those)
        from collections import Counter
        key_counts = Counter(label_keys)
        rare_keys = {k for k, v in key_counts.items() if v < 2}

        if rare_keys:
            # Move rare-combo docs to train, stratify the rest
            rare_mask = np.array([label_keys[i] in rare_keys for i in range(n)])
            common_idx = idx[~rare_mask]
            rare_idx = idx[rare_mask]
            common_keys = [label_keys[i] for i in common_idx]

            common_train, common_test = train_test_split(
                common_idx, test_size=test_ratio,
                stratify=common_keys, random_state=seed,
            )
            train_idx = np.concatenate([common_train, rare_idx])
            test_idx = common_test
        else:
            train_idx, test_idx = train_test_split(
                idx, test_size=test_ratio,
                stratify=label_keys, random_state=seed,
            )

    train_idx = np.sort(train_idx)
    test_idx = np.sort(test_idx)

    X_train, X_test = theta[train_idx], theta[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]

    print(f"\nSplit: {len(train_idx)} train / {len(test_idx)} test "
          f"(ratio={test_ratio})")

    return X_train, X_test, Y_train, Y_test


def run_classifiers(X_train, X_test, Y_train, Y_test, label_names: list[str]):
    """
    Multi-label classifiers:
      - OneVsRest + LinearSVC
      - OneVsRest + LogisticRegression
      - RandomForest (native multi-output)
    """
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        hamming_loss, classification_report,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    classifiers = {
        "OVR_SVM": (
            OneVsRestClassifier(LinearSVC(max_iter=5000, random_state=42, dual="auto")),
            True
        ),
        "OVR_LogReg": (
            OneVsRestClassifier(LogisticRegression(max_iter=2000, random_state=42)),
            True
        ),
        "RF": (
            RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            False
        ),
    }

    results = {}
    for name, (clf, needs_scaling) in classifiers.items():
        Xtr = X_train_s if needs_scaling else X_train
        Xte = X_test_s  if needs_scaling else X_test

        clf.fit(Xtr, Y_train)
        Y_pred = clf.predict(Xte)

        # Overall metrics
        subset_acc  = float(accuracy_score(Y_test, Y_pred))       # exact match
        hamming     = float(hamming_loss(Y_test, Y_pred))
        micro_p     = float(precision_score(Y_test, Y_pred, average="micro", zero_division=0))
        micro_r     = float(recall_score(Y_test, Y_pred, average="micro", zero_division=0))
        micro_f1    = float(f1_score(Y_test, Y_pred, average="micro", zero_division=0))
        macro_p     = float(precision_score(Y_test, Y_pred, average="macro", zero_division=0))
        macro_r     = float(recall_score(Y_test, Y_pred, average="macro", zero_division=0))
        macro_f1    = float(f1_score(Y_test, Y_pred, average="macro", zero_division=0))
        samples_f1  = float(f1_score(Y_test, Y_pred, average="samples", zero_division=0))

        # Per-label metrics
        per_label = {}
        for i, lname in enumerate(label_names):
            y_true_i = Y_test[:, i]
            y_pred_i = Y_pred[:, i]
            per_label[lname] = {
                "precision": round(float(precision_score(y_true_i, y_pred_i, zero_division=0)), 4),
                "recall":    round(float(recall_score(y_true_i, y_pred_i, zero_division=0)), 4),
                "f1":        round(float(f1_score(y_true_i, y_pred_i, zero_division=0)), 4),
                "support":   int(y_true_i.sum()),
                "predicted_pos": int(y_pred_i.sum()),
            }

        report_text = classification_report(
            Y_test, Y_pred,
            target_names=label_names,
            zero_division=0,
        )

        results[name] = {
            "subset_accuracy": round(subset_acc, 4),
            "hamming_loss":    round(hamming, 4),
            "micro_precision": round(micro_p, 4),
            "micro_recall":    round(micro_r, 4),
            "micro_f1":        round(micro_f1, 4),
            "macro_precision": round(macro_p, 4),
            "macro_recall":    round(macro_r, 4),
            "macro_f1":        round(macro_f1, 4),
            "samples_f1":      round(samples_f1, 4),
            "per_label":       per_label,
            "report_text":     report_text,
        }

        print(f"\n  {name}:")
        print(f"    subset_acc={subset_acc:.4f}  hamming={hamming:.4f}")
        print(f"    micro_F1={micro_f1:.4f}  macro_F1={macro_f1:.4f}  samples_F1={samples_f1:.4f}")

    return results


def print_per_label_table(results: dict, label_names: list[str]):
    """Print a per-label comparison across classifiers."""
    print(f"\n{'='*90}")
    print("PER-LABEL F1 COMPARISON")
    print(f"{'='*90}")

    clf_names = list(results.keys())
    header = f"{'label':<28s}" + "".join(f"{c:>12s}" for c in clf_names) + f"{'support':>10s}"
    print(header)
    print("-" * len(header))

    for lname in label_names:
        row = f"{lname:<28s}"
        support = 0
        for cn in clf_names:
            pl = results[cn]["per_label"].get(lname, {})
            f1 = pl.get("f1", 0)
            support = pl.get("support", 0)
            row += f"{f1:>12.4f}"
        row += f"{support:>10d}"
        print(row)

    # Macro row
    row = f"{'--- MACRO ---':<28s}"
    for cn in clf_names:
        row += f"{results[cn]['macro_f1']:>12.4f}"
    row += f"{'':>10s}"
    print(row)


def main():
    ap = argparse.ArgumentParser(description="Multi-label classification on θ features")
    ap.add_argument("--theta", required=True, help="Path to doc_topic.npy")
    ap.add_argument("--csv",   required=True, help="Path to CSV with text + label columns")
    ap.add_argument("--label_cols", nargs="+", default=None,
                    help="Explicit list of label column names")
    ap.add_argument("--label_start_col", default=None,
                    help="First label column — all columns from here rightward are labels")
    ap.add_argument("--text_col", default="ABSTRACT",
                    help="Column containing document text (for inspection, default ABSTRACT)")
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default=None,
                    help="Where to save reports (default: same dir as --theta)")
    args = ap.parse_args()

    theta, Y, label_names, texts = load_data(
        args.theta, args.csv,
        args.label_cols, args.label_start_col,
        args.text_col,
    )

    X_train, X_test, Y_train, Y_test = split_data(
        theta, Y, args.test_ratio, args.seed,
    )

    clf_results = run_classifiers(X_train, X_test, Y_train, Y_test, label_names)

    print_per_label_table(clf_results, label_names)

    # Tail labels: F1 < 0.5 on SVM
    svm = clf_results.get("OVR_SVM", {})
    per_label = svm.get("per_label", {})
    tail = {k: v for k, v in per_label.items() if v["f1"] < 0.5}
    if tail:
        print(f"\n⚠ TAIL LABELS (OVR_SVM F1 < 0.5):")
        for lname, v in sorted(tail.items(), key=lambda x: x[1]["f1"]):
            print(f"  {lname:<28s}  F1={v['f1']:.3f}  prec={v['precision']:.3f}  "
                  f"rec={v['recall']:.3f}  support={v['support']}  predicted={v['predicted_pos']}")

    # Save
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.theta).parent.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "meta": {
            "theta_path":   args.theta,
            "csv_path":     args.csv,
            "n_docs":       int(theta.shape[0]),
            "K":            int(theta.shape[1]),
            "n_labels":     len(label_names),
            "label_names":  label_names,
            "test_ratio":   args.test_ratio,
            "seed":         args.seed,
            "n_train":      int(Y_train.shape[0]),
            "n_test":       int(Y_test.shape[0]),
        },
        "classifiers": clf_results,
    }

    json_path = out_dir / "classification_multilabel_report.json"
    json_path.write_text(json.dumps(output, indent=2))

    txt_path = out_dir / "classification_multilabel_report.txt"
    with open(txt_path, "w") as f:
        f.write(f"θ: {args.theta}\nCSV: {args.csv}\n")
        f.write(f"K={theta.shape[1]}  docs={theta.shape[0]}  labels={len(label_names)}\n")
        f.write(f"Train: {Y_train.shape[0]}  Test: {Y_test.shape[0]}\n\n")
        for cn, r in clf_results.items():
            f.write(f"--- {cn} ---\n")
            f.write(r["report_text"])
            f.write(f"\nsubset_accuracy={r['subset_accuracy']:.4f}  "
                    f"hamming_loss={r['hamming_loss']:.4f}\n\n")

    print(f"\nSaved: {json_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()