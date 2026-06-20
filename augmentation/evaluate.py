import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from typing import List, Dict, Optional


# ── Binary label columns for GoogleNews-10 ───────────────────────────────────
BINARY_LABEL_COLS = [
    "China", "Kanyewest", "Taylor_swift", "black_friday_thanksgiving",
    "climate_change", "gaming_console", "google_map", "mobile_accessory",
    "scottist", "sport_soccer",
]


def _load_binary_labels(csv_path: str, text_col: str = "text") -> Dict[str, np.ndarray]:
    """Returns {text: binary_label_vector} for per-doc lookup."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[text_col]).reset_index(drop=True)
    df[text_col] = df[text_col].astype(str).str.strip()
    result = {}
    for _, row in df.iterrows():
        vec = np.array([int(row[c]) for c in BINARY_LABEL_COLS], dtype=np.float32)
        result[row[text_col]] = vec
    return result


def compute_rf_multilabel_f1(
    theta_train: np.ndarray,
    theta_test: np.ndarray,
    train_docs: List[str],
    test_docs: List[str],
    csv_path: str,
    label_cols: Optional[List[str]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Train RF on theta_train, evaluate on theta_test.
    Returns per-label F1 DataFrame.

    Uses ground-truth binary labels from CSV (not argmax heuristic).
    """
    if label_cols is None:
        label_cols = BINARY_LABEL_COLS

    text_to_labels = _load_binary_labels(csv_path)

    def gather_labels(docs):
        mat = []
        for d in docs:
            vec = text_to_labels.get(d.strip(), np.zeros(len(label_cols)))
            mat.append(vec)
        return np.array(mat)

    Y_train = gather_labels(train_docs)
    Y_test  = gather_labels(test_docs)

    rf = OneVsRestClassifier(
        RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
        n_jobs=-1,
    )
    rf.fit(theta_train, Y_train)
    Y_pred = rf.predict(theta_test)

    rows = []
    for i, col in enumerate(label_cols):
        f1   = f1_score(Y_test[:, i], Y_pred[:, i], zero_division=0)
        prec = f1_score(Y_test[:, i], Y_pred[:, i], average="binary", zero_division=0)
        sup  = int(Y_test[:, i].sum())
        rows.append({"class": col, "f1": round(f1, 4), "support": sup})

    macro_f1 = f1_score(Y_test, Y_pred, average="macro", zero_division=0)
    rows.append({"class": "MACRO", "f1": round(macro_f1, 4), "support": -1})

    return pd.DataFrame(rows)


def compare_baseline_augmented(
    theta_baseline_train: np.ndarray,
    theta_baseline_test: np.ndarray,
    theta_augmented_train: np.ndarray,
    theta_augmented_test: np.ndarray,
    train_docs: List[str],
    test_docs: List[str],
    csv_path: str,
    rare_class_names: List[str],
    label_cols: Optional[List[str]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Side-by-side multilabel RF F1: baseline vs augmented.
    Evaluates on held-out test_docs using ground-truth binary labels from CSV.
    """
    if label_cols is None:
        label_cols = BINARY_LABEL_COLS

    df_base = compute_rf_multilabel_f1(
        theta_baseline_train, theta_baseline_test,
        train_docs, test_docs, csv_path, label_cols, seed,
    ).rename(columns={"f1": "f1_baseline"})

    df_aug = compute_rf_multilabel_f1(
        theta_augmented_train, theta_augmented_test,
        train_docs, test_docs, csv_path, label_cols, seed,
    ).rename(columns={"f1": "f1_augmented"})

    df = df_base.merge(df_aug[["class", "f1_augmented"]], on="class")
    df["f1_delta"] = (df["f1_augmented"] - df["f1_baseline"]).round(4)
    df["is_rare"] = df["class"].isin(rare_class_names)

    print("\n=== Rare Class F1 Delta ===")
    rare = df[df["is_rare"]][["class", "f1_baseline", "f1_augmented", "f1_delta", "support"]]
    print(rare.to_string(index=False))

    print("\n=== Common Class F1 Delta ===")
    common = df[~df["is_rare"] & (df["class"] != "MACRO")][
        ["class", "f1_baseline", "f1_augmented", "f1_delta", "support"]
    ]
    print(common.to_string(index=False))

    print("\n=== Macro F1 ===")
    macro = df[df["class"] == "MACRO"][["class", "f1_baseline", "f1_augmented", "f1_delta"]]
    print(macro.to_string(index=False))

    return df


def verify_cluster_coherence_post_augmentation(
    detector_before,
    documents_augmented: List[str],
    embedding_model: str = "all-MiniLM-L6-v2",
) -> Dict:
    """
    Re-runs HDBSCAN on augmented corpus.
    Checks if rare clusters grew in size while retaining tightness.
    """
    from .cluster_detector import RareClusterDetector

    detector_after = RareClusterDetector(
        embedding_model=embedding_model,
        min_cluster_size=detector_before.min_cluster_size,
        min_samples=detector_before.min_samples,
        silhouette_threshold=detector_before.silhouette_threshold,
        contamination=detector_before.contamination,
        max_rare_size_frac=detector_before.max_rare_size_frac,
        device=detector_before.device,
    )
    detector_after.fit(documents_augmented)

    before_df = detector_before.get_summary_df()
    after_df = detector_after.get_summary_df()

    print("\n=== Cluster State Before Augmentation ===")
    rare_before = before_df[before_df["is_rare_candidate"]]
    print(rare_before.to_string(index=False) if not rare_before.empty else "None detected.")

    print("\n=== Cluster State After Augmentation ===")
    small_threshold = rare_before["size"].max() if not rare_before.empty else 0
    grew = after_df[after_df["size"] > small_threshold]
    print(grew.sort_values("size", ascending=False).head(10).to_string(index=False))

    return {"before": before_df, "after": after_df}
