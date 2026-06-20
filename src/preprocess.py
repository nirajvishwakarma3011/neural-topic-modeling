# from typing import Dict, Any
# from sklearn.datasets import fetch_20newsgroups

# def load_dataset(cfg: Dict[str, Any]):
#     name = cfg.get("name", "unknown")
#     loader = cfg.get("loader", "dummy")
#     labels = cfg.get("labels", False)
#     min_doc_len = int(cfg.get("min_doc_len", 0))

#     if loader == "sklearn_20news":
#         data = fetch_20newsgroups(subset='all', remove=('headers','footers','quotes'))
#         docs = [" ".join(d.split()) for d in data.data]
#         if min_doc_len > 0:
#             keep = [i for i, d in enumerate(docs) if len(d.split()) >= min_doc_len]
#             docs = [docs[i] for i in keep]
#             y = [int(data.target[i]) for i in keep] if labels else None
#         else:
#             y = [int(v) for v in data.target] if labels else None
#         target_names = data.target_names if labels else None
#         return {"name": name, "docs": docs, "labels": y, "target_names": target_names} #name is the model name
#     else:
#         # dummy placeholder
#         docs = [
#             "this is a short dummy document about sports",
#             "another dummy doc about politics and policy and news",
#             "third dummy about technology and science"
#         ]
#         y = [0,1,2] if labels else None
#         return {"name": name, "docs": docs, "labels": y, "target_names": None}

# src/data/preprocess.py
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json, gzip
from dataclasses import dataclass

# #---------- Public API ----------
# load_dataset(cfg) returns:
#   { "name": str, "docs": List[str], "labels": Optional[List[int]], "target_names": Optional[List[str]] }

# Caching:
#   RAW:        data/raw/{name}/{loader}/{split}/raw.jsonl.gz (+ meta.json)
#   PROCESSED:  data/processed/{name}/{loader}/{split}/min{min_doc_len}_labels{0|1}.jsonl.gz (+ meta.json)

# Config keys (common):
#   name: str
#   loader: str  ("sklearn_20news" | "guten_nltk" | "guten_pg19" | "nyt" | "yelp_polarity" | "yelp_full" | "dummy")
#   labels: bool
#   min_doc_len: int
#   split: str   (loader-specific default below)
#   cache_root: str = "data"
#   overwrite_raw: bool = False
#   overwrite_processed: bool = False

# Loader-specific optional keys are same as previous message.


# ----------------- utils -----------------
# @dataclass
# class DS:
#     docs: List[str]
#     y: Optional[List[int]]
#     target_names: Optional[List[str]]


@dataclass
class DS:
    docs: List[str]
    y: Optional[List[int]]
    target_names: Optional[List[str]]
    eval_docs: Optional[List[str]] = None 

def _normalize_docs(texts: List[str]) -> List[str]:
    return [" ".join((t or "").split()) for t in texts]

def _apply_min_len(docs: List[str], y: Optional[List[int]], min_doc_len: int) -> DS:
    if min_doc_len <= 0:
        return DS(docs, y, None)  # target_names handled separately
    keep = [i for i, d in enumerate(docs) if len(d.split()) >= min_doc_len]
    docs2 = [docs[i] for i in keep]
    y2 = [y[i] for i in keep] if y is not None else None
    return DS(docs2, y2, None)

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _paths(cache_root: Path, name: str, loader: str, split: str, min_doc_len: int, want_labels: bool):
    raw_dir = cache_root / "raw" / name / loader / split
    proc_dir = cache_root / "processed" / name / loader / split
    raw_jsonl = raw_dir / "raw.jsonl.gz"
    raw_meta = raw_dir / "meta.json"
    proc_jsonl = proc_dir / f"min{min_doc_len}_labels{1 if want_labels else 0}.jsonl.gz"
    proc_meta = proc_dir / f"min{min_doc_len}_labels{1 if want_labels else 0}.meta.json"
    return raw_dir, raw_jsonl, raw_meta, proc_dir, proc_jsonl, proc_meta

def _write_jsonl_gz(path: Path, rows: List[dict]) -> None:
    _ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)

def _read_jsonl_gz(path: Path) -> List[dict]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def _write_meta(path: Path, meta: dict) -> None:
    _ensure_dir(path.parent)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    tmp.replace(path)

def _read_meta(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

def _to_rows(docs: List[str], y: Optional[List[int]]) -> List[dict]:
    if y is None:
        return [{"text": t} for t in docs]
    return [{"text": t, "label": int(lbl)} for t, lbl in zip(docs, y)]

def _from_rows(rows: List[dict]) -> Tuple[List[str], Optional[List[int]]]:
    docs = [r.get("text", "") for r in rows]
    y: Optional[List[int]] = None
    if "label" in rows[0] if rows else False:
        y = [int(r.get("label")) for r in rows]
    return docs, y


import re

def _clean_text(text: str) -> str:
    """
    Lowercase and remove punctuation.
    Keeps only letters, digits, and single spaces.
    """
    text = text.lower()                        # "Hello World!" → "hello world!"
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # "hello world!" → "hello world "
    text = " ".join(text.split())              # collapse multiple spaces → "hello world"
    return text

# ----------------- core orchestration -----------------
def load_dataset(cfg: Dict[str, Any]):
    name = str(cfg.get("name", "unknown"))
    loader = str(cfg.get("loader", "dummy"))
    want_labels = bool(cfg.get("labels", False))
    min_doc_len = int(cfg.get("min_doc_len", 0))
    cache_root = Path(cfg.get("cache_root", "data"))
    # overwrite_raw = bool(cfg.get("overwrite_raw", False))
    # overwrite_proc = bool(cfg.get("overwrite_processed", False))
    overwrite_raw = True
    overwrite_proc = True

    # default split per loader
    split = str(cfg.get("split", _default_split(loader)))

    raw_dir, raw_jsonl, raw_meta, proc_dir, proc_jsonl, proc_meta = _paths(
        cache_root, name, loader, split, min_doc_len, want_labels
    )
    print("Inside loader")
    # 1) If processed exists and not overwrite -> load and return
    if proc_jsonl.exists() and proc_meta.exists() and not overwrite_proc:
        print("Inside loader and processed exits")
        rows = _read_jsonl_gz(proc_jsonl)
        docs, y = _from_rows(rows)
        meta = _read_meta(proc_meta)
        target_names = meta.get("target_names")
        return {"name": name, "docs": docs, "labels": y, "target_names": target_names}

    # 2) Ensure RAW exists (load or fetch)
    if raw_jsonl.exists() and raw_meta.exists() and not overwrite_raw:
        print("RAW exists  Laoding from it")
        raw_rows = _read_jsonl_gz(raw_jsonl)
        raw_docs, raw_y = _from_rows(raw_rows)
        raw_meta_obj = _read_meta(raw_meta)
        target_names_raw = raw_meta_obj.get("target_names")
    else:
        # fetch using underlying loader
        fetched = _fetch_dataset_via_loader(cfg, loader, split, want_labels)
        print("Laoding from loader")
        raw_docs = fetched.docs
        raw_y = fetched.y if want_labels else None
        target_names_raw = fetched.target_names if want_labels else None
        # save RAW
        _ensure_dir(raw_dir)
        _write_jsonl_gz(raw_jsonl, _to_rows(raw_docs, raw_y))
        _write_meta(raw_meta, {
            "name": name, "loader": loader, "split": split,
            "target_names": target_names_raw
        })

    # 3) Process (min_doc_len) and store PROCESSED
    normalized_docs = _normalize_docs(raw_docs)
    ds = _apply_min_len(normalized_docs, raw_y, min_doc_len)
    docs, y = ds.docs, ds.y
    target_names = target_names_raw  # unchanged by filtering 

    _ensure_dir(proc_dir)
    _write_jsonl_gz(proc_jsonl, _to_rows(docs, y))
    _write_meta(proc_meta, {
        "name": name, "loader": loader, "split": split,
        "min_doc_len": min_doc_len, "labels": want_labels,
        "target_names": target_names
    })

    return {"name": name, "docs": docs, "labels": y, "target_names": target_names, "eval_docs": ds.eval_docs}


def _default_split(loader: str) -> str:
    # sensible defaults per dataset
    if loader in ("yelp_polarity", "yelp_full", "nyt"):
        return "train"
    if loader == "sklearn_20news":
        return "all"
    if loader in ("guten_nltk", "guten_pg19"):
        return "train"
    return "default"


# ----------------- dataset fetchers (no caching here) -----------------
def _fetch_dataset_via_loader(cfg: Dict[str, Any], loader: str, split: str, want_labels: bool) -> DS:
    if loader == "sklearn_20news":
        from sklearn.datasets import fetch_20newsgroups
        data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
        docs = list(data.data)
        y = list(map(int, data.target)) if want_labels else None
        target_names = list(data.target_names) if want_labels else None
        return DS(docs, y, target_names)

    if loader == "guten_nltk":
        guten_files = cfg.get("guten_files")
        nltk_download = bool(cfg.get("nltk_download", True))
        try:
            import nltk
            from nltk.corpus import gutenberg
            if nltk_download:
                nltk.download("gutenberg", quiet=True)
        except Exception as e:
            raise RuntimeError("NLTK Gutenberg not available. Install nltk and try again.") from e
        files = gutenberg.fileids() if not guten_files else guten_files
        texts = [" ".join(gutenberg.words(fid)) for fid in files]
        return DS(texts, None, None)

    if loader == "guten_pg19":
        from datasets import load_dataset
        sample_n = cfg.get("sample_n")
        ds = load_dataset("pg19", split="train")
        texts = [r["text"] for r in ds]
        if sample_n: texts = texts[: int(sample_n)]
        return DS(texts, None, None)

    if loader == "nyt":
        # Expect preprocessed files like TopClus:
        # datasets/nyt/
        #   docs.txt          (one document per line)
        #   labels_topic.txt  (optional; one integer label [0..9] per line)
        #   labels_loc.txt    (optional; one integer label [0..9] per line)
        from pathlib import Path

        base_dir = Path(cfg.get("nyt_dir", "datasets/nyt"))
        label_scheme = cfg.get("nyt_label_scheme", "location")  # "topic" | "location"
        use_all = (split == "all")  # kept for API symmetry; NYT often ships as a single split

        docs_fp = base_dir / "texts.txt"
        lbl_topic_fp = base_dir / "label_topic.txt"
        lbl_loc_fp = base_dir / "label_location.txt"

        if not docs_fp.exists():
            raise FileNotFoundError(f"NYT docs not found at {docs_fp}. "
                                    f"Place TopClus-style files under {base_dir}.")

        # load docs
        texts = [ln.rstrip("\n") for ln in docs_fp.open("r", encoding="utf-8", errors="ignore")]

        # optional labels
        y = None
        target_names = None
        if want_labels:
            if label_scheme == "topic":
                if not lbl_topic_fp.exists():
                    raise FileNotFoundError(f"NYT topic labels missing at {lbl_topic_fp}")
                y = [int(ln.strip()) for ln in lbl_topic_fp.open("r", encoding="utf-8")]
                # 10 human-defined topics; if you have names, put them here:
                target_names = cfg.get("nyt_topic_names")  # or keep None
            elif label_scheme in ("location", "loc"):
                if not lbl_loc_fp.exists():
                    raise FileNotFoundError(f"NYT location labels missing at {lbl_loc_fp}")
                y = [int(ln.strip()) for ln in lbl_loc_fp.open("r", encoding="utf-8")]
                # 10 pre-defined countries; provide the list if available:
                target_names = cfg.get("nyt_location_names")  # or keep None
            else:
                raise ValueError("nyt_label_scheme must be 'topic' or 'location'")

            if len(y) != len(texts):
                raise ValueError(f"Label count ({len(y)}) != doc count ({len(texts)})")

        return DS(texts, y, target_names)


    if loader == "yelp":
        from pathlib import Path
        import gzip
        import random

        base_dir   = Path(cfg.get("yelp_dir", "datasets/yelp"))
        text_file  = Path(cfg.get("text_file", "text.txt"))
        label_file = Path(cfg.get("labels_file", "labels.txt"))  # optional
        seed       = int(cfg.get("seed", 42))
        sample_n   = cfg.get("sample_n")  # optional int
        shuffle    = bool(cfg.get("shuffle_before_sample", True))

        # Resolve paths (allow .gz transparently)
        def _resolve(p: Path) -> Path:
            if (base_dir / p).exists():
                return (base_dir / p)
            gz = (base_dir / (p.name + ".gz"))
            if gz.exists():
                return gz
            raise FileNotFoundError(f"Expected file not found: {(base_dir / p)} or {gz}")

        text_fp  = _resolve(text_file)

        # Read text lines (skip empty)
        def _read_lines_any(path: Path):
            open_fn = gzip.open if str(path).endswith(".gz") else open
            with open_fn(path, "rt", encoding="utf-8", errors="ignore") as f:
                return [ln.rstrip("\n") for ln in f if ln.strip()]

        texts = _read_lines_any(text_fp)

        # Optional labels
        y = None
        target_names = None
        if want_labels:
            lbl_path = base_dir / label_file
            if not lbl_path.exists() and (base_dir / (label_file.name + ".gz")).exists():
                lbl_path = base_dir / (label_file.name + ".gz")
            if lbl_path.exists():
                labels_raw = _read_lines_any(lbl_path)
                y = [int(v) for v in labels_raw]
                if len(y) != len(texts):
                    raise ValueError(f"labels.txt length ({len(y)}) != text.txt length ({len(texts)})")
                # If you have names, pass via cfg e.g. ["1","2","3","4","5"]
                target_names = cfg.get("target_names")

        # Deterministic downsample (optional)
        if sample_n is not None:
            n = int(sample_n)
            idx = list(range(len(texts)))
            if shuffle:
                random.seed(seed)
                random.shuffle(idx)
            idx = idx[:n]
            texts = [texts[i] for i in idx]
            if y is not None:
                y = [y[i] for i in idx]

        return DS(texts, y, target_names)        
    # if loader == "yelp_polarity":
    #     from datasets import load_dataset
    #     ds = load_dataset("yelp_polarity", split=split)
    #     texts = [r["text"] for r in ds]
    #     y = [int(r["label"]) for r in ds] if want_labels else None
    #     target_names = ["negative", "positive"] if want_labels else None
    #     return DS(texts, y, target_names)

    # # if loader == "yelp_full":
    # #     from datasets import load_dataset
    # #     ds = load_dataset("yelp_review_full", split=split)
    # #     texts = [r["text"] for r in ds]
    # #     y = [int(r["label"]) for r in ds] if want_labels else None
    # #     target_names = ["1", "2", "3", "4", "5"] if want_labels else None
    # #     return DS(texts, y, target_names)
    # if loader == "yelp_full":
    #     from datasets import load_dataset

    #     star_label = 2   # 0..4 (default 2 => 3-star)
    #     take_n = 30000    # default 30k
    #     seed = 42

    #     def _load_one(split_name: str):
    #         ds_part = load_dataset("yelp_review_full", split=split_name)
    #         ds_part = ds_part.filter(lambda r: int(r["label"]) == star_label)
    #         return ds_part

    #     ds = _load_one("train")

    #     # deterministic downsample to exactly take_n (or fewer if not enough after filtering)
    #     if len(ds) > take_n:
    #         ds = ds.shuffle(seed=seed).select(range(take_n))

    #     texts = [r["text"] for r in ds]
    #     y = [int(r["label"]) for r in ds] if want_labels else None
    #     target_names = ["1", "2", "3", "4", "5"] if want_labels else None
    #     return DS(texts, y, target_names)
    
#     if loader == "googlenewst":
#         # from pathlib import Path
#         # import gzip
#         # import json

#         # # Path config
#         # base_dir = Path(cfg.get("data_dir", "data/raw"))
#         # file_name = cfg.get("file_name", "GoogleNewsT.txt")

#         # file_path = base_dir / file_name
#         # if not file_path.exists():
#         #     raise FileNotFoundError(f"GoogleNewsT file not found at {file_path}")

#         # docs = []
#         # y = []

#         # # Support .gz automatically
#         # open_fn = gzip.open if str(file_path).endswith(".gz") else open
#         # with open_fn(file_path, "rt", encoding="utf-8") as f:
#         #     for line in f:
#         #         if not line.strip():
#         #             continue
#         #         obj = json.loads(line)
#         #         docs.append(obj["text"])
#         #         if want_labels:
#         #             y.append(int(obj["cluster"]))

#         # if want_labels:
#         #     return DS(docs, y, None)
#         # else:
#         #     return DS(docs, None, None)

# ######### CSV 

#         # from pathlib import Path
#         # import csv

#         # base_dir = Path(cfg.get("data_dir", ""))
#         # file_name = cfg.get("file_name", "")

#         # file_path = base_dir / file_name
#         # if not file_path.exists():
#         #     raise FileNotFoundError(f"GOOGLENEWS file not found at {file_path}")

#         # docs = []
#         # clusters = []

#         # with open(file_path, "r", encoding="utf-8") as f:
#         #     reader = csv.DictReader(f)
#         #     if "text" not in reader.fieldnames:
#         #         raise ValueError("CSV must contain a 'text' column")
#         #     if want_labels and "cluster" not in reader.fieldnames:
#         #         raise ValueError("CSV must contain a 'cluster' column when labels=True")

#         #     for row in reader:
#         #         text = _clean_text(row["text"])
#         #         if not text:
#         #             continue
#         #         docs.append(text)

#         #         if want_labels:
#         #             clusters.append(int(row["cluster"]))

#         # # Remap cluster IDs to 0..K-1 (important for evaluation)
#         # if want_labels:
#         #     unique = sorted(set(clusters))
#         #     id_map = {old: i for i, old in enumerate(unique)}
#         #     y = [id_map[c] for c in clusters]
#         # else: 
#         #     y = None

#         # return DS(docs, y, None)



# ##### Cluster 
#         from pathlib import Path
#         import csv

#         base_dir = Path(cfg.get("data_dir", ""))
#         file_name = cfg.get("file_name", "")

#         file_path = base_dir / file_name
#         if not file_path.exists():
#             raise FileNotFoundError(f"GOOGLENEWS file not found at {file_path}")

#         docs = []
#         clusters = []
        
#         # --- NEW: Define which clusters get the extended text ---
#         # Replace these numbers with your actual cluster IDs
#         EXTENDED_TEXT_CLUSTERS = {50, 104} 

#         with open(file_path, "r", encoding="utf-8") as f:
#             reader = csv.DictReader(f)
            
#             # --- NEW: Added safety check for extended_text ---
#             if "text" not in reader.fieldnames or "extended_text" not in reader.fieldnames:
#                 raise ValueError("CSV must contain both 'text' and 'extended_text' columns")
#             if want_labels and "cluster" not in reader.fieldnames:
#                 raise ValueError("CSV must contain a 'cluster' column when labels=True")

#             for row in reader:
#                 # Grab the cluster ID first so we can check it
#                 current_cluster = int(float(row["cluster"]))
                
#                 # --- NEW: Conditional text selection ---
#                 if current_cluster in EXTENDED_TEXT_CLUSTERS:
#                     raw_text = row["extended_text"]
#                 else:
#                     raw_text = row["text"]
                
#                 # Clean whichever text we selected
#                 text = _clean_text(raw_text)
                
#                 if not text:
#                     continue
                
#                 docs.append(text)

#                 if want_labels:
#                     clusters.append(current_cluster)

#         # Remap cluster IDs to 0..K-1 (important for evaluation)
#         if want_labels:
#             unique = sorted(set(clusters))
#             id_map = {old: i for i, old in enumerate(unique)}
#             y = [id_map[c] for c in clusters]
#         else: 
#             y = None

#         return DS(docs, y, None)

    if loader == "googlenewst":
        from pathlib import Path
        import csv

        base_dir = Path(cfg.get("data_dir", ""))
        file_name = cfg.get("file_name", "")
        file_path = base_dir / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"GOOGLENEWS file not found at {file_path}")

        # Raw cluster IDs (pre-remap) that should be trained on extended text.
        EXTENDED_TEXT_CLUSTERS = set(cfg.get("extended_clusters", []))

        docs = []          # what the model trains on (may be extended)
        eval_docs = []     # always original text, row-aligned with docs
        clusters = []

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "text" not in reader.fieldnames or "extended_text" not in reader.fieldnames:
                raise ValueError("CSV must contain both 'text' and 'extended_text' columns")
            if want_labels and "cluster" not in reader.fieldnames:
                raise ValueError("CSV must contain a 'cluster' column when labels=True")

            for row in reader:
                current_cluster = int(float(row["cluster"]))

                # Original text is what we always want at eval time.
                orig_text = _clean_text(row["text"])
                # Train-time text: extended for selected clusters, else original.
                if current_cluster in EXTENDED_TEXT_CLUSTERS:
                    train_text = _clean_text(row["extended_text"])
                else:
                    train_text = orig_text

                # CRITICAL: skip the row iff EITHER side is empty, otherwise
                # docs and eval_docs would drift out of alignment.
                if not orig_text or not train_text:
                    continue

                docs.append(train_text)
                eval_docs.append(orig_text)
                if want_labels:
                    clusters.append(current_cluster)

        # Cluster ID remap happens AFTER the extended-text decision, as agreed.
        if want_labels:
            unique = sorted(set(clusters))
            id_map = {old: i for i, old in enumerate(unique)}
            y = [id_map[c] for c in clusters]
        else:
            y = None

        return DS(docs, y, None, eval_docs=eval_docs)


    if loader == "tweet":
        from pathlib import Path
        import gzip
        import json
        print("Loaded Tweet Dattset")
        base_dir = Path(cfg.get("data_dir", "data/raw"))
        file_name = cfg.get("file_name", "Tweet.txt")

        file_path = base_dir / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"Tweet file not found at {file_path}")

        docs = []
        clusters = []

        open_fn = gzip.open if str(file_path).endswith(".gz") else open
        with open_fn(file_path, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                docs.append(obj["text"])
                if want_labels:
                    clusters.append(int(obj["cluster"]))

        # Remap cluster ids to 0..K-1 (important for evaluation)
        if want_labels:
            unique = sorted(set(clusters))
            id_map = {old: i for i, old in enumerate(unique)}
            y = [id_map[c] for c in clusters]
        else:
            y = None

        return DS(docs, y, None)


    if loader == "stackoverflow":
        from pathlib import Path
        import csv

        base_dir = Path(cfg.get("data_dir", "data/raw"))
        file_name = cfg.get("file_name", "StackOverflow.csv")

        file_path = base_dir / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"StackOverflow file not found at {file_path}")

        docs = []
        clusters = []

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "text" not in reader.fieldnames:
                raise ValueError("CSV must contain a 'text' column")
            if want_labels and "cluster" not in reader.fieldnames:
                raise ValueError("CSV must contain a 'cluster' column when labels=True")

            for row in reader:
                text = _clean_text(row["text"])
                if not text:
                    continue
                docs.append(text)

                if want_labels:
                    clusters.append(int(row["cluster"]))

        # Remap cluster IDs to 0..K-1 (important for evaluation)
        if want_labels:
            unique = sorted(set(clusters))
            id_map = {old: i for i, old in enumerate(unique)}
            y = [id_map[c] for c in clusters]
        else:
            y = None

        return DS(docs, y, None)

    if loader == "abstract":
        from pathlib import Path
        import csv

        base_dir = Path(cfg.get("data_dir", "data/raw"))
        file_name = cfg.get("file_name", "StackOverflow.csv")

        file_path = base_dir / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"StackOverflow file not found at {file_path}")

        docs = []
        clusters = []

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "ABSTRACT" not in reader.fieldnames:
                raise ValueError("CSV must contain a 'text' column")
            if want_labels and "cluster" not in reader.fieldnames:
                raise ValueError("CSV must contain a 'cluster' column when labels=True")

            for row in reader:
                text = _clean_text(row["ABSTRACT"])
                if not text:
                    continue
                docs.append(text)

                if want_labels:
                    clusters.append(int(row["cluster"]))

        # Remap cluster IDs to 0..K-1 (important for evaluation)
        if want_labels:
            unique = sorted(set(clusters))
            id_map = {old: i for i, old in enumerate(unique)}
            y = [id_map[c] for c in clusters]
        else:
            y = None

        return DS(docs, y, None) 


    if loader == "mpst":
        from pathlib import Path
        import csv

        base_dir = Path(cfg.get("data_dir", "data"))
        file_name = cfg.get("file_name", "mpst_8_labels_final.csv")
        file_path = base_dir / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"MPST file not found at {file_path}")

        docs = []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "plot" not in reader.fieldnames:
                raise ValueError("CSV must contain a 'plot' column")
            for row in reader:
                text = _clean_text(row["plot"])
                if not text:
                    continue
                docs.append(text)

        return DS(docs, None, None)

    if loader == "reuters_10":
        from pathlib import Path
        import csv

        base_dir = Path(cfg.get("data_dir", "data"))
        file_name = cfg.get("file_name", "reuters_10.csv")
        file_path = base_dir / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"Reuters-10 file not found at {file_path}")

        docs = []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "text" not in reader.fieldnames:
                raise ValueError("CSV must contain a 'text' column")
            for row in reader:
                text = _clean_text(row["text"])
                if not text:
                    continue
                docs.append(text)

        return DS(docs, None, None)

    if loader == "tweet_csv":
        from pathlib import Path
        import csv

        base_dir = Path(cfg.get("data_dir", "data"))
        file_name = cfg.get("file_name", "tweet_10_labels.csv")
        file_path = base_dir / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"Tweet CSV file not found at {file_path}")

        # Clusters whose docs should train on extended_text (if column present)
        extended_clusters = set(cfg.get("extended_clusters", []))

        docs = []
        eval_docs = []
        clusters = []

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "text" not in reader.fieldnames:
                raise ValueError("CSV must contain a 'text' column")
            if want_labels and "cluster" not in reader.fieldnames:
                raise ValueError("CSV must contain a 'cluster' column when labels=True")

            has_extended = "extended_text" in reader.fieldnames

            for row in reader:
                orig_text = _clean_text(row["text"])
                if not orig_text:
                    continue

                current_cluster = int(float(row["cluster"]))
                if has_extended and current_cluster in extended_clusters:
                    train_text = _clean_text(row["extended_text"]) or orig_text
                else:
                    train_text = orig_text

                docs.append(train_text)
                eval_docs.append(orig_text)
                if want_labels:
                    clusters.append(current_cluster)

        if want_labels:
            unique = sorted(set(clusters))
            id_map = {old: i for i, old in enumerate(unique)}
            y = [id_map[c] for c in clusters]
        else:
            y = None

        return DS(docs, y, None)

    if loader == "20news":
        from pathlib import Path
        import csv

        base_dir = Path(cfg.get("data_dir", "data"))
        file_name = cfg.get("file_name", "20news_5class.csv")
        file_path = base_dir / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"20news file not found at {file_path}")

        docs = []
        clusters = []

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "text" not in reader.fieldnames:
                raise ValueError("CSV must contain a 'text' column")
            if want_labels and "cluster" not in reader.fieldnames:
                raise ValueError("CSV must contain a 'cluster' column when labels=True")

            for row in reader:
                text = _clean_text(row["text"])
                if not text:
                    continue
                docs.append(text)
                if want_labels:
                    clusters.append(int(row["cluster"]))

        if want_labels:
            unique = sorted(set(clusters))
            id_map = {old: i for i, old in enumerate(unique)}
            y = [id_map[c] for c in clusters]
        else:
            y = None

        return DS(docs, y, None)

    if loader == "dummy":
        docs = [
            "this is a short dummy document about sports",
            "another dummy doc about politics and policy and news",
            "third dummy about technology and science",
        ]
        y = [0, 1, 2] if want_labels else None
        return DS(docs, y, ["sports", "politics", "tech"] if want_labels else None)

    raise ValueError(f"Unknown loader: {loader}")
