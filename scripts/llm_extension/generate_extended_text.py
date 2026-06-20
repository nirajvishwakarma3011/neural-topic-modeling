"""
generate_extended_texts.py
──────────────────────────
Step 1 of PVTM: use Mistral-7B-Instruct-v0.3 to expand each short text
into a richer, longer paragraph. Run this ONCE per dataset; the output
is cached and reused by pvtm_model.py during training.

Multi-GPU strategy
──────────────────
Each GPU gets its own model copy and its own slice of docs.
Each worker writes to a private temp file (no locking needed).
The main process merges temp files into the final JSONL after all workers finish.

Resume safety (single AND multi-GPU)
─────────────────────────────────────
On restart, done_indices are collected from:
  1. The final merged JSONL (if exists)
  2. Any surviving worker temp files from a crashed run
Remaining docs are re-split across GPUs. Nothing is ever overwritten.

Usage
─────
# 8 GPUs (fastest — ~3-4 min for 16k StackOverflow docs)
python generate_extended_texts.py \
    --dataset_cfg data_config/stackoverflow.json \
    --gpus 0 1 2 3 4 5 6 7

# 4 GPUs
python generate_extended_texts.py \
    --dataset_cfg data_config/stackoverflow.json \
    --gpus 0 1 2 3

# Single GPU (original behaviour)
python generate_extended_texts.py \
    --dataset_cfg data_config/stackoverflow.json \
    --gpus 0

Output format (.jsonl — one line per doc, sorted by idx)
──────────────────────────────────────────────────────────
{"idx": 0, "label": 7, "original": "...", "extended": "..."}
{"idx": 1, "label": 3, "original": "...", "extended": "..."}
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import torch


# ── Prompt ────────────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = (
    "Given the short text: \"{text}\", expand it into a detailed paragraph "
    "that provides background and elaborates on the key points to enrich "
    "its context. Try to make it as detailed as possible."
)

MODEL_REPO   = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_SUBDIR = "Mistral-7B-Instruct-v0.3"


# =============================================================================
# Download
# =============================================================================

def download_model(model_dir: Path) -> Path:
    """Download Mistral-7B-Instruct-v0.3 if not already present."""
    local_path = model_dir / MODEL_SUBDIR
    if (local_path / "config.json").exists():
        print(f"[download] Model already present at {local_path} — skipping.")
        return local_path

    print(f"[download] Downloading {MODEL_REPO} → {local_path}  (~15GB)")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[error] pip install huggingface_hub")
        sys.exit(1)

    local_path.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id         = MODEL_REPO,
        local_dir       = str(local_path),
        ignore_patterns = ["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
    )
    print(f"[download] Done → {local_path}")
    return local_path


# =============================================================================
# Worker — runs in a subprocess, one per GPU
# =============================================================================

def worker_fn(
    gpu_id:         int,
    model_path:     str,
    work_items:     list,       # list of (idx, doc) assigned to this GPU
    labels:         list,       # full labels list (indexed by original idx)
    tmp_path:       str,        # this worker's private output file
    max_new_tokens: int,
    beam_size:      int,
    batch_size:     int,
    max_input_len:  int,
    result_queue,               # mp.Queue — for progress reporting
):
    """
    Subprocess entry point.
    Loads the model on gpu_id, generates extended texts for work_items,
    writes results to tmp_path (append, line-buffered).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ── Load model on this GPU ────────────────────────────────────────────────
    device = torch.device(f"cuda:{gpu_id}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype       = torch.float16,
            device_map        = {"": gpu_id},
            low_cpu_mem_usage = True,
        )
        model.eval()
        result_queue.put(("loaded", gpu_id, len(work_items)))
    except Exception as e:
        result_queue.put(("error", gpu_id, str(e)))
        return

    # ── Generation helpers ────────────────────────────────────────────────────
    def build_prompt(text):
        msgs = [{"role": "user", "content": PROMPT_TEMPLATE.format(text=text)}]
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    def gen_batch(texts):
        prompts = [build_prompt(t) for t in texts]
        inputs  = tokenizer(
            prompts,
            return_tensors = "pt",
            padding        = True,
            truncation     = True,
            max_length     = max_input_len,
        ).to(device)
        prompt_len = inputs["input_ids"].shape[1]

        gen_kwargs = dict(
            max_new_tokens = max_new_tokens,
            pad_token_id   = tokenizer.eos_token_id,
            eos_token_id   = tokenizer.eos_token_id,
        )
        if beam_size > 1:
            gen_kwargs["num_beams"]      = beam_size
            gen_kwargs["early_stopping"] = True
        else:
            gen_kwargs["do_sample"] = False   # greedy

        with torch.no_grad():
            out_ids = model.generate(**inputs, **gen_kwargs)

        results = []
        for ids in out_ids:
            new_ids = ids[prompt_len:]
            results.append(
                tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            )
        return results

    # ── Generate in batches, write immediately ────────────────────────────────
    with open(tmp_path, "a", buffering=1) as f:
        for b_start in range(0, len(work_items), batch_size):
            batch        = work_items[b_start : b_start + batch_size]
            indices, docs = zip(*batch)

            try:
                extended = gen_batch(list(docs))
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    result_queue.put(("oom", gpu_id, batch_size))
                    return
                result_queue.put(("error", gpu_id, str(e)))
                return

            for idx, orig, ext in zip(indices, docs, extended):
                lbl = int(labels[idx]) if (labels is not None and idx < len(labels)) else None
                rec = json.dumps({
                    "idx":      int(idx),
                    "label":    lbl,
                    "original": orig,
                    "extended": ext,
                }, ensure_ascii=False)
                f.write(rec + "\n")

            result_queue.put(("progress", gpu_id, len(batch)))


# =============================================================================
# Data / dataset helpers
# =============================================================================

def load_docs_and_labels(dataset_cfg_path: str):
    """Load docs + labels using the same pipeline as main.py."""
    sys.path.insert(0, str(Path(__file__).parent))
    from src.preprocess import load_dataset
    cfg    = json.loads(Path(dataset_cfg_path).read_text())
    ds     = load_dataset(cfg)
    return ds["docs"], ds.get("labels", None), cfg["name"]


def collect_done_indices(out_path: Path, tmp_paths: list) -> set:
    """
    Read all already-generated idx values from:
      1. The merged final JSONL (if it exists)
      2. Any surviving worker temp files (from a crashed previous run)
    """
    done = set()
    files_to_check = [out_path] + tmp_paths
    for p in files_to_check:
        if Path(p).exists():
            with open(p, "r") as f:
                for line in f:
                    try:
                        done.add(json.loads(line.strip())["idx"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    return done


def merge_and_sort(tmp_paths: list, out_path: Path) -> int:
    """
    Merge all worker temp files + existing records in out_path into a single
    sorted JSONL. Deduplicates by idx. Returns number of records written.
    """
    records = {}

    # Read existing final file
    if out_path.exists():
        with open(out_path, "r") as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    records[rec["idx"]] = line.strip()
                except (json.JSONDecodeError, KeyError):
                    pass

    # Read worker temp files
    for tmp in tmp_paths:
        if Path(tmp).exists():
            with open(tmp, "r") as f:
                for line in f:
                    try:
                        rec = json.loads(line.strip())
                        records[rec["idx"]] = line.strip()
                    except (json.JSONDecodeError, KeyError):
                        pass

    # Write sorted output
    with open(out_path, "w") as f:
        for idx in sorted(records.keys()):
            f.write(records[idx] + "\n")

    # Clean up temp files
    for tmp in tmp_paths:
        if Path(tmp).exists():
            Path(tmp).unlink()

    return len(records)


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Generate LLM-extended texts for PVTM (multi-GPU)"
    )
    ap.add_argument("--dataset_cfg", required=True,
                    help="Path to dataset config JSON")
    ap.add_argument("--model_dir",
                    default="/data4/home/nirajv/small_text/LLM",
                    help="Directory containing / to download Mistral into")
    ap.add_argument("--output_dir",
                    default="/data4/home/nirajv/small_text/LLM/extended",
                    help="Directory to save .jsonl files")
    ap.add_argument("--gpus", type=int, nargs="+", default=[0],
                    help="GPU IDs to use  (e.g. --gpus 0 1 2 3 4 5 6 7)")
    ap.add_argument("--batch_size", type=int, default=16,
                    help="Docs per batch per GPU (16 is safe for 40GB + 500 tokens)")
    ap.add_argument("--max_new_tokens", type=int, default=500,
                    help="Max tokens generated per doc (paper: 500)")
    ap.add_argument("--beam_size", type=int, default=1,
                    help="Beam size: 1=greedy (fast), 5=paper (slow)")
    ap.add_argument("--max_input_len", type=int, default=128,
                    help="Max tokenized prompt length")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir  = Path(args.model_dir)

    n_gpus = len(args.gpus)
    print(f"[config] GPUs: {args.gpus}  ({n_gpus} workers)")

    # ── Load docs + labels ────────────────────────────────────────────────────
    print(f"[data] Loading from {args.dataset_cfg}...")
    docs, labels, dataset_name = load_docs_and_labels(args.dataset_cfg)
    print(f"[data] {len(docs)} docs | "
          f"{len(set(labels)) if labels else 0} classes")

    # ── Paths ─────────────────────────────────────────────────────────────────
    out_path  = output_dir / f"{dataset_name}_mistral.jsonl"
    tmp_paths = [
        str(output_dir / f"{dataset_name}_mistral_gpu{g}.tmp")
        for g in args.gpus
    ]

    # ── Resume: collect already-done indices ──────────────────────────────────
    done_indices = collect_done_indices(out_path, tmp_paths)
    if done_indices:
        print(f"[resume] {len(done_indices)} docs already done — skipping.")

    remaining = [(i, doc) for i, doc in enumerate(docs) if i not in done_indices]
    if not remaining:
        print("[done] All docs already generated.")
        # Still merge in case there are leftover temp files
        n = merge_and_sort(tmp_paths, out_path)
        print(f"[merged] {n} total records in {out_path}")
        return

    print(f"[generate] {len(remaining)} docs to generate across {n_gpus} GPU(s)")

    # ── Split docs across GPUs (round-robin for even load) ────────────────────
    # Round-robin guarantees each GPU gets ±1 doc regardless of dataset size
    splits = [[] for _ in range(n_gpus)]
    for i, item in enumerate(remaining):
        splits[i % n_gpus].append(item)

    for i, (gpu_id, split) in enumerate(zip(args.gpus, splits)):
        print(f"  GPU {gpu_id}: {len(split)} docs  "
              f"→ {tmp_paths[i]}")

    # ── Download model (once, in main process) ────────────────────────────────
    model_path = str(download_model(model_dir))

    # ── Launch worker processes ───────────────────────────────────────────────
    mp.set_start_method("spawn", force=True)
    result_queue = mp.Queue()
    processes    = []

    for i, gpu_id in enumerate(args.gpus):
        if not splits[i]:          # this GPU has nothing to do
            continue
        p = mp.Process(
            target = worker_fn,
            args   = (
                gpu_id,
                model_path,
                splits[i],
                labels,
                tmp_paths[i],
                args.max_new_tokens,
                args.beam_size,
                args.batch_size,
                args.max_input_len,
                result_queue,
            ),
            daemon = True,
        )
        p.start()
        processes.append((gpu_id, p))
        print(f"[spawn] Worker PID={p.pid} on GPU {gpu_id}")

    # ── Progress monitor (main process) ──────────────────────────────────────
    t_start       = time.time()
    n_loaded      = 0          # GPUs that finished loading
    n_total_gpus  = len(processes)
    per_gpu_done  = {g: 0 for g, _ in processes}
    per_gpu_total = {args.gpus[i]: len(splits[i]) for i in range(n_gpus)}
    all_loaded    = False

    while any(p.is_alive() for _, p in processes):
        try:
            msg = result_queue.get(timeout=2.0)
        except Exception:
            # Timeout — just update display
            _total_done = sum(per_gpu_done.values())
            _elapsed    = time.time() - t_start
            if _total_done > 0 and all_loaded:
                _per_doc = _elapsed / _total_done
                _eta     = (len(remaining) - _total_done) * _per_doc
                print(
                    f"\r[progress] {_total_done}/{len(remaining)} "
                    f"({100*_total_done/len(remaining):.1f}%)  "
                    f"ETA {_eta/60:.1f}min",
                    end="", flush=True
                )
            continue

        kind = msg[0]

        if kind == "loaded":
            _, gpu_id, n_items = msg
            n_loaded += 1
            print(f"\n[GPU {gpu_id}] Model loaded. {n_items} docs assigned.")
            if n_loaded == n_total_gpus:
                all_loaded = True
                print(f"[generate] All {n_total_gpus} GPUs ready — generating...")

        elif kind == "progress":
            _, gpu_id, n_batch = msg
            per_gpu_done[gpu_id] = per_gpu_done.get(gpu_id, 0) + n_batch

        elif kind == "error":
            _, gpu_id, err_msg = msg
            print(f"\n[ERROR] GPU {gpu_id}: {err_msg}")

        elif kind == "oom":
            _, gpu_id, batch_s = msg
            print(f"\n[OOM] GPU {gpu_id}: reduce --batch_size (was {batch_s}). "
                  f"Script is resume-safe — restart with smaller --batch_size.")

    # Wait for all processes to finish cleanly
    for gpu_id, p in processes:
        p.join()

    elapsed = time.time() - t_start
    print(f"\n\n[done] Generation complete in {elapsed/60:.1f} minutes.")

    # ── Merge all temp files into final sorted JSONL ──────────────────────────
    print(f"[merge] Combining {n_gpus} temp files → {out_path}")
    n_total = merge_and_sort(tmp_paths, out_path)
    print(f"[merge] {n_total} records written to {out_path}")

    # ── Speed summary ─────────────────────────────────────────────────────────
    newly_done = len(remaining)
    if elapsed > 0 and newly_done > 0:
        print(f"[speed] {newly_done/elapsed:.1f} docs/sec  |  "
              f"{newly_done*args.max_new_tokens/elapsed:.0f} tokens/sec total")

    print(f"\nNext step:")
    print(f"  python main.py --dataset_cfg {args.dataset_cfg} "
          f"--method_cfg model_config/pvtm.json")


if __name__ == "__main__":
    main()