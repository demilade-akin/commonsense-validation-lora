"""Run every method on both ComVE subtasks and report accuracy and timing.

Usage, from the repository root:
    python run_all.py --quick            # tiny run: checks every stage and estimates full run time
    python run_all.py                    # full run on the test split
    python run_all.py --split dev        # full run on the dev split (for tuning)
    python run_all.py --methods random,zero-shot --subtasks A
    python run_all.py --methods random,zero-shot,few-shot,lora,roberta-base,distilbert

Results print as a table and are saved to results/results.json
(results/results_quick.json for --quick).
"""

from __future__ import annotations

import argparse
import gc
import traceback
from dataclasses import asdict, replace

from transformers.utils import logging as hf_logging

from src.baselines import random_predictions
from src.config import (
    ENCODER,
    ENCODER_MODELS,
    EVAL_N,
    EVAL_SPLIT,
    EXPECTED_COUNTS,
    FEW_SHOT_K,
    LORA,
    MODEL_NAME,
    QUICK_EVAL_N,
    QUICK_TRAIN_N,
    RESULTS_DIR,
    SEED,
)
from src.data import LETTERS, N_OPTIONS, downsample, load_subtask
from src.modeling import load_model_and_tokenizer
from src.prompting import evaluate_lm, few_shot_examples
from src.train_encoder import run_encoder
from src.train_lora import run_lora
from src.utils import Timer, accuracy, save_json, set_seed

METHODS = ["random", "zero-shot", "few-shot", "lora", *ENCODER_MODELS]
# DistilBERT is optional: leaving it out keeps the default run well under one hour
# on the course VM. Add it with --methods.
DEFAULT_METHODS = [m for m in METHODS if m != "distilbert"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run every method on ComVE Subtasks A and B.")
    parser.add_argument("--quick", action="store_true", help="tiny sizes: check every stage and time it")
    parser.add_argument("--split", default=EVAL_SPLIT, choices=["dev", "test"])
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS), help=f"comma-separated, from {METHODS}")
    parser.add_argument("--subtasks", default="AB", help="A, B or AB")
    args = parser.parse_args()
    args.methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = set(args.methods) - set(METHODS)
    if unknown:
        parser.error(f"unknown methods {sorted(unknown)}; choose from {METHODS}")
    return args


def run_method(method: str, subtask: str, eval_examples, args, lora_cfg, encoder_cfg, lm) -> dict:
    """Run one method on one subtask and return its result row."""
    labels = [ex.label for ex in eval_examples]

    if method == "random":
        with Timer(verbose=False) as t:
            predictions = random_predictions(len(eval_examples), N_OPTIONS[subtask], SEED)
        counts = {LETTERS[i]: predictions.count(i) for i in range(N_OPTIONS[subtask])}
        return {
            "accuracy": accuracy(predictions, labels),
            "n_eval": len(eval_examples),
            "prediction_counts": counts,
            "eval_seconds": t.seconds,
        }

    if method in ("zero-shot", "few-shot"):
        model, tokenizer = lm()
        shots = few_shot_examples(subtask) if method == "few-shot" else []
        result = evaluate_lm(model, tokenizer, subtask, eval_examples, shots)
        result["shots"] = len(shots)
        return result

    train_all = load_subtask(subtask, "train")
    if method == "lora":
        train = downsample(train_all, lora_cfg.train_n, SEED)
        return run_lora(subtask, train, eval_examples, lora_cfg, SEED)

    train = downsample(train_all, encoder_cfg.train_n, SEED)
    return run_encoder(ENCODER_MODELS[method], subtask, train, eval_examples, encoder_cfg, SEED)


def print_table(rows: list[dict]) -> None:
    header = f"{'Subtask':<8}{'Method':<14}{'Accuracy':>9}{'n_eval':>8}{'n_train':>9}{'Train s':>9}{'Eval s':>8}"
    print("\n" + header + "\n" + "-" * len(header))
    for r in rows:
        acc = f"{r['accuracy']:.3f}" if "accuracy" in r else "ERROR"
        print(
            f"{r['subtask']:<8}{r['method']:<14}{acc:>9}{r.get('n_eval', 0):>8}"
            f"{r.get('n_train', 0):>9}{r.get('train_seconds', 0):>9.0f}{r.get('eval_seconds', 0):>8.0f}"
        )


def estimate_full_run(rows: list[dict], split: str) -> None:
    """Scale the quick run's per-example times up to the full settings."""
    full_eval_n = EVAL_N or EXPECTED_COUNTS[split]
    full_train_n = {"lora": LORA.train_n or EXPECTED_COUNTS["train"]}
    full_train_n.update({m: ENCODER.train_n or EXPECTED_COUNTS["train"] for m in ENCODER_MODELS})
    full_train_n = {m: n * (LORA.epochs if m == "lora" else ENCODER.epochs) for m, n in full_train_n.items()}

    print("\nEstimated full run on this machine (model loading not included):")
    total = 0.0
    for r in rows:
        if "accuracy" not in r:
            continue
        seconds = r["eval_seconds"] / r["n_eval"] * full_eval_n
        if r.get("n_train"):
            seconds += r["train_seconds"] / r["n_train"] * full_train_n[r["method"]]
        total += seconds
        print(f"  {r['subtask']} {r['method']:<14}{seconds / 60:>7.1f} min")
    print(f"  {'Total':<16}{total / 60:>7.1f} min  (limit: 60 min; lower train_n in src/config.py if needed)")


def main() -> None:
    args = parse_args()
    hf_logging.set_verbosity_error()  # hide expected warnings, e.g. the new multiple-choice head

    eval_n = QUICK_EVAL_N if args.quick else EVAL_N
    lora_cfg = replace(LORA, train_n=QUICK_TRAIN_N) if args.quick else LORA
    encoder_cfg = replace(ENCODER, train_n=QUICK_TRAIN_N) if args.quick else ENCODER
    out_path = RESULTS_DIR / ("results_quick.json" if args.quick else "results.json")

    # The base language model is loaded once and shared by zero-shot and
    # few-shot prompting. LoRA loads its own fresh copy for each subtask.
    cache: dict = {}

    def lm():
        if "lm" not in cache:
            cache["lm"] = load_model_and_tokenizer()
        return cache["lm"]

    rows: list[dict] = []
    settings = {
        "quick": args.quick,
        "split": args.split,
        "eval_n": eval_n,
        "seed": SEED,
        "model": MODEL_NAME,
        "encoders": ENCODER_MODELS,
        "few_shot_k": FEW_SHOT_K,
        "lora": asdict(lora_cfg),
        "encoder": asdict(encoder_cfg),
    }

    with Timer("Total", verbose=False) as total:
        for method in args.methods:
            if method in ("lora", *ENCODER_MODELS):
                cache.clear()  # free the shared base model before training
                gc.collect()
            for subtask in args.subtasks:
                print(f"\n=== Subtask {subtask} | {method} ===")
                eval_examples = downsample(load_subtask(subtask, args.split), eval_n, SEED)
                set_seed(SEED)
                row = {"subtask": subtask, "method": method, "split": args.split}
                try:
                    row.update(run_method(method, subtask, eval_examples, args, lora_cfg, encoder_cfg, lm))
                    print(f"accuracy {row['accuracy']:.3f} | picks {row['prediction_counts']}")
                except Exception as exc:  # keep going so one failure does not lose every result
                    traceback.print_exc()
                    row["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)
                # Save after every stage so a crash or time-out keeps earlier results.
                save_json({"settings": settings, "results": rows}, out_path)

    print_table(rows)
    print(f"\nTotal time: {total.seconds / 60:.1f} min")
    save_json({"settings": settings, "results": rows, "total_seconds": total.seconds}, out_path)
    print(f"Saved {out_path.relative_to(out_path.parents[1])}")
    if args.quick:
        estimate_full_run(rows, args.split)


if __name__ == "__main__":
    main()
