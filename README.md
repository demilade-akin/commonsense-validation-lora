# Commonsense Validation and Explanation with LoRA

CS4765/6765 Natural Language Processing project, University of New Brunswick, Fall 2026.

This project asks whether a small language model, fine-tuned cheaply with LoRA, can tell which statement defies common sense and pick the reason why. It uses the ComVE dataset from SemEval-2020 Task 4 and compares the proposed approach against prompting, a random baseline and fully fine-tuned encoder benchmarks.

## Tasks

* **Subtask A (validation):** given two similar statements, pick the one that does not make sense.
* **Subtask B (explanation):** given the nonsensical statement and three candidate reasons, pick the reason that explains why.

Both subtasks are multiple choice and use accuracy, the official metric of the shared task.

## Methods

| Method | Role | How it works |
|---|---|---|
| Random | Baseline | Picks an answer uniformly at random (fixed seed) |
| Zero-shot | Prompting | Qwen3-0.6B answers the question with no examples and no training |
| Few-shot | Prompting | Qwen3-0.6B sees solved training examples first (2 for A, 3 for B, letters balanced) |
| **LoRA** | **Proposed approach** | Qwen3-0.6B with low-rank adapters on `q_proj` and `v_proj`, trained on the task |
| RoBERTa-base | Benchmark | Full fine-tuning with a multiple-choice head |
| DistilBERT | Optional benchmark | Full fine-tuning with a multiple-choice head (not in the default run) |

**Letter scoring.** The language model never generates text. Each question lists its options with letters, and the model runs one forward pass. We compare its scores for the answer letters at the first position of its reply and pick the highest. This always gives a valid answer, needs no parsing and reads the model's answer the same way during training and evaluation.

**LoRA from scratch.** The course environment does not include the `peft` library, so `src/lora.py` implements LoRA directly in PyTorch. All base weights stay frozen. Each adapted layer adds two small matrices whose product is a trainable update, and the update starts at zero so the model begins exactly like the original.

## Project structure

```
run_all.py            Runs every method on both subtasks and reports accuracy and timing
src/
  config.py           All paths, model names, sample sizes and training settings
  data.py             Loads ComVE and joins questions with answers by id
  prompts.py          Turns examples into chat prompts, picks few-shot examples
  modeling.py         Loads Qwen3-0.6B and its tokenizer on CPU
  scoring.py          Letter scoring (shared by evaluation and LoRA training)
  prompting.py        Zero-shot and few-shot evaluation
  lora.py             LoRA layer, injection into the model, parameter counts
  train_lora.py       LoRA training loop and evaluation
  train_encoder.py    RoBERTa-base and DistilBERT multiple-choice fine-tuning
  baselines.py        Random baseline
  utils.py            Seeding, timing, accuracy, saving results
data/                 ComVE Subtask A and B files (train, dev, test)
results/              Output of run_all.py
```

## Setup

**On the course NLP VM**, activate the provided environment. It already includes every library the project needs.

```
source nlpfall2026venv/bin/activate
```

**On another machine**, create a Python 3.12 environment and install the same versions:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first run downloads Qwen3-0.6B (about 1.5 GB), RoBERTa-base and DistilBERT from the HuggingFace Hub. If the machine cannot reach the Hub, download the models elsewhere, copy the folders over and set their paths in `src/config.py` (`MODEL_NAME` and `ENCODER_MODELS`).

## How to run

Run every command from the repository root.

```
python run_all.py --quick     # tiny run: checks every stage and estimates the full run time
python run_all.py             # full run, reported results on the test split (about 35 min on the course VM)
python run_all.py --split dev # full run on the dev split
```

Run a subset with `--methods` and `--subtasks`, for example:

```
python run_all.py --methods random,zero-shot,lora --subtasks A
```

The script prints a results table and saves every number, setting and timing to `results/results.json` (`results/results_quick.json` for `--quick`). It saves after every stage, so a crash keeps the results finished so far.

### Fitting the one-hour limit

The full run must finish within one hour on the course VM (4 CPU cores, 5.8 GB RAM). With the settings in `src/config.py`, the default run took 35 minutes there: 400 test questions per subtask, 400 LoRA training examples and 2,000 RoBERTa training examples per subtask. Close other programs, such as the browser, before running: when memory runs short the VM swaps to disk and training slows down several times.

`python run_all.py --quick` measures the speed of every stage on the current machine and prints an estimate of the full run. If the estimate is too long, lower `train_n` in `LoRASettings` and `EncoderSettings` or `EVAL_N` in `src/config.py`.

## Reproducibility

Every random choice (sampling, shuffling, few-shot examples, adapter and head initialisation, the random baseline) uses the seed in `src/config.py`. The project runs on CPU in float32, so the same settings give the same numbers on every run.

## Results

To be filled in after the final run on the course VM.

## Data and model licenses

* **ComVE** (Wang et al., 2020): CC BY-SA 4.0. The official files are included unchanged in `data/`. Source: https://github.com/wangcunxiang/SemEval2020-Task4-Commonsense-Validation-and-Explanation
* **Qwen3-0.6B**: Apache 2.0
* **RoBERTa-base**: MIT
* **DistilBERT**: Apache 2.0

## Team

Akinbobola Akin, Okejimi Faridah, Adejuwon Oluwasegun Isaac
