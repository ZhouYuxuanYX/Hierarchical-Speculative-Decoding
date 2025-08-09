# Hierarchical-Speculative-Decoding
Hierarchical Speculative Decoding is the state-of-the-art lossless verification algorithm.

## HSD vs. EAGLE3 — MT-Bench Generation

This script runs a comparison between **EAGLE3** decoding and **our HSD (Hierarchical Speculative Decoding)** on MT-Bench–style prompts. In typical use, you only need to vary **`--temperature`**, **`--top_p`** (see note), and **`--hsd`**.

### What it does

- Loads a **base chat model** (e.g., Llama-3.1-Instruct-8B) and an **EAGLE3 head**.
- Optionally enables **HSD** during generation.
- Iterates over MT-Bench questions, generates answers (optionally multiple choices), and writes:
  - `..._ea.jsonl` (EAGLE3) or `..._ea_hsd.jsonl` (EAGLE3+HSD) with model outputs.
  - `..._info.jsonl` with timing and draft/accept stats per turn.


### Quick start

Baseline EAGLE3 (no HSD):
```bash
python -m eagle.evaluation.gen_ea_answer_llama3chat   --ea-model-path yuhuili/EAGLE3-LLaMA3.1-Instruct-8B   --base-model-path meta-llama/Llama-3.1-8B-Instruct --folder_id test1  --use_eagle3   --temperature 1.0
```

EAGLE3 + **HSD**:
```bash
python -m eagle.evaluation.gen_ea_answer_llama3chat   --ea-model-path yuhuili/EAGLE3-LLaMA3.1-Instruct-8B   --base-model-path meta-llama/Llama-3.1-8B-Instruct  --folder_id test1  --use_eagle3   --hsd   --temperature 1.0
```
---

### The three arguments you’ll typically change

- `--temperature` (float, default `1.0`)  
  Usual sampling temperature. Lower = safer outputs, often longer acceptance chains.

- `--top_p` (float) **(NOTE below)**  
  Nucleus sampling cutoff. Useful to study acceptance vs. diversity.

- `--hsd` (flag)  
  Turn **on** HSD (omit for EAGLE3 baseline). This toggles hierarchical drafting in `model.eagenerate(..., hsd=True)`.


### Outputs

- **Answers:**  
  `mt_bench/<folder_id>/<model_id>_ea.jsonl` (EAGLE3)  
  `mt_bench/<folder_id>/<model_id>_ea_hsd.jsonl` (EAGLE3+HSD)  
  Each line: `{question_id, answer_id, model_id, choices, tstamp}`

- **Run info (per question):**  
  `..._info.jsonl` with arrays for:
  `tokenizer_time`, `tokenizer_decode_time`, `generate_time`,  
  `processor_time`, `kv_time`, `reset_tree_time`, `tree_time`,  
  `eval_time`, `update_time`, `accept_length`, `draft_length`.

### Folder structure

```
EAGLE-main/
├─ eagle/
   └─ evaluation/
      └─ gen_ea_answer_llama3chat.py
   └─ data/
      └─ mt_bench/
          └─ question.jsonl
└─ mt_bench/
   └─ <folder_id>/
      ├─ <model_id>_ea.jsonl
      ├─ <model_id>_ea_info.jsonl
      ├─ <model_id>_ea_hsd.jsonl
      └─ <model_id>_ea_hsd_info.jsonl
```
