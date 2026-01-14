"""
Baseline evaluation for Llama-3.1-8B-Instruct on GSM8K.

This script runs the base model without EAGLE speculative decoding for comparison.

Usage:
    python baselin.py --gsm8k-subset --gsm8k-fewshot
"""
import argparse
import json
import os
import time
from datetime import datetime

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import shortuuid
from fastchat.llm_judge.common import load_questions

# Set seed for reproducibility
from accelerate.utils import set_seed
set_seed(0)


def load_gsm8k_fewshot_prompt():
    """Load the few-shot prompt from prompt_original.txt for GSM8K."""
    prompt_file = "EAGLE-main/eagle/data/gsm8k/prompt_original.txt"
    if os.path.exists(prompt_file):
        with open(prompt_file, 'r') as f:
            return f.read().strip()
    return ""


def load_gsm8k_questions(question_begin=None, question_end=None, use_subset=False):
    """Load questions from GSM8K dataset via Hugging Face."""
    print("Loading GSM8K dataset from Hugging Face...")
    dataset = load_dataset('openai/gsm8k', 'main', trust_remote_code=True)
    gsm8k_test = dataset['test']
    
    total_questions = len(gsm8k_test['question'])
    print(f"Found {total_questions} questions in GSM8K test set")
    
    if use_subset:
        num_samples = total_questions // 10
        print(f"Using subset: first {num_samples} questions (10%)")
    else:
        num_samples = total_questions
    
    start_idx = question_begin if question_begin is not None else 0
    end_idx = question_end if question_end is not None else num_samples
    end_idx = min(end_idx, num_samples)
    
    questions = []
    for idx in range(start_idx, end_idx):
        question_text = gsm8k_test['question'][idx]
        answer_text = gsm8k_test['answer'][idx]
        
        questions.append({
            "question_id": idx,
            "category": "math",
            "turns": [question_text],
            "reference": [answer_text]
        })
    
    print(f"Loaded {len(questions)} questions (indices {start_idx} to {end_idx-1})")
    return questions


def load_flores200_questions(question_begin=None, question_end=None, source_lang="eng_Latn", target_lang="fra_Latn"):
    """Load questions from flores200_devtest_translation_pairs dataset.
    
    Args:
        question_begin: Starting index for questions (optional)
        question_end: Ending index for questions (optional)
        source_lang: Source language code (default: "eng_Latn" for English)
        target_lang: Target language code (default: "fra_Latn" for French)
    
    Returns:
        List of questions in the expected format with question_id, category, turns, and reference
    """
    print(f"Loading flores200 dataset with {source_lang} -> {target_lang}...")
    dataset = load_dataset("bri25yu/flores200_devtest_translation_pairs", split="devtest")
    
    # Filter dataset for the specified language pair using efficient filtering
    print(f"Filtering for language pair...")
    filtered_dataset = dataset.filter(
        lambda x: x['source_lang'] == source_lang and x['target_lang'] == target_lang,
        num_proc=4
    )
    print(f"Found {len(filtered_dataset)} translation pairs for {source_lang} -> {target_lang}")
    
    # Apply question_begin and question_end
    start_idx = question_begin if question_begin is not None else 0
    end_idx = question_end if question_end is not None else len(filtered_dataset)
    
    questions = []
    for idx in range(start_idx, min(end_idx, len(filtered_dataset))):
        item = filtered_dataset[idx]
        source_text = item['source']
        target_text = item['target']
        
        # Format as a translation task with clear instruction
        source_lang_name = source_lang.split('_')[0].capitalize()
        target_lang_name = target_lang.split('_')[0].capitalize()
        prompt = f"Translate the following {source_lang_name} text to {target_lang_name}. Only provide the translation without any explanation or additional text:\n\n{source_text}"
        
        questions.append({
            "question_id": idx,
            "category": "translation",
            "turns": [prompt],
            "reference": [target_text]
        })
    
    return questions


def get_system_prompt(bench_name):
    """Get appropriate system prompt based on benchmark type."""
    if bench_name == "flores200":
        return "You are a helpful translation assistant. Provide accurate and natural translations."
    else:
        return "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."


@torch.inference_mode()
def run_baseline_eval(args):
    """Run baseline evaluation without EAGLE."""
    
    # Load model and tokenizer
    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    
    # Load questions based on benchmark type
    if args.bench_name == "flores200":
        source_lang = getattr(args, 'source_lang', 'eng_Latn')
        target_lang = getattr(args, 'target_lang', 'fra_Latn')
        questions = load_flores200_questions(args.question_begin, args.question_end, source_lang, target_lang)
    elif args.bench_name == "gsm8k":
        # Load GSM8K from HuggingFace for consistency with gen_ea_answer_llama3chat.py
        questions = load_gsm8k_questions(args.question_begin, args.question_end, use_subset=args.gsm8k_subset)
    else:
        # Load from question file using fastchat
        script_dir = os.path.dirname(__file__)
        parent_dir = os.path.dirname(script_dir)
        question_file = f"{parent_dir}/data/{args.bench_name}/question.jsonl"
        questions = load_questions(question_file, args.question_begin, args.question_end)
    
    # Load few-shot prompt if enabled (only for gsm8k)
    gsm8k_fewshot_prompt = ""
    if args.gsm8k_fewshot and args.bench_name == "gsm8k":
        gsm8k_fewshot_prompt = load_gsm8k_fewshot_prompt()
        if gsm8k_fewshot_prompt:
            print(f"Loaded GSM8K few-shot prompt ({len(gsm8k_fewshot_prompt)} chars)")
    
    # Setup output file
    folder_id = args.folder_id
    os.makedirs(f"{args.bench_name}/{folder_id}", exist_ok=True)
    answer_file = f"{args.bench_name}/{folder_id}/{args.model_id}-temperature-{args.temperature}_baseline.jsonl"
    info_file = answer_file.replace(".jsonl", "_info.jsonl")
    
    print(f"Output to {answer_file}")
    print(f"Info output to {info_file}")
    
    # Warmup
    print("Warming up...")
    question = questions[0]
    for _ in range(3):
        torch.manual_seed(0)
        messages = [
            {"role": "system", "content": get_system_prompt(args.bench_name)},
        ]
        qs = question["turns"][0]
        if gsm8k_fewshot_prompt:
            qs = gsm8k_fewshot_prompt + "\n\nQuestion: " + qs + "\nLet's think step by step"
        messages.append({"role": "user", "content": qs})
        
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = tokenizer([prompt], add_special_tokens=False, return_tensors="pt").input_ids.cuda()
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=args.max_new_token,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.temperature > 1e-5,
                pad_token_id=tokenizer.eos_token_id,
            )
    print("Warmup done")
    
    # Main evaluation loop
    stop_token_ids = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    for question in tqdm(questions):
        torch.manual_seed(0)
        
        messages = [
            {"role": "system", "content": get_system_prompt(args.bench_name)},
        ]
        
        qs = question["turns"][0]
        if gsm8k_fewshot_prompt:
            qs = gsm8k_fewshot_prompt + "\n\nQuestion: " + qs + "\nLet's think step by step"
        messages.append({"role": "user", "content": qs})
        
        # Tokenize
        time_start = time.perf_counter()
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = tokenizer([prompt], add_special_tokens=False, return_tensors="pt").input_ids.cuda()
        input_len = input_ids.shape[1]
        time_end = time.perf_counter()
        tokenizer_time = time_end - time_start
        
        # Generate
        torch.cuda.synchronize()
        start_time = time.time()
        time_start = time.perf_counter()
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=args.max_new_token,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.temperature > 1e-5,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        torch.cuda.synchronize()
        total_time = time.time() - start_time
        time_end = time.perf_counter()
        generate_time = time_end - time_start
        
        # Process output
        output_ids = output_ids[0][input_len:]
        new_token = len(output_ids)
        
        # Truncate at stop tokens
        for i, id in enumerate(output_ids):
            if id in stop_token_ids:
                output_ids = output_ids[:i]
                break
        
        # Decode
        time_start = time.perf_counter()
        output = tokenizer.decode(output_ids, spaces_between_special_tokens=False)
        time_end = time.perf_counter()
        decode_time = time_end - time_start
        
        # Clean up special tokens
        for special_token in tokenizer.special_tokens_map.values():
            if isinstance(special_token, list):
                for special_tok in special_token:
                    output = output.replace(special_tok, "")
            else:
                output = output.replace(special_token, "")
        output = output.strip()
        
        # Save answer
        ans_json = {
            "question_id": question["question_id"],
            "answer_id": shortuuid.uuid(),
            "model_id": args.model_id,
            "choices": [{
                "index": 0,
                "turns": [output],
                "idxs": [0],
                "new_tokens": [new_token],
                "wall_time": [total_time]
            }],
            "tstamp": time.time(),
        }
        
        with open(answer_file, "a") as fout:
            fout.write(json.dumps(ans_json) + "\n")
        
        # Save timing info
        info_json = {
            "tokenizer_time": [tokenizer_time],
            "tokenizer_decode_time": [decode_time],
            "generate_time": [generate_time],
        }
        
        with open(info_file, "a") as fout:
            fout.write(json.dumps(info_json) + "\n")
    
    print(f"\nEvaluation complete!")
    print(f"Results saved to: {answer_file}")
    
    # Reorganize answer file (sort by question_id, deduplicate)
    reorg_answer_file(answer_file)


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Path to the model.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="llama38b2_40",
        help="Model ID for output file naming.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="Beginning index of questions.",
    )
    parser.add_argument(
        "--question-end",
        type=int,
        help="Ending index of questions.",
    )
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p (nucleus) sampling parameter.",
    )
    parser.add_argument(
        "--gsm8k-subset",
        action="store_true",
        help="Use only first 10% of GSM8K questions.",
    )
    parser.add_argument(
        "--gsm8k-fewshot",
        action="store_true",
        help="Use few-shot prompting for GSM8K.",
    )
    parser.add_argument(
        "--folder-id",
        type=str,
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Folder ID for output.",
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="gsm8k",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--source-lang",
        type=str,
        default="eng_Latn",
        help="Source language code for flores200 translation (e.g., eng_Latn for English)",
    )
    parser.add_argument(
        "--target-lang",
        type=str,
        default="deu_Latn",
        help="Target language code for flores200 translation (e.g., fra_Latn for French)",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Baseline Evaluation Configuration")
    print("=" * 60)
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("=" * 60)
    
    run_baseline_eval(args)
