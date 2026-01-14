"""
Combined Performance Evaluation Script for EAGLE.

Evaluates:
- ACC: Accuracy on GSM8K (or other benchmarks)
- BE: Block Efficiency (average accepted tokens per step)
- DS: Decoding Speed (tokens per second)

Usage:
    python performance_evaluation.py --answer-file gsm8k/folder/model_ea.jsonl
    python performance_evaluation.py --answer-file gsm8k/folder/model_ea.jsonl --question-file eagle/data/gsm8k/question.jsonl
"""

import argparse
import orjson
import json
import numpy as np
import os
import re
import time


# ============================================================
# Accuracy Evaluation (ACC)
# ============================================================

def extract_answer(text):
    """Extract the final numerical answer from text."""
    pattern = r'\d*\.?\d+'
    numbers = re.findall(pattern, text.replace(',', ''))
    if len(numbers) >= 1:
        return numbers[-1]
    return None


def extract_gold_answer(reference_text):
    """Extract the gold answer from GSM8K reference (#### format)."""
    match = re.search(r'####\s*(-?\d+\.?\d*)', reference_text)
    if match:
        return match.group(1)
    return extract_answer(reference_text)


def test_answer(pred_str, ans_str):
    """Test if the predicted answer matches the gold answer."""
    pred = extract_answer(pred_str)
    gold = extract_gold_answer(ans_str)
    
    if pred is None or gold is None:
        return False
    
    if pred == gold:
        return True
    
    try:
        pred_num = float(pred)
        gold_num = float(gold)
        return abs(pred_num - gold_num) < 1e-6
    except ValueError:
        return False


def load_gsm8k_questions(question_file=None):
    """Load GSM8K questions with references."""
    questions = {}
    
    # Always load from HuggingFace for consistency with gen_ea_answer_llama3chat.py
    print("Loading GSM8K questions from Hugging Face...")
    from datasets import load_dataset
    dataset = load_dataset('openai/gsm8k', 'main')
    gsm8k_test = dataset['test']
    print(f"Loaded {len(gsm8k_test)} questions from GSM8K test set")
    for idx in range(len(gsm8k_test['question'])):
            questions[idx] = {
                'question_id': idx,
                'turns': [gsm8k_test['question'][idx]],
                'reference': [gsm8k_test['answer'][idx]]
            }
    
    return questions


def evaluate_accuracy(answer_file, question_file=None):
    """Evaluate accuracy of model predictions."""
    questions = load_gsm8k_questions(question_file)
    
    results = []
    with open(answer_file, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    
    num_correct = 0
    num_total = 0
    
    for result in results:
        question_id = result['question_id']
        
        if question_id not in questions:
            continue
        
        question = questions[question_id]
        reference = question.get('reference', [''])[0]
        
        if result['choices'] and result['choices'][0]['turns']:
            prediction = result['choices'][0]['turns'][0]
        else:
            prediction = ""
        
        is_correct = test_answer(prediction, reference)
        num_total += 1
        if is_correct:
            num_correct += 1
    
    accuracy = num_correct / num_total if num_total > 0 else 0.0
    return accuracy, num_correct, num_total


# ============================================================
# Block Efficiency (BE) and Decoding Speed (DS) Evaluation
# ============================================================

def evaluate_efficiency_and_speed(info_file, answer_file, gamma=7):
    """
    Evaluate Block Efficiency and Decoding Speed from info file and answer file.
    
    Uses the same evaluation logic as eval_speculative_decoding_llm_gsm8k.py:
    - Only counts steps where draft_length == gamma (full draft steps)
    - BE = sum(accept_length where draft==gamma) / count(steps where draft==gamma)
    - DS = count(full_draft_steps) / total_time * gamma
    
    Args:
        info_file: Path to the _info.jsonl file
        answer_file: Path to the answer jsonl file
        gamma: Max draft length (default 7 for EAGLE)
    
    Returns:
        be: Block Efficiency (average accepted tokens per step)
        ds: Decoding Speed (tokens per second)
        avg_time: Average time per sample
    """
    if not os.path.exists(info_file):
        return None, None, None
    
    with open(info_file, "r") as f:
        lines = [orjson.loads(line) for line in f if line.strip()]
    
    # Accumulate stats matching eval_speculative_decoding_llm_gsm8k.py
    sample = 0  # sum of accept_length where draft_length == gamma
    len_ = 0    # count of steps where draft_length == gamma
    total_time = 0.0
    
    for line_data in lines:
        # Handle nested structure: [[val1, val2, ...]]
        draft_lengths = line_data.get("draft_length", [[]])
        accept_lengths = line_data.get("accept_length", [[]])
        generate_time = line_data.get("generate_time", [0])
        
        # Flatten if nested
        if draft_lengths and isinstance(draft_lengths[0], list):
            draft_lengths = draft_lengths[0]
        if accept_lengths and isinstance(accept_lengths[0], list):
            accept_lengths = accept_lengths[0]
        if isinstance(generate_time, list):
            generate_time = generate_time[0]
        
        # Convert to numpy arrays for filtering
        draft_arr = np.array(draft_lengths)
        accept_arr = np.array(accept_lengths)
        
        # Only count steps where draft_length == gamma (full draft steps)
        # This matches the logic: sample += sample_list[draft_list==gamma].sum()
        sample += accept_arr[draft_arr == gamma].sum()
        len_ += len(accept_arr[draft_arr == gamma])
        
        total_time += float(generate_time)
    
    # Calculate Block Efficiency: average accepted tokens per full-draft step
    # BE = sample / len_ (same as sample_length in eval_speculative_decoding)
    be = sample / len_ if len_ > 0 else 0.0
    
    # Calculate Decoding Speed: DS = len_ / time_ * gamma
    # This is the effective throughput based on full-draft steps
    ds = (len_ / total_time * gamma) if total_time > 0 else 0.0
    
    avg_time = total_time / len(lines) if lines else 0.0
    
    return be, ds, avg_time


def evaluate_baseline_speed(info_file):
    """
    Evaluate Decoding Speed for baseline (no EAGLE).
    
    For baseline, we calculate speed from generate_time and new_tokens in answer file.
    """
    if not os.path.exists(info_file):
        return None, None
    
    with open(info_file, "r") as f:
        lines = [orjson.loads(line) for line in f if line.strip()]
    
    total_time = 0.0
    
    for line_data in lines:
        generate_time = line_data.get("generate_time", [0])
        if isinstance(generate_time, list):
            generate_time = generate_time[0]
        total_time += float(generate_time)
    
    avg_time = total_time / len(lines) if lines else 0.0
    return total_time, avg_time


def get_total_tokens_from_answer(answer_file):
    """Get total generated tokens from answer file."""
    total_tokens = 0
    with open(answer_file, 'r') as f:
        for line in f:
            result = json.loads(line)
            if result['choices'] and result['choices'][0].get('new_tokens'):
                total_tokens += sum(result['choices'][0]['new_tokens'])
    return total_tokens


# ============================================================
# Main Evaluation
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Combined Performance Evaluation (ACC, BE, DS)")
    parser.add_argument(
        "--answer-file",
        type=str,
        required=True,
        help="Path to the answer JSONL file (e.g., model_ea.jsonl)"
    )
    parser.add_argument(
        "--question-file",
        type=str,
        default='eagle/data/gsm8k/question.jsonl',
        help="Path to the question JSONL file (optional, loads from HF if not provided)"
    )
    parser.add_argument(
        "--gamma",
        type=int,
        default=7,
        help="Max draft length for EAGLE (default: 7)"
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="gsm8k",
        help="Benchmark name (default: gsm8k)"
    )
    
    args = parser.parse_args()
    
    answer_file = args.answer_file
    info_file = answer_file.replace(".jsonl", "_info.jsonl")
    
    print("=" * 70)
    print("Combined Performance Evaluation")
    print("=" * 70)
    print(f"Answer file: {answer_file}")
    print(f"Info file:   {info_file}")
    print(f"Benchmark:   {args.bench_name}")
    print("=" * 70)
    
    # 1. Evaluate Accuracy (ACC)
    print("\n[1] Evaluating Accuracy (ACC)...")
    if args.bench_name == "gsm8k":
        accuracy, num_correct, num_total = evaluate_accuracy(answer_file, args.question_file)
        print(f"    Correct: {num_correct}/{num_total}")
    else:
        accuracy = None
        print("    Skipping accuracy evaluation (not gsm8k)")
    
    # 2. Evaluate Block Efficiency (BE) and Decoding Speed (DS)
    print("\n[2] Evaluating Block Efficiency (BE) and Decoding Speed (DS)...")
    
    is_baseline = "_baseline" in answer_file
    
    if is_baseline:
        # Baseline evaluation (no draft/accept lengths)
        total_time, avg_time = evaluate_baseline_speed(info_file)
        total_tokens = get_total_tokens_from_answer(answer_file)
        
        be = 1.0  # Baseline always accepts 1 token per step
        ds = total_tokens / total_time if total_time and total_time > 0 else 0.0
        print(f"    Mode: Baseline (no speculative decoding)")
    else:
        be, ds, avg_time = evaluate_efficiency_and_speed(info_file, answer_file, args.gamma)
        print(f"    Mode: EAGLE (speculative decoding, gamma={args.gamma})")
    
    # 3. Print Summary
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    if accuracy is not None:
        print(f"  ACC (Accuracy):           {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    if be is not None:
        print(f"  BE  (Block Efficiency):   {be:.4f} tokens/step")
    
    if ds is not None:
        print(f"  DS  (Decoding Speed):     {ds:.2f} tokens/s")
    
    if avg_time is not None:
        print(f"  Avg Time per Sample:      {avg_time:.4f} s")
    
    print("=" * 70)
    
    # Return results for programmatic use
    return {
        "accuracy": accuracy,
        "block_efficiency": be,
        "decoding_speed": ds,
        "avg_time": avg_time
    }


if __name__ == "__main__":
    main()



