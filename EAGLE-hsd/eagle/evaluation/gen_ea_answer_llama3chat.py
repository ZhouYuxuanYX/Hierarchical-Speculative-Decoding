"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
from datetime import datetime
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from accelerate.utils import set_seed
set_seed(0)

import time

import shortuuid
from fastchat.llm_judge.common import load_questions
from tqdm import tqdm
from datasets import load_dataset

try:
    from ..model.ea_model import EaModel
    from ..model.kv_cache import initialize_past_key_values
    from ..model.utils import *
except:
    from eagle.model.ea_model import EaModel
    from eagle.model.kv_cache import initialize_past_key_values
    from eagle.model.utils import *


def get_system_prompt(bench_name):
    """Get appropriate system prompt based on benchmark type."""
    if bench_name == "flores200":
        return "You are a helpful translation assistant. Provide accurate and natural translations."
    else:
        return "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."


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



def run_eval(
        base_model_path,
        ea_model_path,
        model_id,
        question_file,
        question_begin,
        question_end,
        answer_file,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        num_gpus_total,
        max_gpu_memory,
        temperature,
        args
):
    # Load questions based on benchmark type
    if args.bench_name == "flores200":
        source_lang = getattr(args, 'source_lang', 'eng_Latn')
        target_lang = getattr(args, 'target_lang', 'fra_Latn')
        questions = load_flores200_questions(question_begin, question_end, source_lang, target_lang)
    else:
        questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    # random.shuffle(questions)
    shuffled_ids = [q["question_id"] for q in questions]
    # with open(f"data/{args.bench_name}/model_ids/{args.model_id}.shuffled_ids", "w") as fout:
    #     json.dump(shuffled_ids, fout)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)  # // 2
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                base_model_path,
                ea_model_path,
                model_id,
                questions[i: i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                temperature,
                args
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
        base_model_path,
        ea_model_path,
        model_id,
        questions,
        answer_file,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        max_gpu_memory,
        temperature,
        args
):
    # temperature = 0.0

    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        # load_in_8bit=True,
        device_map="auto",
        use_eagle3=args.use_eagle3,
    )

    tokenizer = model.get_tokenizer()

    if temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=temperature)
    else:
        logits_processor = None

    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    question = questions[0]

    # warmup
    for _ in range(3):
        torch.manual_seed(0)

        messages = [
            {"role": "system",
             "content": get_system_prompt(args.bench_name)},
        ]
        turns = []
        idxs = []
        new_tokens = []
        wall_time = []
        for j in range(len(question["turns"])):
            qs = question["turns"][j]
            messages.append({
                "role": "user",
                "content": qs
            })
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_ids = tokenizer([prompt],add_special_tokens=False,).input_ids

            # try:
            torch.cuda.synchronize()
            start_time = time.time()

            output_ids, new_token, idx, _ = model.eagenerate(
                torch.as_tensor(input_ids).cuda(),
                temperature=temperature,
                log=True,
                is_llama3=True,
                hsd=args.hsd
            )
            torch.cuda.synchronize()
            total_time = time.time() - start_time
            output_ids = output_ids[0][len(input_ids[0]):]
            # be consistent with the template's stop_token_ids
            stop_token_ids = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            if stop_token_ids:
                stop_token_ids_index = [
                    i
                    for i, id in enumerate(output_ids)
                    if id in stop_token_ids
                ]
                if len(stop_token_ids_index) > 0:
                    output_ids = output_ids[: stop_token_ids_index[0]]

            output = tokenizer.decode(
                output_ids,
                spaces_between_special_tokens=False,
            )
            # stop_str = "</s>"
            # if stop_str and output.find(stop_str) > 0:
            #     output = output[: output.find(stop_str)]
            for special_token in tokenizer.special_tokens_map.values():
                if isinstance(special_token, list):
                    for special_tok in special_token:
                        output = output.replace(special_tok, "")
                else:
                    output = output.replace(special_token, "")
            output = output.strip()



            turns.append(output)
            idxs.append(int(idx))
            new_tokens.append(int(new_token))
            wall_time.append(total_time)
            messages.append({
                "role": "assistant",
                "content": output
            })
    print('Warmup done')


    # questions=questions[6:]
    for question in tqdm(questions):

        choices = []
        question_info = []
        for i in range(num_choices):

            torch.manual_seed(i)
            messages = [
                {"role": "system",
                 "content": get_system_prompt(args.bench_name)},
            ]
            turns = []
            idxs = []
            new_tokens = []
            wall_time = []
            total_info = {'tokenizer_time': [], 'tokenizer_decode_time': [], 'generate_time': [], 'processor_time':[], 'kv_time':[],'reset_tree_time':[], 
                       'tree_time':[], 'eval_time':[], 'update_time':[], 'accept_length':[], 'draft_length':[]}
            for j in range(len(question["turns"])):
                
                qs = question["turns"][j]
                messages.append({
                    "role": "user",
                    "content": qs
                })
                time_start = time.perf_counter()
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                input_ids = tokenizer([prompt], add_special_tokens=False, ).input_ids
                time_end = time.perf_counter()
                total_info['tokenizer_time'].append(time_end - time_start)

                # try:
                torch.cuda.synchronize()
                start_time = time.time()
                time_start = time.perf_counter()

                output_ids, new_token, idx, return_info = model.eagenerate(
                    torch.as_tensor(input_ids).cuda(),
                    temperature=temperature,
                    log=True,
                    is_llama3=True,
                )
                torch.cuda.synchronize()
                total_time = time.time() - start_time
                time_end = time.perf_counter()
                total_info['generate_time'].append(time_end - time_start)

                output_ids = output_ids[0][len(input_ids[0]):]
                # be consistent with the template's stop_token_ids
                stop_token_ids = [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]

                if stop_token_ids:
                    stop_token_ids_index = [
                        i
                        for i, id in enumerate(output_ids)
                        if id in stop_token_ids
                    ]
                    if len(stop_token_ids_index) > 0:
                        output_ids = output_ids[: stop_token_ids_index[0]]
                time_start = time.perf_counter()
                output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                )
                time_end = time.perf_counter()
                total_info['tokenizer_decode_time'].append(time_end - time_start)
                # stop_str = "</s>"
                # if stop_str and output.find(stop_str) > 0:
                #     output = output[: output.find(stop_str)]
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

                turns.append(output)
                idxs.append(int(idx))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                messages.append({
                    "role": "assistant",
                    "content": output
                })
                total_info['processor_time'].append(return_info['processor_time'])
                total_info['kv_time'].append(return_info['kv_time'])
                total_info['reset_tree_time'].append(return_info['reset_tree_time'])
                total_info['tree_time'].append(return_info['tree_time'])
                total_info['eval_time'].append(return_info['eval_time'])
                total_info['update_time'].append(return_info['update_time'])
                total_info['accept_length'].append(return_info['accept_length'])
                total_info['draft_length'].append(return_info['draft_length'])


            # torch.cuda.empty_cache()
            choices.append({"index": i, "turns": turns, "idxs": idxs, "new_tokens": new_tokens, "wall_time": wall_time})
            question_info.append({"tokenizer_time":total_info['tokenizer_time'],
                                   "tokenizer_decode_time":total_info['tokenizer_decode_time'], 
                                   "generate_time":total_info['generate_time'],
                                   "processor_time":total_info['processor_time'],
                                   "kv_time":total_info['kv_time'],
                                   "reset_tree_time":total_info['reset_tree_time'],
                                   "tree_time":total_info['tree_time'],
                                   "eval_time":total_info['eval_time'],
                                   "update_time":total_info['update_time'],
                                   "accept_length":total_info['accept_length'],
                                   "draft_length":total_info['draft_length'],})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")

        os.makedirs(os.path.dirname(info_file), exist_ok=True)
        with open(os.path.expanduser(info_file), "a") as fout:
            for info in question_info:
                fout.write(json.dumps(info) + "\n")

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
        "--ea-model-path",
        type=str,
        default="/home/lyh/weights/hf/eagle3/llama31chat/8B/",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--base-model-path", type=str, default="/home/lyh/weights/hf/llama31chat/8B/",
                        help="1")
    parser.add_argument(
        "--load-in-8bit", action="store_false", help="Use 8-bit quantization"
    )
    parser.add_argument("--model-id", type=str, default="llama38b2_40")
    parser.add_argument(
        "--bench-name",
        type=str,
        default="gsm8k",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--total-token",
        type=int,
        default=60,
        help="total-token = The total number of drafted tokens in the tree + 1",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=5,
        help="depth = The maximum number of draft length - 1",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="The maximum number of drafted tokens in each layer.",
    )

    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--tree-choices",
        type=str,
        default="mc_sim_7b_63",
    )
    parser.add_argument(
        "--use_eagle3",
        action="store_true"
    )

    parser.add_argument(
        "--hsd",
        action="store_true"
    )
    parser.add_argument(
        "--folder_id",
        type=str,
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
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

    # eagle: hsd + not hsd  temperate=1 top_p=1, top_k=0

    args = parser.parse_args()

    for k,v in vars(args).items():
        print(f"{k}={v}")

    args.model_id = args.model_id + "-temperature-" + str(args.temperature)
    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    question_file = f"{parent_dir}/data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        if args.use_eagle3:
            if args.hsd:
                answer_file = f"{args.bench_name}/{args.folder_id}/{args.model_id}_ea_hsd.jsonl"
            else:
                answer_file = f"{args.bench_name}/{args.folder_id}/{args.model_id}_ea.jsonl"
        else:
            answer_file = f"{args.bench_name}/{args.folder_id}/{args.model_id}_baseline.jsonl"

    print(f"Output to {answer_file}")
    info_file = answer_file.replace(".jsonl", "_info.jsonl")
    print(f"info Output to {info_file}")

    run_eval(
        args.base_model_path,
        args.ea_model_path,
        args.model_id,
        question_file,
        args.question_begin,
        args.question_end,
        answer_file,
        args.max_new_token,
        args.num_choices,
        args.num_gpus_per_model,
        args.num_gpus_total,
        args.max_gpu_memory,
        args.temperature,
        args
    )

    reorg_answer_file(answer_file)
