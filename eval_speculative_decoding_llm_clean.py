import re
import os
# !!!!!!!!!!!!must set the environment variable before importing transformers, otherwise it won't work!!!!!!!!
######### use the local cache on haicore
# os.environ['HF_HOME'] = '/home/c02yuzh/CISPA-projects/rectified_softmax_ml-2023/hf_home'
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
from tqdm import tqdm
from torch import LongTensor, FloatTensor, eq, device
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, MaxLengthCriteria, StoppingCriteria, StoppingCriteriaList
import datasets
from accelerate import dispatch_model
from torch import nn
import numpy as np
import torch
import json
import argparse
import numpy as np
import time
import random
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download
import cProfile, pstats, io


def argparse_setup():
    parser = argparse.ArgumentParser(prog='myprogram')

    # Clean: remove backward/clever/blockwise from CLI
    parser.add_argument('--speculative', action='store_true', default=False)
    parser.add_argument('--HSD', action='store_true', default=False, help='Run our HSD algorithm.')
    parser.add_argument('--naive', action='store_true', default=False, help='Run naive speculative baseline.')

    parser.add_argument('--multidraft', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--top_p', type=float, default=1)
    parser.add_argument('--parallel', action='store_true', default=False)
    parser.add_argument('--gamma',  default=10, type=int, help='number of assisted tokens')
    parser.add_argument("--approxi", action='store_true', default=False)   # kept; applies to HSD if needed
    parser.add_argument('--model', help='must be target or draft', default="target")
    parser.add_argument('--prompt',  default='original', help='must be complex or original')
    parser.add_argument('--target-model',  default='Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8', help='target HF model id')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--name', type=str, default='', help='additional name to distinguish different runs')
    args = parser.parse_args()
    print(args)
    return args

def tp(name, x):
    """Print name, type, and a short summary (no big dumps)."""
    import numpy as np
    try:
        import torch
    except Exception:
        torch = None

    def brief(v):
        if torch is not None and isinstance(v, torch.Tensor):
            return f"Tensor(shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device})"
        if isinstance(v, np.ndarray):
            return f"ndarray(shape={v.shape}, dtype={v.dtype})"
        if isinstance(v, dict):
            return f"dict(len={len(v)}, keys={list(v)[:3]})"
        if isinstance(v, (list, tuple)):
            head = v[:3]
            return f"{type(v).__name__}(len={len(v)}, head={head})"
        return repr(v)
    print(f"{name}: <{type(x).__name__}> {brief(x)}")

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids
    def __call__(self, input_ids: LongTensor, scores: FloatTensor, **kwargs) -> bool:
        for stop_ids in self.stop_token_ids:
            if (input_ids[0][-len(stop_ids[0]):] == stop_ids[0]).all():
                print("input tail ids:", input_ids[0][-len(stop_ids[0]):])
                print("stop ids:", stop_ids[0])
                return True
        return False

def flatten_hist_lengths(hist_lengths_list):
    # list-of-lists -> flat list; skip each inner leading 0
    return [l for hl in hist_lengths_list for l in hl[1:]]

def kv_bytes_for_chunk(cfg, L_i, a_i, batch=1, dtype_bytes=2):
    """
    Closed-form KV bytes for a verification chunk that accepts a_i tokens,
    starting from sequence length L_i.
    """
    H   = getattr(cfg, "num_attention_heads")
    H_kv = getattr(cfg, "num_key_value_heads", H)  # GQA-aware
    D   = cfg.hidden_size
    Dh  = D // H
    layers = cfg.num_hidden_layers
    factor_1 = (2 * a_i * L_i) + (a_i * (a_i - 1)) 
    factor_2 = (2 * a_i)
    factor = factor_1 + factor_2

    return layers * batch * H_kv * Dh * dtype_bytes * factor, factor_1, factor_2

def kv_time_ms_target(cfg, prefix_len, per_verify_lengths, batch, dtype_bytes, bw_bytes_per_s):
    L = prefix_len
    total_bytes = 0
    f1_append = []
    f2_append = []
    for a in per_verify_lengths:
        bytes_, f1, f2 = kv_bytes_for_chunk(cfg, L_i=L, a_i=a, batch=batch, dtype_bytes=dtype_bytes)
        total_bytes += bytes_
        f1_append.append(f1)
        f2_append.append(f2)
    return (total_bytes / max(1.0, bw_bytes_per_s)) * 1000.0, f1_append, f2_append

def effective_bandwidth_Bps():
    import torch, time
    if not torch.cuda.is_available():
        # No CUDA: return NaN so you don't divide by zero; skip KV split on CPU/MPS
        return float("nan")
    x = torch.empty((1024,1024,512), device="cuda", dtype=torch.float16)  # ~1GB
    torch.cuda.synchronize(); t0 = time.time()
    _ = x.clone(); torch.cuda.synchronize(); t1 = time.time()
    return x.numel()*x.element_size()/(t1 - t0)

def move_rotary_emb_to_device(model):
    # Get the device of the layer where rotary embedding is applied
    try:
        device_ = model.model.embed_tokens.weight.device
        if hasattr(model.model, "rotary_emb") and hasattr(model.model.rotary_emb, "inv_freq"):
            model.model.rotary_emb.inv_freq = model.model.rotary_emb.inv_freq.to(device_)
    except AttributeError:
        print("Could not move rotary_emb.inv_freq — structure may be different")

def manual_device_map(model, same_device_for_input_output=True):
    """
    Manually create a balanced device map for a Hugging Face transformer model.
    """
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        raise RuntimeError("No CUDA GPUs available")

    # Find model layers
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h
        prefix = 'transformer.h'
        embedding_key = 'transformer.wte'
        norm_key = 'transformer.ln_f'
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
        prefix = 'model.layers'
        embedding_key = 'model.embed_tokens'
        norm_key = 'model.norm'
    else:
        raise ValueError("Unknown model structure")

    num_layers = len(layers)
    layers_per_gpu = (num_layers + n_gpus - 1) // n_gpus
    device_map = {}

    # Distribute transformer layers
    for i in range(num_layers):
        gpu_id = i // layers_per_gpu
        key = f"{prefix}.{i}"
        device_map[key] = f"cuda:{gpu_id}"

    # Assign special components
    if same_device_for_input_output:
        device_map[embedding_key] = 'cuda:0'
        device_map['lm_head'] = 'cuda:0'
    else:
        device_map[embedding_key] = 'cuda:0'
        device_map['lm_head'] = f"cuda:{n_gpus - 1}"

    # Assign final normalization to last GPU
    device_map[norm_key] = f"cuda:{n_gpus - 1}"

    return device_map

def test_answer(pred_str, ans_str):
    """
    Extract numbers and compare.
    """
    pattern = '\d*\.?\d+'
    pred = re.findall(pattern, pred_str)
    if (len(pred) >= 1):
        pred = pred[-1]
        gold = re.findall(pattern, ans_str)
        gold = gold[-1]
        return pred == gold
    else:
        return False

def parse_pred_ans(filename):
    with open(filename) as fd:
        lines = fd.readlines()
    am, a = None, None
    num_q, acc = 0, 0
    current_mode = 'none'
    questions = []
    ans_pred = []
    ans_gold = []
    debug = False
    for l in lines:
        if (l.startswith('Q: ')):
            if (am is not None and a is not None):
                questions.append(q)
                ans_pred.append(am)
                ans_gold.append(a)
                if (test_answer(am, a)):
                    acc += 1
            current_mode = 'q'
            q = l
            num_q += 1
        elif (l.startswith('A_model:')):
            current_mode = 'am'
            am = l
        elif (l.startswith('A:')):
            current_mode = 'a'
            a = l
        else:
            if (current_mode == 'q'):
                q += l
            elif (current_mode == 'am'):
                am += l
            elif (current_mode == 'a'):
                a += l
            else:
                raise ValueError(current_mode)

    questions.append(q)
    ans_pred.append(am)
    ans_gold.append(a)
    if (test_answer(am, a)):
        acc += 1
    print('num_q %d correct %d ratio %.4f' % (num_q, acc, float(acc / num_q)))
    return questions, ans_pred, ans_gold


class HSD():
    def __init__(self):
        self.args = argparse_setup()
        self.target_model_name = self.args.target_model
        self.draft_model_name = "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8"
        self.model_size = self.target_model_name.split("/")[1].split("-")[1]
        print(f'model size: {self.model_size}')
        if float(self.model_size[:-1])>3:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            dev = torch.device("cuda" if torch.cuda.is_available() else "mps")
        print(f'using device: {dev}')

        # =======================Loading gsm8k dataset=======================
        gsm8k = load_dataset('gsm8k', 'main')
        self.gsm8k_test = gsm8k['test']
        self.num_samples = len(self.gsm8k_test['question'])//20 if int(self.model_size[:-1])>16 else len(self.gsm8k_test['question'])
        print(f"num_samples:{self.num_samples}")

        # =======================Model setup=================================
        self.model_setup()
        self.sd = self.test_setup()

        if self.args.prompt == "original":
            self.prompt = open('lib_prompt/prompt_original.txt').read()
        else:
            self.prompt = open('lib_prompt/prompt_hardest.txt').read()

    def speculative_decoding(self, input_ids):
        """
        Support two modes under speculative:
        - HSD: internal enable of the HSD path (backward+clever), not exposed to CLI.
        - naive: vanilla speculative.
        """
        if self.args.HSD and self.args.naive:
            raise ValueError("Choose only one: --HSD or --naive")

        # shared kwargs
        gen_kwargs = dict(
            max_new_tokens=512,
            do_sample=True,
            assistant_model=self.draft_model,
            assistant_confidence_threshold=0,
            assistant_tokenizer=self.tokenizer1 if not self.same_tokenizer else None,
            tokenizer=self.tokenizer1,
            recursive=self.args.recursive,
        )

        if self.args.HSD:
            # Internally select HSD path; DO NOT expose backward/clever to CLI
            outputs, counts, return_runtime = self.target_model.generate(
                input_ids,
                **gen_kwargs,
                backward=True,
                clever=True,
                return_probs=True,
                approxi=self.args.approxi
            )
        elif self.args.naive:
            outputs, counts, return_runtime = self.target_model.generate(
                input_ids,
                **gen_kwargs,
                # naive: no backward/clever, no probs
                return_probs=False
            )
        else:
            raise ValueError("Please run with --speculative and either --HSD or --naive")

        # Always available
        self.total_counts["draft_eval"].append(counts.get("draft_eval", []))
        self.total_counts["sample_length"].append(counts.get("sample_length", []))
        self.total_counts["target_eval"].append(counts.get("target_eval", []))

        # HSD-specific (guarded)
        if self.args.HSD:
            self.total_counts["p_i"].append(counts.get("p_i", []))
            self.total_counts["q_i"].append(counts.get("q_i", []))
            self.total_counts["hist_lengths"].append(counts.get("hist_lengths", []))
            self.total_counts["step_back_probs"].append(counts.get("step_back_probs", []))
            self.total_counts["total_step"].append(counts.get("total_step", []))
            self.total_counts["ids"].append(counts.get("ids", []))

        # --- N and A ---
        N = sum(counts.get("sample_length", []))
        per_verify_lengths = flatten_hist_lengths(counts.get("hist_lengths", [])) if self.args.HSD else []
        A = (sum(per_verify_lengths) / len(per_verify_lengths)) if per_verify_lengths else 0.0

        # --- KV I/O estimate for TARGET verification steps ---
        prefix_len = int(input_ids.shape[1])  # prompt length
        batch_size = int(input_ids.shape[0])
        dtype_bytes = next(self.target_model.parameters()).element_size()   # 2 for fp16/bf16, 4 for fp32

        if np.isfinite(self.BW) and self.args.HSD and per_verify_lengths:
            kv_io_ms_target, f1_, f2_ = kv_time_ms_target(
                self.target_model.config, prefix_len, per_verify_lengths,
                batch=batch_size, dtype_bytes=dtype_bytes, bw_bytes_per_s=self.BW
            )
        else:
            kv_io_ms_target, f1_, f2_ = float("nan"), [], []

        # --- per-token costs & totals ---
        drafted  = sum(counts.get("draft_eval", []))
        accepted = N
        cA = counts.get("timing_assistant_ms", 0.0) / max(1, drafted)     # ms / drafted token
        cV = counts.get("timing_verify_ms", 0.0)    / max(1, accepted)     # ms / accepted token
        cH = counts.get("timing_hsd_cpu_ms", 0.0)   / max(1, accepted)

        # Target compute only (empirical residual):
        target_compute_ms = counts.get("timing_verify_ms", 0.0) - (kv_io_ms_target if np.isfinite(kv_io_ms_target) else 0.0)

        total_ms = (counts.get("timing_prefill_ms", 0.0) +
                    counts.get("timing_assistant_ms", 0.0) +
                    counts.get("timing_verify_ms", 0.0) +
                    counts.get("timing_hsd_cpu_ms", 0.0))

        # --- print a compact row ---
        print(
            f"[sample {self.progress}] total={total_ms/1000:.2f}s | "
            f"prefill={counts.get('timing_prefill_ms', 0.0):.0f}ms | "
            f"assist={counts.get('timing_assistant_ms', 0.0):.0f}ms | "
            f"verify={counts.get('timing_verify_ms', 0.0):.0f}ms | "
            f"KV(target)≈{(0 if not np.isfinite(kv_io_ms_target) else kv_io_ms_target):.0f}ms | "
            f"target_compute≈{target_compute_ms:.0f}ms | "
            f"HSD‑CPU={counts.get('timing_hsd_cpu_ms', 0.0):.0f}ms | "
            f"N={accepted} | A={A:.2f} | cA={cA:.2f} | cV={cV:.2f} | cH={cH:.2f}"
        )
        if self.args.HSD:
            print(f"f1: {f1_} || f2: {f2_}")

        # --- store for aggregates ---
        for k, v in {
            "prefill_ms": counts.get("timing_prefill_ms", 0.0),
            "assistant_ms": counts.get("timing_assistant_ms", 0.0),
            "verify_ms": counts.get("timing_verify_ms", 0.0),
            "kv_io_ms_target": kv_io_ms_target,
            "target_compute_ms": target_compute_ms,
            "hsd_cpu_ms": counts.get("timing_hsd_cpu_ms", 0.0),
            "N": accepted, "A": A, "cA": cA, "cV": cV, "cH": cH
        }.items():
            self.total_counts.setdefault(k, []).append(float(v))

        return outputs

    def __call__(self):
        if self.args.debug:
            self.debug()
        else:
            self.total_counts = {
                "draft_eval":[], "target_eval":[], "total_step":[], "sample_length":[],
                "step_back_probs":[], "p_i":[], "q_i":[], "hist_lengths": [], "time":[], "ids":[]
            }

            print("start training")
            self.BW = effective_bandwidth_Bps()
            final_txt_file = f'AAA_final_t_6/test_Qwen2.5-{self.model_size}_{self.args.prompt}_{self.sd}.txt'

            self.progress = 0
            with open(final_txt_file, 'w') as fd:
                for q, a in tqdm(zip(self.gsm8k_test['question'][:self.num_samples], self.gsm8k_test['answer'][:self.num_samples]),
                                total=self.num_samples):
                    print(f"progress: {self.progress}/{self.num_samples}")
                    self.progress += 1

                    prompt_q = self.prompt + '\nQuestion: ' + q + '\n'

                    # use chat template to avoid generating strange strings with repetition penalty
                    messages = [
                        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                        {"role": "user", "content": prompt_q}
                    ]
                    input_text = self.tokenizer2.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )

                    input_ids = self.tokenizer1(input_text, return_tensors="pt").input_ids
                    embedding_device = self.draft_model.model.embed_tokens.weight.device
                    input_ids = input_ids.to(embedding_device)

                    start = time.time()

                    if self.args.speculative:
                        outputs = self.speculative_decoding(input_ids)
                    else:
                        if self.args.model == "target":
                            outputs = self.target_model.generate(
                                input_ids, max_new_tokens=512, do_sample=True, tokenizer=self.tokenizer2
                            )
                        else:
                            outputs = self.draft_model.generate(
                                input_ids, max_new_tokens=512, do_sample=True, tokenizer=self.tokenizer1
                            )

                    end = time.time()
                    self.total_counts["time"].append(end - start)

                    ans_ = self.tokenizer1.decode(outputs[0], skip_special_tokens=True)
                    fd.write('Q: %s\nA_model:\n%s\nA:\n%s\n\n' % (q, ans_, a))

                save_path = f"AAA_final_t_6/{self.sd}_total_counts.json"
                print(f'saving to {save_path}')
                with open(save_path, "w") as f:
                    json.dump(self.total_counts, f)

            _, _, _ = parse_pred_ans(final_txt_file)

    def model_setup(self):
        print(f"load draft model: {self.draft_model_name}")
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            self.draft_model_name,
            device_map={"": device} if int(self.model_size[:-1])<32 else None
        )

        print(f"load target model: {self.target_model_name}")
        self.target_model = AutoModelForCausalLM.from_pretrained(
            self.target_model_name,
            device_map={"": device} if int(self.model_size[:-1])<32 else None
        )

        self.draft_model.generation_config.num_assistant_tokens = self.args.gamma
        self.draft_model.generation_config.assistant_confidence_threshold = 0
        self.draft_model.generation_config.temperature = self.args.temperature
        self.draft_model.generation_config.top_k = 0
        self.draft_model.generation_config.top_p = self.args.top_p

        self.target_model.generation_config.num_assistant_tokens = self.args.gamma
        self.target_model.generation_config.assistant_confidence_threshold = 0
        self.target_model.generation_config.temperature = self.args.temperature
        self.target_model.generation_config.top_k = 0
        self.target_model.generation_config.top_p = self.args.top_p

        vocab_size = min(self.draft_model.config.vocab_size, self.target_model.config.vocab_size)
        self.draft_model.config.vocab_size = vocab_size
        self.target_model.config.vocab_size = vocab_size
        self.same_tokenizer =  self.target_model.config.get_text_config().vocab_size == self.draft_model.config.get_text_config().vocab_size

        # Resize lm_head if needed (to align vocab size)
        if hasattr(self.draft_model, "lm_head"):
            old_lm_head = self.draft_model.lm_head
            dtype = old_lm_head.weight.dtype
            self.draft_model.lm_head = nn.Linear(old_lm_head.in_features, vocab_size, bias=False).to(old_lm_head.weight.device, dtype=dtype)
            self.draft_model.lm_head.weight.data[:old_lm_head.out_features] = old_lm_head.weight.data[:vocab_size]

        if hasattr(self.target_model, "lm_head"):
            old_lm_head = self.target_model.lm_head
            dtype = old_lm_head.weight.dtype
            new_lm_head = nn.Linear(old_lm_head.in_features, vocab_size, bias=False).to(old_lm_head.weight.device, dtype=dtype)
            with torch.no_grad():
                new_lm_head.weight[:min(old_lm_head.out_features, vocab_size)] = \
                    old_lm_head.weight[:min(old_lm_head.out_features, vocab_size)]
            self.target_model.lm_head = new_lm_head

        if torch.cuda.is_available() and int(self.model_size[:-1])>14:
            device_map1 = manual_device_map(self.draft_model)
            device_map2 = manual_device_map(self.target_model)
            self.draft_model = dispatch_model(self.draft_model, device_map=device_map1, offload_dir=None)
            self.target_model = dispatch_model(self.target_model, device_map=device_map2, offload_dir=None)

        print("dispatch model finished")

        self.draft_model.eval()
        self.target_model.eval()

        # Fix rotary embedding buffers that may still be on CPU
        move_rotary_emb_to_device(self.draft_model)
        move_rotary_emb_to_device(self.target_model)

        # load tokenizers
        self.tokenizer1 = AutoTokenizer.from_pretrained(self.draft_model_name)
        self.tokenizer2 = AutoTokenizer.from_pretrained(self.target_model_name)

    def test_setup(self):
        sd = f"Qwen_{self.model_size}_0.5B_"
        if self.args.speculative:
            if self.args.HSD:
                sd += "HSD"
            elif self.args.naive:
                sd += "naive"
            else:
                sd += "spec"
        else:
            sd += self.args.model
        sd += f"_gamma_{self.args.gamma}"

        if self.args.multidraft > 1:
            sd += f"_multidraft_{self.args.multidraft}"
        if self.args.parallel:
            sd += "_parallel"

        if self.args.temperature < 1:
            sd += f"_t{self.args.temperature}"
        if self.args.top_p < 10:
            sd += f"_topp_{self.args.top_p}"

        sd += f'{self.args.name}'
        return sd

    def debug(self):
        # Example single-item debug path
        input_text = self.gsm8k_test[0]['question']
        prompt_q = self.prompt + '\nQuestion: ' + input_text + '\n'

        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt_q}
        ]
        input_text = self.tokenizer2.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        input_ids = self.tokenizer1(input_text, return_tensors="pt").input_ids
        embedding_device = self.draft_model.model.embed_tokens.weight.device
        input_ids = input_ids.to(embedding_device)

        if self.args.speculative:
            if not (self.args.HSD or self.args.naive):
                raise ValueError("In debug with --speculative, choose --HSD or --naive")
            _ = self.speculative_decoding(input_ids)
        else:
            _ = self.target_model.generate(input_ids, max_new_tokens=50, do_sample=True, tokenizer=self.tokenizer2)


def main():
    hsd = HSD()
    hsd()

if __name__ == "__main__":
    main()
