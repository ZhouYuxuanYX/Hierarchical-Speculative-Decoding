import re
import os
# !!!!!!!!!!!!must set the environment variable before importing transformers, otherwise it won't work!!!!!!!!
######### use the local cache on haicore
os.environ['HF_HOME'] = '/home/c02yuzh/CISPA-projects/rectified_softmax_ml-2023/hf_home'
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
import time
import random
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser(prog='myprogram')
parser.add_argument('--backward', action='store_true', default=False)
parser.add_argument('--clever', action='store_true', default=False)
parser.add_argument('--multidraft', type=int, default=1)
parser.add_argument('--temperature', type=float, default=1)
# parser.add_argument('--top-p', type=float, default=1)
parser.add_argument('--blockwise', action='store_true', default=False)
parser.add_argument('--recursive', action='store_true', default=False)
parser.add_argument('--speculative', action='store_true', default=False)
parser.add_argument('--parallel', action='store_true', default=False)
parser.add_argument('--gamma',  default=10, type=int, help='number of assited tokens')
parser.add_argument("--approxi", action='store_true', default=False)
parser.add_argument('--model', help='must be target or draft', default="target")
parser.add_argument('--prompt',  default='original', help='must be complex or original')
parser.add_argument('--target-model',  default='Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8', help='must be complex or original')
parser.add_argument('--debug', action='store_true', default=False)
args = parser.parse_args()
########### it is super slow to use cpu ##########
########## takes 4.5 h to use Llama-3-7B on the full test set of 1319 test questions#########

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


######### use only a fraction to speed up the development process #########

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# for mac
### for 7B which is too large for mac mps cache, no solution works, so just use cpu
# model2_name = "Qwen/Qwen2.5-7B-Instruct"
# model2_name = "Qwen/Qwen2.5-32B-Instruct"
# model2_name = "Qwen/Qwen2.5-72B-Instruct"
model1_name = "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8"
model2_name = args.target_model

model_size = model2_name.split("/")[1].split("-")[1]
print(model_size)
# exit()

if float(model_size[:-1])>3:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

print(device)

### for 7B which is too large for mac mps cache, no solution works, so just use cpu
# local_path = "./qwen-7b-offload"

# if not os.path.exists(local_path):
#     snapshot_download(model_2_name, local_dir=local_path, local_dir_use_symlinks=False)
# # Load model with disk offload
# with init_empty_weights():
#     model2 = AutoModelForCausalLM.from_pretrained(local_path)
#
# tokenizer2 = AutoTokenizer.from_pretrained(local_path)
# model2 = load_checkpoint_and_dispatch(
#     model2,
#     local_path,
#     device_map={"": "mps"},  # or "cpu" if MPS isn't stable
#     offload_folder="./offload",  # path to store offloaded weights
#     offload_state_dict=True,
# )


gsm8k = load_dataset('gsm8k', 'main')

gsm8k_test = gsm8k['test']

num_samples = len(gsm8k_test['question'])//5
# num_samples = 10

print(f"num_samples:{num_samples}")


validation_index = np.load('lib_prompt/validation_index.npy')
validation_data = gsm8k['train'].select(validation_index)

tokenizer1 = AutoTokenizer.from_pretrained(model1_name)
tokenizer2 = AutoTokenizer.from_pretrained(model2_name)


print("load draft model")
model1 = AutoModelForCausalLM.from_pretrained(model1_name,
    device_map={"": device} if int(model_size[:-1])<32 else None)
# tokenizer2 = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
# model2 = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", device_map='auto')

print("load target model")
model2 = AutoModelForCausalLM.from_pretrained(model2_name,
    device_map={"": device} if int(model_size[:-1])<32 else None)


print("load model finished")

model1.generation_config.num_assistant_tokens = args.gamma
# otherwise the draft length will change dynamically
model1.generation_config.assistant_confidence_threshold = 0
model1.generation_config.temperature = args.temperature
model1.generation_config.top_k = 0
# model1.generation_config.top_p = args.top-p

model2.generation_config.num_assistant_tokens = args.gamma
# otherwise the draft length will change dynamically
model2.generation_config.assistant_confidence_threshold = 0
model2.generation_config.temperature = args.temperature
model2.generation_config.top_k = 0
# model2.generation_config.top_p = args.top-p

vocab_size = min(model1.config.vocab_size, model2.config.vocab_size)
model1.config.vocab_size = vocab_size
model2.config.vocab_size = vocab_size
same_tokenizer =  model2.config.get_text_config().vocab_size == model1.config.get_text_config().vocab_size


# # Check the tokenizer's vocabulary size
# # 151665
# tokenizer_vocab_size = len(tokenizer1)
# print(f"Tokenizer Vocabulary Size: {tokenizer_vocab_size}")
#
# # Check the model's embedding layer size (vocab size in config)
# embedding_vocab_size = vocab_size
# # 151936
# print(f"Model Embedding Vocab Size: {embedding_vocab_size}")
#
# # Ensure that the tokenizer's vocabulary size matches the model's embedding size
# if tokenizer_vocab_size != embedding_vocab_size:
#     print("Warning: Tokenizer vocab size and model vocab size do not match.")
#     # Handle the mismatch, e.g., by trimming the tokenizer's vocab to match the model's embedding size
#     tokenizer_vocab_size = min(tokenizer_vocab_size, embedding_vocab_size)
#
# # Select the pad_token_id
# # Choose a pad_token_id that doesn't conflict with actual tokens (must be beyond tokenizer vocab size)
# if tokenizer_vocab_size < embedding_vocab_size:
#     pad_token_id = tokenizer_vocab_size  # Typically the next unused ID beyond tokenizer's vocab size
# else:
#     pad_token_id = 0  # Safe default if vocab size and tokenizer match
#
# # Set the pad_token_id in model config
# model1.config.pad_token_id = pad_token_id
# model2.config.pad_token_id = pad_token_id
#
# # Set it in the generation configuration as well (if needed)
# model1.generation_config.pad_token_id = pad_token_id
# model2.generation_config.pad_token_id = pad_token_id

# just changing the config.vocab_size is not enough, RuntimeError: The size of tensor a (152064) must match the size of tensor b (151936) at non-singleton dimension 2
# change output size too
# Manually resize lm_head if needed
if hasattr(model1, "lm_head"):
    old_lm_head = model1.lm_head
    dtype = old_lm_head.weight.dtype  # preserve dtype, likely torch.float16 or torch.int8 (for GPTQ)
    model1.lm_head = nn.Linear(old_lm_head.in_features, vocab_size, bias=False).to(old_lm_head.weight.device, dtype=dtype)
    model1.lm_head.weight.data[:old_lm_head.out_features] = old_lm_head.weight.data[:vocab_size]

if hasattr(model2, "lm_head"):
    old_lm_head = model2.lm_head
    dtype = old_lm_head.weight.dtype  # preserve dtype, likely torch.float16 or torch.int8 (for GPTQ)

    # Create new lm_head with correct dtype and device
    new_lm_head = nn.Linear(old_lm_head.in_features, vocab_size, bias=False).to(old_lm_head.weight.device, dtype=dtype)

    # Copy existing weights if within bounds
    with torch.no_grad():
        new_lm_head.weight[:min(old_lm_head.out_features, vocab_size)] = \
            old_lm_head.weight[:min(old_lm_head.out_features, vocab_size)]

    model2.lm_head = new_lm_head
# redistribute after changing the layer, otherwise it won't work using "balanced" device map for multi-gpu context
# Get a recommended device map first
# device_map = infer_auto_device_map(model1, max_memory={i: "40GiB" for i in range(torch.cuda.device_count())})


def manual_device_map(model, same_device_for_input_output=True):
    """
    Manually create a balanced device map for a Hugging Face transformer model.

    Args:
        model: The loaded model (e.g. AutoModelForCausalLM).
        same_device_for_input_output: If True, places embedding and output (lm_head) on same device (cuda:0).

    Returns:
        device_map dict to use with `dispatch_model`.
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

if torch.cuda.is_available() and int(model_size[:-1])>14:
    device_map1 = manual_device_map(model1)
    device_map2 = manual_device_map(model2)
    model1 = dispatch_model(model1, device_map=device_map1, offload_dir=None)
    model2 = dispatch_model(model2, device_map=device_map2, offload_dir=None)

print("dispatch model finished")

model1.eval()
model2.eval()
# Fix rotary embedding buffers that may still be on CPU
def move_rotary_emb_to_device(model):
    # Get the device of the layer where rotary embedding is applied
    try:
        device = model.model.embed_tokens.weight.device
        if hasattr(model.model, "rotary_emb") and hasattr(model.model.rotary_emb, "inv_freq"):
            model.model.rotary_emb.inv_freq = model.model.rotary_emb.inv_freq.to(device)
    except AttributeError:
        print("Could not move rotary_emb.inv_freq — structure may be different")

move_rotary_emb_to_device(model1)
move_rotary_emb_to_device(model2)

# print(same_tokenizer)
# exit()
# huggingface code:   if self.config.get_text_config().vocab_size == assistant_model.config.get_text_config().vocab_size
# just set model.config.vocab_size to the same
# print(model1.config.vocab_size)
# print(model2.config.vocab_size)
# # exit()
# # after checking the actual embedding size and the special tokens, we know that the embedding size is larger than the actual vocabulary, even including the special tokens
# print("Model 1 embedding size:    ", model1.get_input_embeddings().weight.shape[0])
# print("Model 2 embedding size:    ", model2.get_input_embeddings().weight.shape[0])
#
#
# print("Tokenizer 1 added tokens:", tokenizer1.added_tokens_encoder)
# print("All special tokens:", tokenizer1.all_special_tokens)
#
# print("Tokenizer 2 added tokens:", tokenizer2.added_tokens_encoder)
# print("All special tokens:", tokenizer2.all_special_tokens)
#
#
# # check tokenizer differences，they are exactly the same, just merge_file and vocab_file path is different
# # Get their vocabularies
# vocab1 = tokenizer1.get_vocab()
# vocab2 = tokenizer2.get_vocab()
# #
# print(len(tokenizer1))
# print(len(tokenizer2))
# print(len(vocab1))
# print(len(vocab2))

# # Convert keys (tokens) to sets for easy comparison
# tokens1 = set(vocab1.keys())
# tokens2 = set(vocab2.keys())
#
# # Tokens only in tokenizer1
# only_in_1 = tokens1 - tokens2
#
# # Tokens only in tokenizer2
# only_in_2 = tokens2 - tokens1
#
# # Common tokens
# common_tokens = tokens1 & tokens2
# #
# print(f"Tokens only in tokenizer1: {len(only_in_1)}")
# print(f"Tokens only in tokenizer2: {len(only_in_2)}")
# print(f"Common tokens: {len(common_tokens)}")
# changed_indices = {token: (vocab1[token], vocab2[token])
#                    for token in common_tokens if vocab1[token] != vocab2[token]}
# print(changed_indices)
#
# # Check padding token
# print("Tokenizer 1 padding token:", tokenizer1.pad_token)
# print("Tokenizer 2 padding token:", tokenizer2.pad_token)
#
# # Check padding token ID
# print("Tokenizer 1 pad token ID:", tokenizer1.pad_token_id)
# print("Tokenizer 2 pad token ID:", tokenizer2.pad_token_id)
#
# # Check other config differences
# diffs = {k: (tokenizer1.init_kwargs.get(k), tokenizer2.init_kwargs.get(k))
#          for k in set(tokenizer1.init_kwargs) | set(tokenizer2.init_kwargs)
#          if tokenizer1.init_kwargs.get(k) != tokenizer2.init_kwargs.get(k)}
#
# print("\nDifferences in tokenizer config:")
# for key, value in diffs.items():
#     print(f"{key}: {value}")
# exit()
# set repetition penalty to 1 for verifying whether it is the root of the instruct model problem
# model1.generation_config.repetition_penalty =1.0
# model2.generation_config.repetition_penalty =1.0

# check if the original speculative decoding will have this problem
# yes! it also has this problem
# and it is the problem, no problem when setting the large model (model2) with repeptition_penalty=1 (no penalty)

# check the default configuration of the pretained model
# it will not be shown if not print explicitly
# print(model1.generation_config)
# print(model2.generation_config)
# print(model1.generation_config.assistant_confidence_threshold)
# print(model1.generation_config.repetition_penalty)
# print(model2.generation_config.repetition_penalty)
# print(model1.generation_config.pad_token_id)
# exit()
"""
GenerationConfig {
  "bos_token_id": 128000,
  "do_sample": true,
  "eos_token_id": 128001,
  "max_length": 4096,
  "temperature": 0.6,
  "top_p": 0.9
}
"""

# verified this by checking gpt3.5turbo evaluation script, prompt_complex is named as prompt_hardest.txt

if args.prompt == "original":
    prompt = open('lib_prompt/prompt_original.txt').read()
else:
    prompt = open('lib_prompt/prompt_hardest.txt').read()


# exit()
if args.debug:
    q_lens = [len(d['question']) for d in gsm8k['train']]
    print(np.percentile(q_lens, [50, 80, 90, 95, 99, 100]))

    print(np.argmax(q_lens))

    input_text = gsm8k['train'][3331]['question']

    print(gsm8k['train'][3331]['answer'])

    prompt_q = prompt + '\nQuestion: ' + input_text + '\n'
    # print(prompt_q)

    # use chat template to avoid generating strange strings with repetition penalty
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt_q}
    ]
    input_text = tokenizer2.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )



    input_ids = tokenizer1(input_text, return_tensors="pt").input_ids.to(device)
    print(input_ids.size())
    # exit()

    # stop_list = [" \n\nQuestion:", " \n\n", "\n\n", "\n\n ",
    #              " \n\n ", "\n\n\n", "\n\nQuestion",
    #              ".\n\n",
    #              ". \n\n",
    #              "\n \n", " \n \n "
    #              ]
    # stop_token_ids = [tokenizer1(x, return_tensors='pt', add_special_tokens=False)['input_ids'] for x in stop_list]
    # stop_token_ids = [LongTensor(x).to(device) for x in stop_token_ids]

    print(input_ids.device)
    print(model1.device)
    # exit()
    # stop = StopOnTokens(stop_token_ids)
    # Llama3 has a context window of 8k tokens
    # stopping_criteria = StoppingCriteriaList([stop])

    # print("eos_token_id")
    # print(tokenizer1.eos_token_id)
    # print(tokenizer2.eos_token_id)

    # i checked that the model during speculative decoding stops at different token ids each time, and not the eos token id 151643

    # checked that model1 can correclty yield max_new_tokens correctly
    # for i in range(5):
    #     outputs = model1.generate(input_ids, max_new_tokens=5, do_sample=True, backward=True)
    #     print(outputs)
    #     print(outputs.shape)
    #     input_ids = outputs
    # exit()

    # i also confirmed that even for the official assisted decoding code, the candidate generator will often stop early without encountering eos token
    # found a transformers.generation.stopping_criteria.ConfidenceCriteria, might be the reason:
    # This class can be used to stop generation whenever assistant model's confidence in its prediction for the current token is lower than the threshold

    # print(same_tokenizer)

    # if not setting do_sample=True, the program will skip speculative decoding and causes error
    # I have checked that model1 and model2 have the same tokenizer by providing assitant_tokenizer to generate, it will raises error and explains it is redundant
    if args.speculative:
        # the huggingface implementation decide whether they are different tokenizers, based on assistant model keys
        # different_tokenizers = all(v is not None for v in (assistant_model, target_tokenizer, assistant_tokenizer))
        # so i just need to assign assistant model's tokenizer to both to avoid hf thinking they are different
        outputs, counts = model2.generate(input_ids, max_new_tokens=100, do_sample=True,
                              assistant_model=model1,
                              # stopping_criteria=stopping_criteria,
                              assistant_confidence_threshold=0,
                              backward=args.backward,
                              recursive=args.recursive,
                              assistant_tokenizer=tokenizer1 if not same_tokenizer else None,
                              tokenizer=tokenizer1
                              )


    else:
        outputs = model2.generate(input_ids, max_new_tokens=100, do_sample=True,
                                          # assistant_model=model1,
                                          # stopping_criteria=stopping_criteria,
                                          # assistant_confidence_threshold=0,
                                          # backward=True,
                                          # recursive=True,
                                          # assistant_tokenizer=tokenizer1,
                                          tokenizer=tokenizer2
                                          )

# while len(bsd.total_gen)

# print("##############check#################")
# # use skip_special_tokens=True to skip <|im_end|> tokens
# print(tokenizer2.decode(input_ids[0], skip_special_tokens=True))
# print("###################")
# print(tokenizer2.decode(outputs[0], skip_special_tokens=True))
# print(outputs[0].size())


# print(np.array(counts["draft_eval"]).sum())
# print(counts["draft_eval"])
#
# print(np.array(counts["target_eval"]).sum())
# print(counts["target_eval"])
#
# print(np.array(counts["draft_length"]).sum())
# print(counts["draft_length"])
#
# print(np.array(counts["sample_length"]).sum())
# print(counts["sample_length"])



# exit()

else:

    def test_answer(pred_str, ans_str):
        """
        Regular Expression Pattern (pattern = '\d*\.?\d+')
            \d* → Matches zero or more digits before a decimal point.
            \.? → Matches an optional decimal point.
            \d+ → Matches one or more digits after the decimal.
            This pattern extracts numbers, including decimals, from a given string.
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
            # print("check l")
            # print(l)
            # print("#######################")

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

    i = 0
    sd = f"Qwen_{model_size}_0.5B_"
    if args.speculative:
        if args.blockwise:
            sd += "blockwise"
        else:
            sd += "backward" if args.backward else "naive"
            if args.recursive:
                sd += "_recursive"
            elif args.clever:
                sd += "_clever"
                if args.approxi:
                    sd+="_approxi"

    else:
        sd += args.model
    sd += f"_gamma_{args.gamma}"

    if args.multidraft > 1:
        sd += f"_multidraft_{args.multidraft}"

    if args.parallel:
        sd += "_parallel"
    
    if args.temperature<1:
        sd += f"_t{args.temperature}"

    # if args.top-p <1:
    #     sd += f"_topp_{args.top-p}"

    total_counts = {"draft_eval":[], "target_eval":[], "total_step":[], "sample_length":[],
                  "step_back_probs":[], "p_i":[], "q_i":[], "hist_lengths": [], "time":[], "ids":[]}

    progress=0

    print("start training")

    with open(f'outputs/test_Qwen2.5-{model_size}_{args.prompt}_{sd}.txt', 'w') as fd:
        for q, a in tqdm(zip(gsm8k_test['question'][:num_samples], gsm8k_test['answer'][:num_samples]),
                         total=num_samples):
            print(f"progress: {progress}/{num_samples}")
            progress+=1

            prompt_q = prompt + '\nQuestion: ' + q + '\n'

            # use chat template to avoid generating strange strings with repetition penalty
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": prompt_q}
            ]
            input_text = tokenizer2.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            input_ids = tokenizer1(input_text, return_tensors="pt").input_ids

            # Safer: Use the input embedding layer's device
            embedding_device = model1.model.embed_tokens.weight.device
            input_ids = input_ids.to(embedding_device)

            # the model will continue creating new question and answer patterns after answering the real question
            # add "\n\n" as stopping criterion to avoid such a problem

            # stopping_criteria = StoppingCriteriaList([stop])
            start = time.time()

            if args.speculative:
                # the huggingface implementation decide whether they are different tokenizers, based on assistant model keys
                # different_tokenizers = all(v is not None for v in (assistant_model, target_tokenizer, assistant_tokenizer))
                # so i just need to assign assistant model's tokenizer to both to avoid hf thinking they are different
                outputs, counts = model2.generate(input_ids, max_new_tokens=512, do_sample=True,
                                                  assistant_model=model1,
                                                  # stopping_criteria=stopping_criteria,
                                                  assistant_confidence_threshold=0,
                                                  backward=args.backward,
                                                  assistant_tokenizer=tokenizer1 if not same_tokenizer else None,
                                                  tokenizer=tokenizer1,
                                                  recursive=args.recursive,
                                                  return_probs=args.backward or args.blockwise,
                                                  blockwise=args.blockwise,
                                                  clever=args.clever,
                                                  approxi=args.approxi,
                                                  multidraft=args.multidraft,
                                                  parallel= args.parallel
                                                  )

                total_counts["draft_eval"].append(counts["draft_eval"])
                total_counts["sample_length"].append(counts["sample_length"])
                total_counts["target_eval"].append(counts["target_eval"])
                total_counts["p_i"].append(counts["p_i"])
                total_counts["q_i"].append(counts["q_i"])
                total_counts["hist_lengths"].append(counts["hist_lengths"])
                total_counts["step_back_probs"].append(counts["step_back_probs"])
                total_counts["total_step"].append(counts["total_step"])
                total_counts["ids"].append(counts["ids"])

            else:
                if args.model == "target":
                    outputs = model2.generate(input_ids, max_new_tokens=512, do_sample=True,
                                                      # assistant_model=model1 if args.speculative else None,
                                                      # stopping_criteria=stopping_criteria,
                                                      # assistant_confidence_threshold=0,
                                                      # backward=args.backward,
                                                      # assistant_tokenizer=tokenizer1 if args.speculative else None,
                                                      tokenizer=tokenizer2
                                                      )
                else:
                    outputs = model1.generate(input_ids, max_new_tokens=512, do_sample=True,
                                                      # assistant_model=model1 if args.speculative else None,
                                                      # stopping_criteria=stopping_criteria,
                                                      # assistant_confidence_threshold=0,
                                                      # backward=args.backward,
                                                      # assistant_tokenizer=tokenizer1 if args.speculative else None,
                                                      tokenizer=tokenizer1
                                                      )
                total_counts["sample_length"].append(len(outputs[0] - len(input_ids[0])))

            end = time.time()
            total_counts["time"].append(start-end)

            ans_ = tokenizer1.decode(outputs[0], skip_special_tokens=True)
            # print("check q, ans_ and a")
            # print(q)
            # print("###########################")
            # print(ans_)
            # # the model will continue create new question and answer patterns after answering the real question
            # print("###########################")
            # print(a)
            # exit()

            fd.write('Q: %s\nA_model:\n%s\nA:\n%s\n\n' % (q, ans_, a))

        with open(f"{sd}_total_counts.json", "w") as f:
            json.dump(total_counts, f)

    _, _, _ = parse_pred_ans(f'outputs/test_Qwen2.5-{model_size}_{args.prompt}_{sd}.txt')

    # prompt_original = open('lib_prompt/prompt_original.txt').read()

    # i = 0
    # with open('outputs/test_Llama3-8B_original.txt', 'w') as fd:
    #     for q, a in tqdm(zip(gsm8k_test['question'], gsm8k_test['answer']),
    #                      total=len(gsm8k_test['question'])):
    #         prompt_q = prompt_original + '\nQuestion: ' + q + '\n'
    #         input_ids = tokenizer(prompt_q, return_tensors="pt").input_ids.to("cuda:0")
    #         outputs = model.generate(input_ids, max_new_tokens=256)
    #         ans_ = tokenizer.decode(outputs[0])
    #         fd.write('Q: %s\nA_model:\n%s\nA:\n%s\n\n' % (q, ans_, a))
    # _, _, _ = parse_pred_ans('outputs/test_Llama3-8B_original.txt')
    #[ prompt_simple = open('lib_prompt/prompt_simple.txt').read()
