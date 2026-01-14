import copy
import random

# typing 
from typing import List, Tuple
import time
import torch

# TODO
# from transformers import LlamaTokenizer
# tokenizer=LlamaTokenizer.from_pretrained("/home/lyh/weights/hf/vicuna_v13/7B/")

TOPK = 10  # topk for sparse tree

from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


class Timer:
    def __init__(self,name):
        self.name = name
    def __enter__(self):
        torch.cuda.synchronize()
        self.start = time.perf_counter()


    def __exit__(self, exc_type, exc_value, traceback):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start
        print(f'{self.name} took {elapsed} seconds')


def prepare_logits_processor(
        temperature: float = 1.0,
        repetition_penalty: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 10
) -> LogitsProcessorList:
    # print(f'top_p: {top_p} || top_k: {top_k} || temperature: {temperature} || repetition_penalty: {repetition_penalty}')
    processor_list = LogitsProcessorList()
    if temperature > 1e-5:
        if temperature >= 1e-5 and temperature != 1.0:
            processor_list.append(TemperatureLogitsWarper(temperature))
        if repetition_penalty > 1.0:
            processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        if 1e-8 <= top_p < 1.0:
            processor_list.append(TopPLogitsWarper(top_p))
        if top_k > 0:
            processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


# test_processor = prepare_logits_processor(
#         0.0, 0.0, -1, 1
#     )


def pad_path(path: List[int], length: int, pad_value: int = -2) -> List[int]:
    """
    Pad the given path list with a specific value up to a specified length.

    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.

    Returns:
    - list: A new list based on the original path but padded to the desired length.

    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]

    Note:
    If the given path is already longer than the specified length,
    then no padding occurs, and the original path is returned.
    """

    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))


def generate_tree_buffers(tree_choices, device="cuda"):
    def custom_sort(lst):
        # sort_keys=[len(list)]
        sort_keys = []
        for i in range(len(lst)):
            sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
        return sort_keys
    with Timer("sort"):

        sorted_tree_choices = sorted(tree_choices, key=lambda x: (len(x), x))
        tree_len = len(sorted_tree_choices) + 1

    # Initialize depth_counts to keep track of how many choices have a particular depth
        depth_counts = []
        prev_depth = 0
        for path in sorted_tree_choices:
            depth = len(path)
            if depth != prev_depth:
                depth_counts.append(0)
            depth_counts[depth - 1] += 1
            prev_depth = depth

        tree_attn_mask = torch.eye(tree_len, tree_len)
        tree_attn_mask[:, 0] = 1
        start = 0
        for i in range(len(depth_counts)):
            for j in range(depth_counts[i]):
                cur_tree_choice = sorted_tree_choices[start + j]
                # retrieve ancestor position
                if len(cur_tree_choice) == 1:
                    continue
                ancestor_idx = []
                for c in range(len(cur_tree_choice) - 1):
                    ancestor_idx.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]) + 1)
                tree_attn_mask[j + start + 1, ancestor_idx] = 1
            start += depth_counts[i]

        tree_indices = torch.zeros(tree_len, dtype=torch.long)
        p_indices = [0 for _ in range(tree_len - 1)]
        b_indices = [[] for _ in range(tree_len - 1)]
        tree_indices[0] = 0
        start = 0
        bias = 0
        for i in range(len(depth_counts)):
            inlayer_bias = 0
            b = []
            for j in range(depth_counts[i]):
                cur_tree_choice = sorted_tree_choices[start + j]
                cur_parent = cur_tree_choice[:-1]
                if j != 0:
                    if cur_parent != parent:
                        bias += 1
                        inlayer_bias += 1
                        parent = cur_parent
                        b = []
                else:
                    parent = cur_parent
                tree_indices[start + j + 1] = cur_tree_choice[-1] + TOPK * (i + bias) + 1
                p_indices[start + j] = inlayer_bias
                if len(b) > 0:
                    b_indices[start + j] = copy.deepcopy(b)
                else:
                    b_indices[start + j] = []
                b.append(cur_tree_choice[-1] + TOPK * (i + bias) + 1)
            start += depth_counts[i]

        p_indices = [-1] + p_indices
        tree_position_ids = torch.zeros(tree_len, dtype=torch.long)
        start = 0
        for i in range(len(depth_counts)):
            tree_position_ids[start + 1: start + depth_counts[i] + 1] = i + 1
            start += depth_counts[i]

        retrieve_indices_nest = []
        retrieve_paths = []
        for i in range(len(sorted_tree_choices)):
            cur_tree_choice = sorted_tree_choices[-i - 1]
            retrieve_indice = []
            if cur_tree_choice in retrieve_paths:
                continue
            else:
                for c in range(len(cur_tree_choice)):
                    retrieve_indice.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]))
                    retrieve_paths.append(cur_tree_choice[:c + 1])
            retrieve_indices_nest.append(retrieve_indice)
        max_length = max([len(x) for x in retrieve_indices_nest])
        retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        retrieve_indices = retrieve_indices + 1
        retrieve_indices = torch.cat([torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices],
                                     dim=1)

        maxitem = retrieve_indices.max().item() + 5



        retrieve_indices = retrieve_indices.tolist()
        retrieve_indices = sorted(retrieve_indices, key=custom_sort)
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)



    # Aggregate the generated buffers into a dictionary
    tree_buffers = {
        "tree_attn_mask": tree_attn_mask.unsqueeze(0).unsqueeze(0),
        "tree_indices": tree_indices,
        "tree_position_ids": tree_position_ids,
        "retrieve_indices": retrieve_indices,
    }

    # Move the tensors in the dictionary to the specified device
    tree_buffers = {
        k: v.clone().to(device)
        if isinstance(v, torch.Tensor)
        else torch.tensor(v, device=device)
        for k, v in tree_buffers.items()
    }

    return tree_buffers


def initialize_tree0(input_ids, model, past_key_values, logits_processor):
    draft_tokens, retrieve_indices,tree_mask,tree_position_ids, outputs, logits, hidden_state, sample_token = model(
        input_ids, past_key_values=past_key_values, output_orig=True, logits_processor=logits_processor
    )

    #     if logits_processor is not None:
    #         logits = orig[:, -1]
    #         logits = logits_processor(None, logits)
    #         probabilities = torch.nn.functional.softmax(logits, dim=1)
    #         token = torch.multinomial(probabilities, 1)
    #     else:
    #         token = torch.argmax(orig[:, -1])
    #         token = token[None, None]
    #     input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
    #     # Clone the output hidden states
    #
    #     draft_tokens, retrieve_indices,tree_mask,tree_position_ids = self.ea_layer.topK_genrate(hidden_states, input_ids, self.base_model.lm_head)
    #     if output_orig:
    #         return draft_tokens, retrieve_indices,tree_mask,tree_position_ids, outputs, orig, hidden_states, token
    #     return draft_tokens, retrieve_indices,tree_mask,tree_position_ids, hidden_states, token
    return draft_tokens, retrieve_indices,tree_mask,tree_position_ids, logits, hidden_state, sample_token

def initialize_tree(input_ids, model, past_key_values, logits_processor):
    outputs, orig, hidden_states = model(
        input_ids, past_key_values=past_key_values, output_orig=True
    )

    if logits_processor is not None:
        logits = orig[:, -1]
        logits = logits_processor(None, logits)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        token = torch.multinomial(probabilities, 1)
    else:
        token = torch.argmax(orig[:, -1])
        token = token[None, None]
    input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)

    # Clone the output hidden states
    if model.use_eagle3:
        ea_device = model.ea_layer.lm_head.weight.device
        if outputs["hidden_states"][0].device != ea_device:
            outputs["hidden_states"] = [x.to(ea_device) for x in outputs["hidden_states"]]
        hidden_states=torch.cat(outputs["hidden_states"],dim=-1)
    draft_tokens, retrieve_indices,tree_mask,tree_position_ids = model.ea_layer.topK_genrate(hidden_states, input_ids, model.base_model.lm_head,logits_processor)
    return draft_tokens, retrieve_indices,tree_mask,tree_position_ids, orig, hidden_states, token


def reset_tree_mode(
        model,
):
    model.base_model.model.tree_mask = None
    model.base_model.model.tree_mode = None


def reset_past_key_values(passed_key_values: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Resets the current lengths in the passed key-values to zero.

    This function is designed to be used during the evaluation of a baseline model.
    It iterates through each layer's key-values and sets their current lengths to zero,
    effectively resetting their state.

    Args:
    - passed_key_values (list of torch.Tensor): Contains past hidden states and past attention values for each layer.

    Returns:
    - passed_key_values (list of torch.Tensor): Updated past hidden states and past attention values with reset lengths.
    """
    for i in range(len(passed_key_values)):
        for j in range(2):
            passed_key_values[i][j].current_length.fill_(0)
    return passed_key_values


def generate_candidates(tree_logits, tree_indices, retrieve_indices, sample_token, logits_processor):
    sample_token = sample_token.to(tree_indices.device)

    candidates_logit = sample_token[0]

    candidates_tree_logits = tree_logits

    candidates = torch.cat([candidates_logit, candidates_tree_logits.view(-1)], dim=-1)

    tree_candidates = candidates[tree_indices]

    tree_candidates_ext = torch.cat(
        [tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device) - 1], dim=0)

    cart_candidates = tree_candidates_ext[retrieve_indices]


    # Unsqueeze the tree candidates for dimension consistency.
    tree_candidates = tree_candidates.unsqueeze(0)
    return cart_candidates,  tree_candidates


def tree_decoding(
        model,
        tree_candidates,
        past_key_values,
        tree_position_ids,
        input_ids,
        retrieve_indices,
):
    position_ids = tree_position_ids + input_ids.shape[1]
    if position_ids is not None and position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
    outputs, tree_logits, hidden_state = model(
        tree_candidates,
        output_orig=True,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    if model.use_eagle3:
        ea_device = model.ea_layer.lm_head.weight.device
        if outputs["hidden_states"][0].device != ea_device:
            outputs["hidden_states"] = [x.to(ea_device) for x in outputs["hidden_states"]]
        hidden_state = torch.cat(outputs["hidden_states"], dim=-1)

    logits = tree_logits[0, retrieve_indices]
    return logits, hidden_state, outputs





def evaluate_posterior(
        logits: torch.Tensor,
        candidates: torch.Tensor,
        logits_processor,
        hsd=False,
):
    """
    Evaluate the posterior probabilities of the candidates based on the provided logits and choose the best candidate.

    Depending on the temperature value, the function either uses greedy decoding or evaluates posterior
    probabilities to select the best candidate.

    Args:
    - logits (torch.Tensor): Predicted logits of shape (batch_size, sequence_length, vocab_size).
    - candidates (torch.Tensor): Candidate token sequences.
    - temperature (float): Softmax temperature for probability scaling. A value of 0 indicates greedy decoding.
    - posterior_threshold (float): Threshold for posterior probability.
    - posterior_alpha (float): Scaling factor for the threshold.

    Returns:
    - best_candidate (torch.Tensor): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    """
    # Greedy decoding based on temperature value
    if logits_processor is None:
        # Find the tokens that match the maximum logits for each position in the sequence
        posterior_mask = (
                candidates[:, 1:].to(logits.device) == torch.argmax(logits[:, :-1], dim=-1)
        ).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, accept_length, logits[best_candidate, accept_length]

    elif not hsd:
        accept_length = 1
        accept_cand = candidates[0][:1]
        best_candidate = 0
        for i in range(1, candidates.shape[1]):
            if i != accept_length:
                break
            adjustflag = False
            is_eq = (candidates[:, :accept_length] == accept_cand).all(dim=1)
            fi = torch.nonzero(is_eq, as_tuple=True)[0][0]
            gt_logits = logits[fi, i - 1][None]
            gt_logits = logits_processor(None, gt_logits)[0]
            gtp = torch.softmax(gt_logits, dim=0)
            candidates_set = []
            for j in range(candidates.shape[0]):
                if is_eq[j]:
                    x = candidates[j, i]
                    xi = x.item()
                    # avoid considering the same token twice, which will affect the distribution
                    if xi in candidates_set or xi == -1:
                        continue
                    candidates_set.append(xi)
                    r = random.random()
                    px = gtp[xi]
                    qx = 1.0
                    acp = px / qx
                    if r <= acp:
                        accept_cand = torch.cat((accept_cand, x[None]), dim=0)
                        accept_length += 1
                        best_candidate = j
                        break
                    else:
                        gtp[xi] = 0
                        gtp = gtp / gtp.sum()
                        adjustflag = True
        if adjustflag and accept_length != candidates.shape[1]:
            sample_p = gtp
        else:
            gt_logits = logits[best_candidate, accept_length - 1][None]
            gt_logits = logits_processor(None, gt_logits)[0]
            sample_p = torch.softmax(gt_logits, dim=0)
        return torch.tensor(best_candidate), accept_length - 1, sample_p

    else:
        logits = logits_processor(None, logits)
        p = logits.softmax(dim=-1) if "mps" not in str(logits.device) else logits.softmax(dim=-1)
        n_matches = 1
        current_step_match = 0
        ind = 0
        candidate_length = candidates.shape[1]

        for b in range(candidates.shape[0]):

            if (candidates[ind:ind+1, :-(candidate_length - n_matches)] == candidates[b:b+1, :-(candidate_length - n_matches)]).all():
                ind = b
            else:
                continue

            # avoid token id = -1 due to the batch generation after eos token, but the eos token is then not counted into the number of accepted tokens, 
            # affecting a fair comparison of block efficiency
            new_candidate_length = (candidates[ind:ind + 1, -candidate_length:] != -1).sum().item()

            if new_candidate_length == candidate_length:
                new_candidate_input_ids = candidates[ind:ind + 1, -(candidate_length - n_matches):]
            else:
                new_candidate_input_ids = candidates[ind:ind + 1,
                                          -(candidate_length - n_matches):-(
                                                      candidate_length - new_candidate_length)]

            # Gets the probabilities from the logits. q_i and p_i denote the assistant and model probabilities of the tokens
            # selected by the assistant, respectively.
            # to be more precise using double precision
            # mps does not support float64 tensor

            # before squeeze, the shape is (1, 1, candidate_length)
            # after squeeze: (1, candidate_length)

            if b > 0:

                def zero_after_first_zero(x):
                    # Create a mask for where x is zero
                    zero_mask = (x == 0)

                    # Find the index of the first zero in each row
                    first_zero_idx = zero_mask.float().cumsum(dim=1).clamp(max=1)

                    # Invert the mask so that all elements after the first zero become 0
                    keep_mask = (first_zero_idx == 0).float().cumsum(dim=1).clamp(max=1)

                    return x * keep_mask

                # if not using clone this just returns a view not copy
                p_new = p[ind:ind + 1, n_matches-1:new_candidate_length-1].clone()
                p_new[:, 0] = p_primes[:, current_step_match].clone()

                p_new_sum = p_new.sum(-1, keepdim=True)
                # there is already nan in p_new_sum even before deviding by zero
                # probably due to underflow or division by zero
                # if divided by 0, the probabilities will become nan, and leading to strange strings
                p_new_sum[p_new_sum == 0] = 1
                # p_primes evaluated on the draft traject could be zero, if the prefix already becomes with r<1 at some position
                p_new = p_new.div(p_new_sum)
                # p_primes are always truncated in the previous non-first step
                p_i = p_new[:, torch.arange(new_candidate_length - n_matches), new_candidate_input_ids].squeeze(1)
                p_i = zero_after_first_zero(p_i)
                # i can only sample from the tokens which always has prefix with r>1
                # this sequential sample manner makes it different from parrallel computing r^i>1
                # because the distribution becomes even sparser:
                # so p_i also has to be adjusted,
                # can't use one_hot to -1 tokens

                q_i = torch.ones(1, new_candidate_length-n_matches).to(logits.device)
                q_previous = torch.roll(q_i, 1, 1)
                q_previous[:, 0] = joint_q_previous[:, current_step_match]

                # here our p_res is joint probability from 1 to n_matches, not the tokenwise probability
                joint_q_previous = torch.exp(torch.log(q_previous).cumsum(1).unsqueeze(-1))
                q_next = joint_q_previous * torch.nn.functional.one_hot(new_candidate_input_ids, num_classes=logits.shape[-1])

                # p_i has to be re-selected after normalizing p_primes
                # p_primes has to be re-normalized after applying zero masks, before applied with p_previous

                p_previous = torch.roll(p_i, 1, 1)
                p_previous[:, 0] = joint_p_previous[:, current_step_match]

                # p_primes doesn't sum to one, because they are just sub-branches of the full joint distribution tree
                # i have to normalize them to sum=1 when treating as marginal probabilities, just as when i do the resampling during forward sampling
                # i want to control the joint prob ratio of the preceding draft tokens to be <=1,
                joint_p_previous = torch.exp(torch.log(p_previous).cumsum(1)).unsqueeze(-1)

                # ratio = joint_p_previous / joint_q_previous

                # previous_max = 1
                # new_p_previous = torch.ones_like(joint_p_previous).to(joint_p_previous.device)
                # for k in range(new_candidate_length-n_matches):
                #     if ratio[:, k] > previous_max:
                #         previous_max = ratio[:, k]

                #     new_p_previous[:, k] = joint_p_previous[:, k] / previous_max



                # p_next = new_p_previous * p_new

                p_next = torch.minimum(p_previous.cumprod(-1), q_previous.cumprod(-1)).unsqueeze(-1) * p_new

            else:
                # # for multidraft
                q_i = torch.ones(1, new_candidate_length-n_matches).to(logits.device)

                q_previous = torch.roll(q_i, 1, 1)
                q_previous[:, 0] = 1
                joint_q_previous = torch.exp(torch.log(q_previous).cumsum(1).unsqueeze(-1))
                q_next = joint_q_previous * torch.nn.functional.one_hot(new_candidate_input_ids, num_classes=logits.shape[-1])
                # else:
                # p_i corresponds to marginal probability
                p_i = p[ind:ind + 1, torch.arange(n_matches-1, new_candidate_length-1),
                      new_candidate_input_ids].squeeze(1)

                p_previous = torch.roll(p_i, 1, 1)
                # in this case, we compensate the subbranch X^i with prefix r^i-1>1 cleverly,
                # then there's no need to do forward sampling
                p_previous[:, 0] = 1

                # i want to control the joint prob ratio of the preceding draft tokens to be <=1,
                joint_p_previous = torch.exp(torch.log(p_previous).cumsum(1)).unsqueeze(-1)

              
                # ratio = joint_p_previous / joint_q_previous

                # previous_max = 1
                # new_p_previous = torch.ones_like(joint_p_previous).to(joint_p_previous.device)
                # for k in range(new_candidate_length-1):
                #     if ratio[:, k] > previous_max:
                #         previous_max = ratio[:, k]

                #     new_p_previous[:, k] = joint_p_previous[:, k] / previous_max

                # p_next = new_p_previous * p[ind:ind + 1, :new_candidate_length-1]
                
                p_next = torch.minimum(log_p_previous, log_q_previous) * p[ind:ind + 1, :new_candidate_length-1]


            # be careful with the positions where diffs=0

            # calculate in log scale (because multiplication becomes addition in log space) to avoid underflow

            diffs = p_next - q_next

            # ratio range is expected to be smaller than absolute probability range, since all probabilities are smaller than 1
            p_plus, p_minus = torch.clamp(diffs, min=0), torch.clamp(-diffs, min=0)

            # to avoid underflow, try to formulate using r_previous
            # for extremely small probabilities of any draft in p_i or q_i, it could be 0 simply due to default truncation sampling
            # so i have to avoid the case of devided by 0! but shall i turn off truncation sampling for the models or not? maybe i just keep
            # their default parameters

            denominator = torch.maximum(p_plus.sum(dim=-1, keepdim=True), p_minus.sum(dim=-1, keepdim=True))
            # divide by zero will cause nan value, i can simply replace nan with 0
            # however, for clever sampling
            # if there differences are 0, it can also mean that  the joint distribution is exactly the same and could simply accept the current token

            # for tokenwise verification, it will never happen when both p_next and q_next=0, because the draft token is sampled from
            # q, thus never has probability >0
            # for my backward joint verfication with forward sampling, the previous token probability could be 0 (accumulated candidates contain resampled token in the forward sampling process) and then the joint probability q_i will always be
            # zero after this token, and a new token sampled from q could also be zero for p, if truncation sampling are applied for p,
            # in this case, we have two zero joint probabilities, causing the problem to happen
            p_primes = torch.nan_to_num(p_plus / denominator)

            # for recursive backward speculative, i actually have to reset the ratio of already accepted tokens to 1,

            # compute the residual probabilities for stepping back
            # for clever backward, we could accept the token at an intermediate step if p = q,
            # sine we assume p_previous always <=1 for clever backward, in this case p_primes.sum()=0, but we should not step back

            step_back_probs = 1 - p_primes.sum(dim=-1)


            step_back_probs[(p_previous / q_previous).cumprod(1) >= 1] = 0

            # randomly sample if stepping back, i.e., neither accepted, nor resampled
            uniform_rand = torch.rand_like(step_back_probs)

            step_back = uniform_rand < step_back_probs
            # find the last index of False value in step_back array, i.e., not stepping back
            # could be done by finding the first index of false value in the reversed step_back array
            if step_back.all():
                stop_positions = 0
            else:
                stop_positions = new_candidate_length - n_matches - 1 - \
                             torch.flip(~step_back, [-1]).max(-1, keepdim=True)[1]

            # create the mask for selecting the elements after different stop positions at each row
            select = torch.zeros_like(step_back).to(step_back.device)

            # apply cumprod on the ratio instead of the raw probabilities to avoid underflow
            # it has to be updated for multidraft!!!
            probability_ratio = (p_i / q_i).cumprod(1).unsqueeze(-1)

            # When probability_ratio > 1 (i.e. q_i(x) < p_i(x), or "assistant probability of the candidate token is smaller
            # than the model probability for the same token"), keep the token. Otherwise reject with p = 1 - probability_ratio
            # (= keep with p = probability_ratio). Keep all the tokens until the first rejection

            r_i = torch.rand_like(probability_ratio)
            is_accepted = r_i <= probability_ratio

            # only decide to accept or not at the last position based on the joint probability ratio
            # assign 0 to all positions when the full draft is rejected, otherwise assign 1 to the rest of the positions
            select[torch.arange(p_primes.shape[0]), stop_positions] = ~is_accepted[:, -1:]
            is_accepted = 1 - torch.cumsum(select, dim=-1)

            #### assume batch_size=1 for the current implementation
            current_step_match = is_accepted.sum().item()

            n_matches += current_step_match

            if n_matches == candidate_length:
                break

        # Next token selection: if there is a rejection, adjust the distribution from the main model before sampling.
        gamma = new_candidate_length
        p_n_plus_1 = p[ind:ind + 1, new_candidate_length-1, :]
        if n_matches < gamma:
            # then we don't have a bonus token, and start the resample from the step where we don't step back, i.e., n_matches+1
            # don't use in_place operation! because slicing is not creating a new tensor
            p_prime = p_primes[:, current_step_match]

            if p_prime.sum() == 0:
                if n_matches + 1 < new_candidate_length:
                    p_prime = torch.nn.functional.one_hot(candidates[ind:ind + 1, n_matches + 1],
                                                          num_classes=logits.shape[-1]).float()
                else:
                    p_prime = torch.nn.functional.one_hot(candidates[ind:ind + 1, n_matches],
                                                          num_classes=logits.shape[-1]).float()
            else:
                p_prime = p_prime.div(p_prime.sum())
        else:
            p_prime = p_n_plus_1

        return ind, n_matches-1, p_prime.squeeze()

@torch.no_grad()
def update_inference_inputs(
        input_ids,
        candidates,
        best_candidate,
        accept_length,
        retrieve_indices,
        logits_processor,
        new_token,
        past_key_values_data_list,
        current_length_data,
        model,
        hidden_state_new,
        sample_p
):
    prev_input_len = input_ids.shape[1]
    # Map the best candidate indices to the original indices in the sequence
    select_indices = (
            retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
    )
    # Append the tokens from the best candidate to the input sequence
    input_ids = torch.cat(
        [input_ids, candidates[None, best_candidate, : accept_length + 1].to(input_ids.device)], dim=-1
    )
    # Update the past key values based on the selected tokens
    # Source tensor that contains relevant past information based on the selected candidate
    for past_key_values_data in past_key_values_data_list:
        tgt = past_key_values_data[..., select_indices.to(past_key_values_data.device), :]
        # Destination tensor where the relevant past information will be stored
        dst = past_key_values_data[..., prev_input_len: prev_input_len + tgt.shape[-2], :]
        # Copy relevant past information from the source to the destination
        dst.copy_(tgt, non_blocking=True)

    # Update the current length tensor (currently only support batch size is 1)
    current_length_data.fill_(prev_input_len + tgt.shape[-2])

    retrieve_hidden_state_new = hidden_state_new[:, retrieve_indices]
    accept_hidden_state_new = retrieve_hidden_state_new[:, best_candidate, : accept_length + 1]
    # token=model.base_model.lm_head(accept_hidden_state_new[:,-1]).argmax()
    # token=token[None,None]
    prob = sample_p
    if logits_processor is not None:
        token = torch.multinomial(prob, 1)
        token = token[None]
    else:
        token = torch.argmax(prob)
        token = token[None, None]
    # hidden_state = torch.cat((hidden_state, accept_hidden_state_new), dim=1)
    draft_tokens, retrieve_indices,tree_mask,tree_position_ids = model.ea_layer.topK_genrate(accept_hidden_state_new,
                                              input_ids=torch.cat((input_ids, token.to(input_ids.device)), dim=1),
                                              head=model.base_model.lm_head,logits_processor=logits_processor)


    new_token += accept_length + 1

    return input_ids, draft_tokens, retrieve_indices,tree_mask,tree_position_ids, new_token, None, token


if __name__ == "__main__":
    logits = torch.randn(1, 5)
    tp = prepare_logits_processor(0.9, 0, 0.9, 0)
    l = tp(None, logits)
    if tp is None:
        print(tp)
