import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# Settings
draft_length = 10
trials = 10000
vocab_size = 2

# Probability distributions
Ms_probs = np.array([2/3, 1/3])
Mb_probs = np.array([1/3, 2/3])

def clever_old(candidate_input_ids, candidate_logits, candidate_length, new_logits):

    q = candidate_logits
    p = new_logits

    new_candidate_input_ids = candidate_input_ids[:, -candidate_length:]
    # Gets the probabilities from the logits. q_i and p_i denote the assistant and model probabilities of the tokens
    # selected by the assistant, respectively.

    q_i = q[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(1)

    # # print("check distribution")
    # print(p[:, torch.arange(hist_length, candidate_length)].topk(10, dim=-1)[0])
    # print(q[:, torch.arange(hist_length, candidate_length)].topk(10, dim=-1)[0])

    q_previous = torch.roll(q_i, 1, 1)
    q_previous[:, 0] = 1
    log_q_previous = torch.exp(torch.log(q_previous).cumsum(1).unsqueeze(-1))
    q_next = log_q_previous * q[:, :candidate_length]



    if True:
        # do not cap the previous ratio
        if False:
            # p_i corresponds to marginal probability
            p_i = p[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(1)

            p_previous = torch.roll(p_i, 1, 1)
            # # print(p_previous.shape)
            # exit()
            p_previous[:, 0] = 1
            p_next = torch.exp(torch.log(p_previous).cumsum(1)).unsqueeze(-1) * p[:, :candidate_length]

        # cap the entire prefix
        else:
            # p_i corresponds to marginal probability
            p_i = p[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(1)

            p_previous = torch.roll(p_i, 1, 1)
            # in this case, we compensate the subbranch X^i with prefix r^i-1>1 cleverly,
            # then there's no need to do forward sampling
            p_previous[:, 0] = 1

            # i want to control the joint prob ratio of the preceding draft tokens to be <=1,
            log_p_previous = torch.exp(torch.log(p_previous).cumsum(1)).unsqueeze(-1)
            p_next = torch.minimum(log_p_previous, log_q_previous) * p[:, :candidate_length]






    else:
        # p_i corresponds to marginal probability
        p_i = p[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(1)

        p_previous = torch.roll(p_i, 1, 1)
        # # print(p_previous.shape)
        # exit()
        p_previous[:, 0] = 1
        p_next = torch.exp(torch.log(p_previous).cumsum(1)).unsqueeze(-1) * p[:, :candidate_length]

    diffs = p_next - q_next

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


    # this is needed for clever algorithm, to avoied when p_prime=0, when i forced p_previous=q_previous when p_previous>q_previsou
    # and p_current = q_current
    step_back_probs[(p_i / q_i).cumprod(1) >= 1] = 0

    # randomly sample if stepping back, i.e., neither accepted, nor resampled
    uniform_rand = torch.rand_like(step_back_probs)

    step_back = uniform_rand < step_back_probs

    stop_positions = candidate_length - 1 - torch.flip(~step_back, [-1]).max(-1, keepdim=True)[1]

    select = torch.zeros_like(step_back).to(step_back.device)

    # apply cumprod on the ratio instead of the raw probabilities to avoid underflow
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
    n_matches = is_accepted.sum().item()



    # for clever, p_prime (regard previous ratio as 1) could be 0, it means p=q, and we don't need to resample
    if n_matches < candidate_length and p_primes[:, n_matches].sum() == 0:
        n_matches += 1
        valid_tokens = new_candidate_input_ids[:, : n_matches]

    else:
        # Next token selection: if there is a rejection, adjust the distribution from the main model before sampling.
        gamma = candidate_logits.shape[1]
        # # print("check gamma")
        # print(gamma)
        # # print("check p")
        # print(p.shape)
        # print(candidate_length)
        p_n_plus_1 = p[:, candidate_length, :]
        if n_matches < gamma:
            # then we don't have a bonus token, and start the resample from the step where we don't step back, i.e., n_matches+1
            # # print("check p_prime")
            # print(p_primes)
            # don't use in_place operation! because slicing is not creating a new tensor
            p_prime = p_primes[:, n_matches]

            p_prime = p_prime.div(p_prime.sum())

        else:
            p_prime = p_n_plus_1
            # # # print("else")
            # # print(p_prime.shape)

        t = torch.multinomial(p_prime, num_samples=1).squeeze(1)[None, :]

        # The selected tokens include the matches (if any) plus the next sampled tokens
        if n_matches > 0:
            valid_tokens = torch.cat((new_candidate_input_ids[:, :n_matches], t), dim=-1)
        else:
            valid_tokens = t


    return valid_tokens, n_matches

# Method 1:
def clever(candidate_input_ids, candidate_logits, candidate_length, new_logits):

    q = candidate_logits
    p = new_logits

    new_candidate_input_ids = candidate_input_ids[:, -candidate_length:]
    # Gets the probabilities from the logits. q_i and p_i denote the assistant and model probabilities of the tokens
    # selected by the assistant, respectively.

    q_i = q[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(1)

    # # print("check distribution")
    # print(p[:, torch.arange(hist_length, candidate_length)].topk(10, dim=-1)[0])
    # print(q[:, torch.arange(hist_length, candidate_length)].topk(10, dim=-1)[0])

    q_previous = torch.roll(q_i, 1, 1)
    q_previous[:, 0] = 1
    log_q_previous = torch.exp(torch.log(q_previous).cumsum(1).unsqueeze(-1))
    q_next = log_q_previous * q[:, :candidate_length]



    if True:
        # do not cap the previous ratio
        if False:
            # p_i corresponds to marginal probability
            p_i = p[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(1)

            p_previous = torch.roll(p_i, 1, 1)
            # # print(p_previous.shape)
            # exit()
            p_previous[:, 0] = 1
            p_next = torch.exp(torch.log(p_previous).cumsum(1)).unsqueeze(-1) * p[:, :candidate_length]

        # cap the entire prefix
        elif False:
            # p_i corresponds to marginal probability
            p_i = p[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(1)

            p_previous = torch.roll(p_i, 1, 1)
            # in this case, we compensate the subbranch X^i with prefix r^i-1>1 cleverly,
            # then there's no need to do forward sampling
            p_previous[:, 0] = 1

            # i want to control the joint prob ratio of the preceding draft tokens to be <=1,
            log_p_previous = torch.exp(torch.log(p_previous).cumsum(1)).unsqueeze(-1)

            p_next = torch.minimum(log_p_previous, log_q_previous) * p[:, :candidate_length]
            # p_next = log_p_previous, log_q_previous * p[:, :candidate_length]

        # cap the maximum prefix ratio
        else:
            # p_i corresponds to marginal probability
            p_i = p[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(1)

            p_previous = torch.roll(p_i, 1, 1)
            # in this case, we compensate the subbranch X^i with prefix r^i-1>1 cleverly,
            # then there's no need to do forward sampling
            p_previous[:, 0] = 1

            # i want to control the joint prob ratio of the preceding draft tokens to be <=1,
            log_p_previous = torch.exp(torch.log(p_previous).cumsum(1)).unsqueeze(-1)

            # p_next = torch.minimum(log_p_previous, log_q_previous) * p[:, :candidate_length]
            ratio = log_p_previous / log_q_previous

            previous_max = 1
            new_p_previous = torch.ones_like(log_p_previous).to(log_p_previous.device)
            for k in range(candidate_length):
                if ratio[:, k] > previous_max:
                    previous_max = ratio[:, k]

                new_p_previous[:, k] = log_p_previous[:, k] / previous_max

            p_next =  new_p_previous * p[:, :candidate_length]





    else:
        # p_i corresponds to marginal probability
        p_i = p[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(1)

        p_previous = torch.roll(p_i, 1, 1)
        # # print(p_previous.shape)
        # exit()
        p_previous[:, 0] = 1
        p_next = torch.exp(torch.log(p_previous).cumsum(1)).unsqueeze(-1) * p[:, :candidate_length]

    diffs = p_next - q_next

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
    #p_primes = torch.nan_to_num(p_plus / denominator)
    p_primes = p_plus / denominator
    # for recursive backward speculative, i actually have to reset the ratio of already accepted tokens to 1,

    # compute the residual probabilities for stepping back
    # for clever backward, we could accept the token at an intermediate step if p = q,
    # sine we assume p_previous always <=1 for clever backward, in this case p_primes.sum()=0, but we should not step back

    step_back_probs = 1 - p_primes.sum(dim=-1)


    # this is needed for clever algorithm, to avoied when p_prime=0, when i forced p_previous=q_previous when p_previous>q_previsou
    # and p_current = q_current
    # step_back_probs[(p_i / q_i).cumprod(1) >= 1] = 0

    # randomly sample if stepping back, i.e., neither accepted, nor resampled
    uniform_rand = torch.rand_like(step_back_probs)

    step_back = uniform_rand < step_back_probs

    # print(step_back)
    if step_back.all():
        stop_position=0
    else:
        stop_positions = candidate_length - 1 - torch.flip(~step_back, [-1]).max(-1, keepdim=True)[1]

    # print(stop_positions)

    select = torch.zeros_like(step_back).to(step_back.device)

    # apply cumprod on the ratio instead of the raw probabilities to avoid underflow
    probability_ratio = (p_i / q_i).cumprod(1).unsqueeze(-1)
    # When probability_ratio > 1 (i.e. q_i(x) < p_i(x), or "assistant probability of the candidate token is smaller
    # than the model probability for the same token"), keep the token. Otherwise reject with p = 1 - probability_ratio
    # (= keep with p = probability_ratio). Keep all the tokens until the first rejection

    r_i = torch.rand_like(probability_ratio)
    is_accepted = r_i <= probability_ratio


    # only decide to accept or not at the last position based on the joint probability ratio
    # assign 0 to all positions when the full draft is rejected, otherwise assign 1 to the rest of the positions
    select[torch.arange(p_primes.shape[0]), stop_positions] = ~is_accepted[:, -1:]
    # print(is_accepted[:, -1:])
    is_accepted = 1 - torch.cumsum(select, dim=-1)

    #### assume batch_size=1 for the current implementation
    n_matches = is_accepted.sum().item()

    # print(n_matches)
    # print(probability_ratio)



    # for clever, p_prime (regard previous ratio as 1) could be 0, it means p=q, and we don't need to resample
    # if n_matches < candidate_length and p_primes[:, n_matches].sum() == 0:
    #     n_matches += 1
    #     valid_tokens = new_candidate_input_ids[:, : n_matches]
    #
    # else:

    # Next token selection: if there is a rejection, adjust the distribution from the main model before sampling.
    gamma = candidate_logits.shape[1]
    # # print("check gamma")
    # print(gamma)
    # # print("check p")
    # print(p.shape)
    # print(candidate_length)
    p_n_plus_1 = p[:, candidate_length, :]
    if n_matches < gamma:
        # then we don't have a bonus token, and start the resample from the step where we don't step back, i.e., n_matches+1
        # # print("check p_prime")
        # print(p_primes)
        # don't use in_place operation! because slicing is not creating a new tensor
        p_prime = p_primes[:, n_matches]

        p_prime = p_prime.div(p_prime.sum())

    else:
        p_prime = p_n_plus_1
        # # # print("else")
        # # print(p_prime.shape)

    t = torch.multinomial(p_prime, num_samples=1).squeeze(1)[None, :]

    # The selected tokens include the matches (if any) plus the next sampled tokens
    if n_matches > 0:
        valid_tokens = torch.cat((new_candidate_input_ids[:, :n_matches], t), dim=-1)
    else:
        valid_tokens = t


    return valid_tokens, n_matches

# Method 2: Mock a "lenient" acceptance (e.g., 80% chance to keep going on mismatch)
def block(candidate_input_ids, candidate_logits, candidate_length, new_logits):
    q = candidate_logits
    q = F.pad(q, pad=(0, 0, 0, 1), mode='constant', value=0)

    # print("debug")
    # print(q[:, -1].sum())

    new_candidate_input_ids = candidate_input_ids[:, -candidate_length:]
    token_sequence = []  # Will include the token sequence we return

    q_i = q[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(0, 1)
    p = new_logits
    p_i = p[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(0, 1)
    probability_ratio = p_i / q_i
    accept_probability = 1

    vocab_size = q.shape[-1]

    reject_probs = []
    for token_index in range(candidate_length + 1):

        # Unnormalized residual probability
        sampling_weights = torch.maximum(torch.zeros_like(p[:, token_index]),
                                         p[:, token_index] * accept_probability - q[:, token_index])
        # unnormalized reject probability
        reject = torch.tensor([1 - accept_probability])[None, :].to(sampling_weights.device)

        # print(weights)
        # print(weights.sum())

        # if could happen that when p exactly equals to q at every position, especially for temperature=0
        # the sampling_weights will sum to zero
        if token_index < candidate_length:
            weights = torch.cat([sampling_weights, reject], dim=-1)
            if weights.sum().item() == 0:
                # this means always accept the previous token, same effect as if chosen_token.item() < vocab_size in the other case
                valid_tokens = new_candidate_input_ids[:, :token_index + 1]
                n_matches = token_index + 1
            else:
                weights = weights / weights.sum()

                chosen_token = torch.multinomial(weights, num_samples=1).squeeze(1)[None, :]

                if chosen_token.item() < vocab_size:
                    valid_tokens = torch.cat([new_candidate_input_ids[:, :token_index], chosen_token], dim=-1)
                    n_matches = token_index

            reject_probs.append(weights[0, -1].cpu().item())
        else:
            # h_gamma = p_gamma
            u_gamma = torch.rand(1)
            is_accepted = u_gamma >= reject.cpu()

            if is_accepted:
                chosen_token = torch.multinomial(p[:, token_index], num_samples=1).squeeze(1)[None, :]

                # if the last token is eos token, then the probability will be all zero for the next token
                # however, the torch.multinomial function will ignore the error and generate random tokens, thus causing the issue

                valid_tokens = torch.cat([new_candidate_input_ids[:, :token_index], chosen_token], dim=-1)
                n_matches = token_index
            reject_probs.append(reject[0, 0].cpu().item())

        # no probability ratio for the bonus token
        if token_index < candidate_length:
            accept_probability = min(1, probability_ratio[token_index] * accept_probability)

    return valid_tokens, n_matches

def naive(candidate_input_ids, candidate_logits, candidate_length, new_logits):
    new_candidate_input_ids = candidate_input_ids[:, -candidate_length:]
    # Gets the probabilities from the logits. q_i and p_i denote the assistant and model probabilities of the tokens
    # selected by the assistant, respectively.
    q = candidate_logits

    q_i = q[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(0, 1)
    p = new_logits

    p_i = p[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(0, 1)
    probability_ratio = p_i / q_i

    # When probability_ratio > 1 (i.e. q_i(x) < p_i(x), or "assistant probability of the candidate token is smaller
    # than the model probability for the same token"), keep the token. Otherwise reject with p = 1 - probability_ratio
    # (= keep with p = probability_ratio). Keep all the tokens until the first rejection
    r_i = torch.rand_like(probability_ratio)
    is_accepted = r_i <= probability_ratio

    n_matches = ((~is_accepted).cumsum(dim=-1) < 1).sum()  # this is `n` in algorithm 1

    # Next token selection: if there is a rejection, adjust the distribution from the main model before sampling.
    gamma = candidate_logits.shape[1]
    p_n_plus_1 = p[:, n_matches, :]
    if n_matches < gamma:
        q_n_plus_1 = q[:, n_matches, :]
        p_prime = torch.clamp((p_n_plus_1 - q_n_plus_1), min=0)
        p_prime.div_(p_prime.sum())
    else:
        p_prime = p_n_plus_1
    t = torch.multinomial(p_prime, num_samples=1).squeeze(1)[None, :]

    # The selected tokens include the matches (if any) plus the next sampled tokens
    if n_matches > 0:
        valid_tokens = torch.cat((new_candidate_input_ids[:, :n_matches], t), dim=-1)
    else:
        valid_tokens = t


    return valid_tokens, n_matches

# Simulation runner
def simulate_spec_sampling(method_func):
    match_lengths = []
    for _ in range(trials):
        candidate_tokens = np.random.choice(vocab_size, size=draft_length, p=Ms_probs)
        candidate_tokens = torch.tensor(candidate_tokens).unsqueeze(0)

        candidate_probs = np.tile(Ms_probs, (draft_length, 1))
        candidate_probs = torch.tensor(candidate_probs).unsqueeze(0)

        new_probs = np.tile(Mb_probs, (draft_length+1, 1))
        new_probs = torch.tensor(new_probs).unsqueeze(0)
        _, n_matches = method_func(candidate_tokens, candidate_probs, draft_length, new_probs)
        match_lengths.append(n_matches)
    return match_lengths

# Run both simulations
matches_1 = simulate_spec_sampling(clever)
matches_2 = simulate_spec_sampling(block)
matches_3 = simulate_spec_sampling(naive)
matches_4 = simulate_spec_sampling(clever_old)

import seaborn as sns
sns.set(style="whitegrid")

plt.figure(figsize=(8, 4.5))
bins = np.arange(draft_length + 2) - 0.5

plt.hist(matches_1, bins=bins, alpha=0.6, label="Method 1: Backward", color="skyblue", edgecolor="black", linewidth=1.2)
plt.hist(matches_2, bins=bins, alpha=0.6, label="Method 2: Blockwise", color="salmon", edgecolor="black", linewidth=1.2)
plt.hist(matches_3, bins=bins, alpha=0.6, label="Method 3: Tokenwise", color="grey", edgecolor="black", linewidth=1.2)
plt.hist(matches_4, bins=bins, alpha=0.6, label="Method 4: Backward Old", color="orange", edgecolor="black", linewidth=1.2)

plt.xlabel("Accepted Token Length", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.title("Speculative Sampling Comparison (1000 Trials)", fontsize=14)
plt.xticks(np.arange(0, draft_length + 2), fontsize=10)
plt.yticks(fontsize=10)
plt.legend(fontsize=10)

# Clean up spines
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.tight_layout()
plt.savefig("refined_histogram.png", dpi=300)

# plt.show()

plt.figure(figsize=(8, 4.5))

def plot_ccdf(data, label, color):
    sorted_data = np.sort(data)
    ccdf = 1.0 - np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.step(sorted_data, ccdf, where='post', label=label, color=color, linewidth=2)

plot_ccdf(matches_1, "Method 1: HSD (ours)", "skyblue")
plot_ccdf(matches_2, "Method 2: Blockwise", "salmon")
plot_ccdf(matches_3, "Method 3: Tokenwise", "grey")
# plot_ccdf(matches_4, "Method 4: Backward old", "orange")

plt.xlabel(r"Accepted Length $\tau$", fontsize=12)
plt.ylabel(r"$P(t > \tau)$", fontsize=12)
plt.title("Empirical CCDF of Accepted Lengths", fontsize=14)
plt.xticks(np.arange(0, draft_length + 2), fontsize=10)
plt.yticks(fontsize=10)
plt.legend(fontsize=10)

# Clean up spines
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("refined_ccdf.png", dpi=300)



# Data
x_labels = ["tokenwise", "blockwise", "ours"]
y_values = [       # ideal (r_upper)
    5.137820725489648e-07,       # tokenwise
    1.1438590845320614e-05,      # blockwise
    1.83308e-05 , # ours
    # 0.36611327081148815
]

plt.figure(figsize=(6, 4))
bars = plt.bar(x_labels, y_values, color=[ "grey",  "skyblue", "salmon"])

# Log scale for better visual contrast
# plt.yscale("log")
plt.ylabel(r"$h(\mathbf{X}_{1:\gamma})$", fontsize="x-large")
# plt.title("Acceptance Probability Comparison")
# Annotate each bar
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
for bar, val in zip(bars, y_values):
    plt.text(bar.get_x() + bar.get_width() / 2, val * 1.1, f"{val:.1e}",
             ha='center', va='bottom', fontsize=16)

plt.tight_layout()
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.savefig("average_probs.png")



