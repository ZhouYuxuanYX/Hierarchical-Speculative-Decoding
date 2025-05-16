import torch

def _speculative_sampling(
        candidate_input_ids,
        candidate_logits,
        candidate_length,
        new_logits,
        is_done_candidate,
        backward=False,
        return_probs=False,
        hist_lengths=[0],
        blockwise=False,
        clever=False,
        approxi=False
):
    """
    For recursive algorithm, each time the p_primes are indeed tokenwise probability for each position, both the start position of draft tokens
    and the draft trajectory evolves every time:
        1.  - sample gamma draft tokens
            - calculate p_i and q_i from 0 - gamma, for this step we need full gamma length
            - calculate p_prime for determining step_back prob and resample prob
            - accept n tokens + resample 1 additional token

        2.  - sample gamma-n-1 tokens
            - calculate p_prime based on p_i and q_i for each newly sampled token, for this step we still need full gamma length
            - treat p_prime as p_i, q_i stays the same
            - calculate p_prime' for determining step_back prob and resample prob
            - accept m tokens, no bonus token unless gamma-n-1 tokens are accepted

        3.  - sample gamma-n-1-m tokens
            - calculate p_prime based on p_i and q_i for each previously sampled token, for this step we still need full gamma length
            - calculate p_prime' based on p_prime and q_i for each newly sampled token, for this step we need the previous n+1 tokens
            - treat p_prime as p_i, q_i stays the same
            - calculate p_prime'' for determining step_back prob and resample prob
    """

    q = candidate_logits.softmax(dim=-1).double() if "mps" not in str(
        candidate_logits.device) else candidate_logits.softmax(dim=-1)

    p = new_logits.softmax(dim=-1).double() if "mps" not in str(new_logits.device) else new_logits.softmax(dim=-1)
    hist_length=0


    ## try to avoid overflow!!!
    new_candidate_input_ids = candidate_input_ids[:, -candidate_length:]
    # Gets the probabilities from the logits. q_i and p_i denote the assistant and model probabilities of the tokens
    # selected by the assistant, respectively.
    # to be more precise using double precision
    # mps does not support float64 tensor


    # before squeeze, the shape is (1, 1, candidate_length)
    # after squeeze: (1, candidate_length)


    q_i = q[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(1)

    q_previous = torch.roll(q_i, 1, 1)
    q_previous[:, 0] = 1
    log_q_previous = torch.exp(torch.log(q_previous).cumsum(1).unsqueeze(-1))
    q_next = log_q_previous *q[:, hist_length:candidate_length]


    # do not cap the previous ratio
    if approxi:
        # p_i corresponds to marginal probability
        p_i = p[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(1)

        p_previous = torch.roll(p_i, 1, 1)
        # # print(p_previous.shape)
        # exit()
        p_previous[:, 0] = 1
        p_next = torch.exp(torch.log(p_previous).cumsum(1)).unsqueeze(-1) * p[:, :candidate_length]

    else:
        # p_i corresponds to marginal probability
        p_i = p[:, torch.arange(hist_length, candidate_length), new_candidate_input_ids].squeeze(1)

        p_previous = torch.roll(p_i, 1, 1)
        # in this case, we compensate the subbranch X^i with prefix r^i-1>1 cleverly,
        # then there's no need to do forward sampling
        p_previous[:, 0] = 1

        # i want to control the joint prob ratio of the preceding draft tokens to be <=1,
        log_p_previous = torch.exp(torch.log(p_previous).cumsum(1)).unsqueeze(-1)
        p_next = torch.minimum(log_p_previous, log_q_previous) * p[:, :candidate_length]

    diffs = p_next - q_next

    p_plus, p_minus = torch.clamp(diffs, min=0), torch.clamp(-diffs, min=0)


    denominator = torch.maximum(p_plus.sum(dim=-1, keepdim=True), p_minus.sum(dim=-1, keepdim=True))
    # divide by zero will cause nan value, i can simply replace nan with 0
    # however, for clever sampling
    # if there differences are 0, it can also mean that  the joint distribution is exactly the same and could simply accept the current token

    # for my backward joint verfication with forward sampling, the previous token probability could be 0 (accumulated candidates contain resampled token in the forward sampling process) and then the joint probability q_i will always be
    # zero after this token, and a new token sampled from q could also be zero for p, if truncation sampling are applied for p,
    # in this case, we have two zero joint probabilities, causing the problem to happen
    p_primes = torch.nan_to_num(p_plus / denominator)


    # we could accept the token at an intermediate step if p = q,
    # sine we assume p_previous always <=1 for clever backward, in this case p_primes.sum()=0, but we should not step back

    step_back_probs = 1 - p_primes.sum(dim=-1)


    if clever:
        # this is needed for clever algorithm, to avoid when p_prime=0, when i forced p_previous=q_previous when p_previous>q_previsou
        # and p_current = q_current
        step_back_probs[(p_i / q_i).cumprod(1)>=1] = 0

    # randomly sample if stepping back, i.e., neither accepted, nor resampled
    uniform_rand = torch.rand_like(step_back_probs)


    step_back = uniform_rand < step_back_probs

    # find the last index of False value in step_back array, i.e., not stepping back
    # could be done by finding the first index of false value in the reversed step_back array

    stop_positions = candidate_length-hist_length - 1 - torch.flip(~step_back, [-1]).max(-1, keepdim=True)[1]
    # # print("stop positions")
    # print(stop_positions)

    # create the mask for selecting the elements after different stop positions at each row
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
    if n_matches < candidate_length-hist_length and p_primes[:, n_matches].sum()==0:
        n_matches +=1
        valid_tokens = new_candidate_input_ids[:, : n_matches]

    else:
        # Next token selection: if there is a rejection, adjust the distribution from the main model before sampling.
        gamma = candidate_logits.shape[1]
        p_n_plus_1 = p[:, candidate_length, :]
        if n_matches < gamma - hist_length:
            # then we don't have a bonus token, and start the resample from the step where we don't step back, i.e., n_matches+1
            # don't use in_place operation! because slicing is not creating a new tensor
            p_prime = p_primes[:, n_matches]

            p_prime = p_prime.div(p_prime.sum())

        else:
            p_prime = p_n_plus_1

        t = torch.multinomial(p_prime, num_samples=1).squeeze(1)[None, :]


        # The selected tokens include the matches (if any) plus the next sampled tokens
        if n_matches > 0:
            valid_tokens = torch.cat((new_candidate_input_ids[:, :n_matches], t), dim=-1)
        else:
            valid_tokens = t


    return valid_tokens, n_matches
