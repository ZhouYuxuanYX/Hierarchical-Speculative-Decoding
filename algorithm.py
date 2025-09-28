import torch

def _speculative_sampling(
        candidate_input_ids,
        candidate_logits,
        candidate_length,
        new_logits,
        is_done_candidate,
        backward=False,
        return_probs=False,
):
    q = candidate_logits.softmax(dim=-1).double() if "mps" not in str(
        candidate_logits.device) else candidate_logits.softmax(dim=-1)
    p = new_logits.softmax(dim=-1).double() if "mps" not in str(new_logits.device) else new_logits.softmax(dim=-1)

    q_i = q[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(1)

    q_previous = torch.roll(q_i, 1, 1)
    q_previous[:, 0] = 1
    joint_q_previous = torch.exp(torch.log(q_previous).cumsum(1).unsqueeze(-1))
    q_next = joint_q_previous *q[:, :candidate_length]


    # p_i corresponds to marginal probability
    p_i = p[:, torch.arange(hist_length, candidate_length), new_candidate_input_ids].squeeze(1)     
    p_previous = torch.roll(p_i, 1, 1)
    p_previous[:, 0] = 1

    joint_p_previous = torch.exp(torch.log(p_previous).cumsum(1)).unsqueeze(-1)

    # compute capped target probability to avoid extra resampling steps
    ratio = joint_p_previous / joint_q_previous
    previous_max = 1
    new_p_previous = torch.ones_like(joint_p_previous).to(joint_p_previous.device)
    for k in range(candidate_length):
        if ratio[:, k] > previous_max:
                previous_max = ratio[:, k]
                new_p_previous[:, k] = joint_p_previous[:, k] / previous_max
    p_next = new_p_previous * p[:, :candidate_length]

    # compute divergence for accept and resample probability
    diffs = p_next - q_next
    p_plus, p_minus = torch.clamp(diffs, min=0), torch.clamp(-diffs, min=0)

    denominator = torch.maximum(p_plus.sum(dim=-1, keepdim=True), p_minus.sum(dim=-1, keepdim=True))

    # compute the unnormalized resampling probability, whose sum equals the accept probability
    p_primes = p_plus / denominator

    # compute the reject probability at t<gamma
    step_back_probs = 1 - p_primes.sum(dim=-1)

    # randomly sample if accept or not
    uniform_rand = torch.rand_like(step_back_probs)
    step_back = uniform_rand < step_back_probs

    # find the last index of False value in step_back array, i.e., not stepping back
    # could be done by finding the first index of false value in the reversed step_back array
    if step_back.all():
        stop_positions = 0
    else:
        stop_positions = candidate_length - 1 - torch.flip(~step_back, [-1]).max(-1, keepdim=True)[1]

    # create the mask for selecting the elements after different stop positions at each row
    select = torch.zeros_like(step_back).to(step_back.device)

    # apply cumprod on the ratio instead of the raw probabilities to avoid underflow
    probability_ratio = (p_i / q_i).cumprod(1).unsqueeze(-1)

    # compute the accept probability for t=gamma
    r_i = torch.rand_like(probability_ratio)
    is_accepted = r_i <= probability_ratio

    # only decide to accept or not at the last position based on the joint probability ratio
    # assign 0 to all positions when the full draft is rejected, otherwise assign 1 to the rest of the positions
    select[torch.arange(p_primes.shape[0]), stop_positions] = ~is_accepted[:, -1:]
    is_accepted = 1 - torch.cumsum(select, dim=-1)

    # assume batch_size=1 for the current implementation
    n_matches = is_accepted.sum().item()

    
    # for clever, p_prime (regard previous ratio as 1) could be 0, it means p=q, and we don't need to resample
    if is_done_candidate[ind:ind+1] and n_matches == candidate_length:
            # Output length is assumed to be `n_matches + 1`. Since we won't generate another token with the target model
            # due to acceptance on EOS we fix `n_matches`
            n_matches -= 1
            valid_tokens = candidate_input_ids[:, -candidate_length:]
    else:
        # Next token selection: if there is a rejection, adjust the distribution from the main model before sampling.
        p_n_plus_1 = p[:, candidate_length, :]
        if n_matches < candidate_length:
            # then we don't have a bonus token, and start the resample from the step where we don't step back, i.e., n_matches+1
            # don't use in_place operation! because slicing is not creating a new tensor
            p_prime = p_primes[:, n_matches]
            p_prime = p_prime.div(p_prime.sum())
        else:
            p_prime = p_n_plus_1

        if n_matches > 0 and n_matches<candidate_length:
            valid_tokens = candidate_input_ids[:, -candidate_length:n_matches-candidate_length]
            t = torch.multinomial(p_prime, num_samples=1)
            valid_tokens = torch.cat((valid_tokens, t), dim=-1)
        else:
            if n_matches==0:
                valid_tokens = t
            else:
                valid_tokens = candidate_input_ids[:, -candidate_length:]
                valid_tokens = torch.cat(
                    (valid_tokens, t), dim=-1)
                    
    return valid_tokens, n_matches
