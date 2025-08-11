import torch

def _speculative_sampling(
    candidate_input_ids,
    candidate_logits,
    candidate_length,
    new_logits,
    is_done_candidate,
    HSD=False,                  
    return_probs=False,
    hist_lengths=[0],
    approxi=False
):
    """
    Speculative sampling with two modes:
    - HSD=True  : Run our HSD algorithm
    - HSD=False : Run naive (default) algorithm
    """

    # q: assistant probs; p: target probs
    q = candidate_logits.softmax(dim=-1).double() if "mps" not in str(candidate_logits.device) else candidate_logits.softmax(dim=-1)
    p = new_logits.softmax(dim=-1).double() if "mps" not in str(new_logits.device) else new_logits.softmax(dim=-1)

    hist_length = hist_lengths[0] if hist_lengths else 0
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
    q_next = log_q_previous * q[:, hist_length:candidate_length]

    # do not cap the previous ratio
    if approxi:
        p_i = p[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(1)
        p_previous = torch.roll(p_i, 1, 1)
        p_previous[:, 0] = 1
        p_next = torch.exp(torch.log(p_previous).cumsum(1)).unsqueeze(-1) * p[:, :candidate_length]
    else:
        p_i = p[:, torch.arange(hist_length, candidate_length), new_candidate_input_ids].squeeze(1)
        p_previous = torch.roll(p_i, 1, 1)
        p_previous[:, 0] = 1
        log_p_previous = torch.exp(torch.log(p_previous).cumsum(1)).unsqueeze(-1)
        p_next = torch.minimum(log_p_previous, log_q_previous) * p[:, :candidate_length]

    diffs = p_next - q_next
    p_plus = torch.clamp(diffs, min=0)
    p_minus = torch.clamp(-diffs, min=0)
    denominator = torch.maximum(p_plus.sum(dim=-1, keepdim=True), p_minus.sum(dim=-1, keepdim=True))
    p_primes = torch.nan_to_num(p_plus / denominator)

    step_back_probs = 1 - p_primes.sum(dim=-1)
    if HSD:
        # HSD: cumulative ratio >= 1 → 不退回
        step_back_probs[(p_i / q_i).cumprod(1) >= 1] = 0

    step_back = torch.rand_like(step_back_probs) < step_back_probs
    stop_positions = candidate_length - hist_length - 1 - torch.flip(~step_back, [-1]).max(-1, keepdim=True)[1]

    select = torch.zeros_like(step_back, device=step_back.device)
    probability_ratio = (p_i / q_i).cumprod(1).unsqueeze(-1)
    is_accepted = torch.rand_like(probability_ratio) <= probability_ratio

    select[torch.arange(p_primes.shape[0]), stop_positions] = ~is_accepted[:, -1:]
    is_accepted = 1 - torch.cumsum(select, dim=-1)

    n_matches = is_accepted.sum().item()

    if n_matches < candidate_length - hist_length and p_primes[:, n_matches].sum() == 0:
        n_matches += 1
        valid_tokens = new_candidate_input_ids[:, :n_matches]
    else:
        gamma = candidate_logits.shape[1]
        p_n_plus_1 = p[:, candidate_length, :]
        if n_matches < gamma - hist_length:
            p_prime = p_primes[:, n_matches]
            p_prime = p_prime.div(p_prime.sum())
        else:
            p_prime = p_n_plus_1
        t = torch.multinomial(p_prime, num_samples=1).squeeze(1)[None, :]
        valid_tokens = torch.cat((new_candidate_input_ids[:, :n_matches], t), dim=-1) if n_matches > 0 else t

    return valid_tokens, n_matches
