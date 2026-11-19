# Time of one Llama-3.1-8B forward pass (bf16) vs number of new tokens N, with a 300-token KV context.
import torch, time
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16).cuda().eval()
print(torch.cuda.get_device_name())
ctx = torch.randint(0, 120000, (1, 300), device="cuda")
with torch.no_grad():
    past = m(ctx, use_cache=True).past_key_values
    base = past.to_legacy_cache() if hasattr(past, "to_legacy_cache") else past
    from transformers import DynamicCache
    for N in [1, 7, 16, 32, 64, 85, 128, 170, 224, 256, 384, 512, 680, 1024, 1360, 2048, 2720]:
        ids = torch.randint(0, 120000, (1, N), device="cuda")
        ts = []
        for r in range(6):
            cache = DynamicCache.from_legacy_cache(tuple((k.clone(), v.clone()) for k, v in base))
            torch.cuda.synchronize(); t = time.perf_counter()
            m.model(ids, past_key_values=cache, use_cache=True)  # no lm_head: SGLang only projects needed rows
            torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
        print(f"N={N:5d}  {sorted(ts)[2]*1e3:7.2f} ms", flush=True)
