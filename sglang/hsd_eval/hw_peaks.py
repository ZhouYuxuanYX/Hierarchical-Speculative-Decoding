"""Achievable memory bandwidth and bf16 GEMM throughput on this GPU (the roofline's two ceilings)."""
import torch, time
torch.backends.cuda.matmul.allow_tf32 = False
dev = "cuda"
def timeit(fn, reps=20):
    fn(); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps
print(torch.cuda.get_device_name())
x = torch.empty(4 * 1024**3 // 2, dtype=torch.bfloat16, device=dev).normal_()
t = timeit(lambda: x.sum(dtype=torch.float32))
print(f"read bandwidth: {x.numel() * 2 / t / 1e9:.0f} GB/s")
# Llama-3.1-8B MLP / attention projection shapes, M = tokens in the pass
for M in (16, 64, 128, 256, 512, 1024, 2720, 8192):
    tf = []
    for K, N in ((4096, 14336), (14336, 4096), (4096, 6144)):
        a = torch.randn(M, K, dtype=torch.bfloat16, device=dev); b = torch.randn(K, N, dtype=torch.bfloat16, device=dev)
        t = timeit(lambda: a @ b)
        tf.append(2 * M * K * N / t / 1e12)
    print(f"GEMM M={M:5d}: {min(tf):6.1f}-{max(tf):6.1f} TFLOPS")
