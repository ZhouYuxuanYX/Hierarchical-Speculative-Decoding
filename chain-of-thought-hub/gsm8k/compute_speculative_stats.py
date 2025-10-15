import orjson
import numpy as np
import matplotlib.pyplot as plt
import mmap
import time
import re

# too large to fit into the local machine ram
# orjson is a super-fast alternative to the standard json module.
# It’s written in Rust and much faster for both reading and writing.
# this is faster than multiprocessing, 820s vs 1729.9 (multiprocessing) seconds!!!
# memoryview is also slower than mm[:]

models = "Qwen_72B_0.5B_"
gamma = 10


def safe_orjson_load(mm_bytes: bytes):
    """Try orjson.loads, fall back to sanitizing NaN/Infinity if needed."""
    try:
        return orjson.loads(mm_bytes)
    except orjson.JSONDecodeError:
        text = mm_bytes.decode("utf-8", errors="replace")
        text = re.sub(r"\bNaN\b", "null", text)
        text = re.sub(r"\bInfinity\b", "1e9999", text)
        text = re.sub(r"\b-Infinity\b", "-1e9999", text)
        return orjson.loads(text.encode("utf-8"))


start = time.time()
with open(f"{models}naive_gamma_{gamma}_total_counts.json", "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    counts_naive = safe_orjson_load(mm[:])   # ✅ fixed
    mm.close()
#
# with open(f"{models}backward_gamma_10_total_counts.json", "rb") as f:
#     mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
#     counts_backward = safe_orjson_load(mm[:])   # ✅ fixed
#     mm.close()
#
# with open(f"{models}backward_recursive_gamma_10_total_counts.json", "rb") as f:
#     mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
#     counts_backward_recursive = safe_orjson_load(mm[:])   # ✅ fixed
#     mm.close()

with open(f"{models}backward_clever_gamma_{gamma}_total_counts.json", "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    counts_backward_clever = safe_orjson_load(mm[:])   # ✅ fixed
    mm.close()
#
# with open(f"{models}backward_clever_approxi_gamma_10_total_counts.json", "rb") as f:
#     mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
#     counts_backward_clever_approxi = safe_orjson_load(mm[:])   # ✅ fixed
#     mm.close()

with open(f"{models}blockwise_gamma_{gamma}_total_counts.json", "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    counts_blockwise = safe_orjson_load(mm[:])   # ✅ fixed
    mm.close()

end = time.time()
print(end - start)


counts = [
    counts_naive,
    # counts_backward, counts_backward_recursive,
    counts_backward_clever,
    # counts_backward_clever_approxi,
    counts_blockwise,
]


draft_eval = []
target_eval = []
total_step = []
sample_length = []
times = []
lens = []
lens_gamma = []


for count in counts:
    draft = 0
    target = 0
    step = 0
    sample = 0
    time_ = 0
    len_ = 0
    len_gamma = 0
    token_ = 0
    calls = []

    for n in range(len(count["draft_eval"])):
        # exclude the draft lengths<10 cases for a fair comparison
        draft_list = np.array(count["draft_eval"][n])
        target_list = np.array(count["target_eval"][n])
        step_list = np.array(count["total_step"][n])
        sample_list = np.array(count["sample_length"][n])
        print("check length")
        print(len(sample_list))
        print(len(step_list))
        print(count["time"][n])

        draft += draft_list[draft_list == gamma].sum()
        target += target_list[draft_list == gamma].sum()
        step += step_list[draft_list == gamma].sum()
        sample += sample_list[draft_list == gamma].sum()
        len_ += len(sample_list[draft_list == gamma])
        len_gamma += len(sample_list)

        calls.append(np.array(sample_list[draft_list == gamma]).mean())

        token_ += sample_list.sum()
        time_ += float(count["time"][n])*sample_list[draft_list == gamma].sum()/sample_list.sum()

    draft_eval.append((draft / len_))
    target_eval.append((target / len_))
    total_step.append((step / len_))
    sample_length.append((sample / len_))
    times.append(time_)
    lens.append(token_)

print("efficiency")
print(np.array(lens) / np.array(times))

draft_eval = np.array(draft_eval)
target_eval = np.array(target_eval)
total_step = np.array(total_step)
sample_length = np.array(sample_length)
times = np.array(times)


x = np.arange(len(counts))  # [0, 1, 2, 3]
width = 0.3  # Width of the bars

# Create plot
fig, ax = plt.subplots()
# bar1 = ax.bar(x - width, [a.sum() for a in times], width, label='List 1')
bar2 = ax.bar(x, [a.sum() for a in target_eval], width, label="List 2")
bar3 = ax.bar(x + width, [a.sum() for a in sample_length], width, label="List 3")

# Add labels, title, and legend
ax.set_ylabel("Scores")
ax.set_title("Comparison of Two Lists")
ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.legend()

ax.bar_label(bar2, padding=3)
ax.bar_label(bar3, padding=3)

plt.tight_layout()
plt.savefig(f"{models}compare_efficiency_gamma_10.png")
