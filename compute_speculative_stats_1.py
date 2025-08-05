import orjson
import numpy as np
import matplotlib.pyplot as plt
import mmap
import time
# too large to fit into the local machine ram
# orjson is a super-fast alternative to the standard json module.
# It’s written in Rust and much faster for both reading and writing.
# this is faster than multiprocessing, 820s vs 1729.9 (multiprocessing) seconds!!!
# memoryview is also slower than mm[:]

models = "Qwen_72B_0.5B_"
gamma = 10

start =time.time()


# top_p=0.8, temp=1.0:
# backward_clever: "AAA_last/Qwen_72B_0.5B_backward_clever_gamma_10_topp_0.8_total_counts.json"
# blockwise: "new_AAA_last/Qwen_72B_0.5B_blockwise_gamma_10_topp_0.8_total_counts.json"
# naive: "new_AAA_last/Qwen_72B_0.5B_naive_gamma_10_topp_0.8_total_counts.json"


# top_p=1.0, temp=0.2:
# backward_clever: "AAA_last/Qwen_72B_0.5B_backward_clever_gamma_10_t0.2_topp_1.0_total_counts.json"
# blockwise: "new_AAA_last/Qwen_72B_0.5B_blockwise_gamma_10_t0.2_topp_1.0_total_counts.json"
# naive: "new_AAA_last/Qwen_72B_0.5B_naive_gamma_10_t0.2_topp_1.0_total_counts.json"

# top_p=1.0, temp=1.0:
# backward_clever: "AAA_last/Qwen_72B_0.5B_backward_clever_gamma_10_topp_1_total_counts.json"
# block: "AAA_last/Qwen_72B_0.5B_blockwise_gamma_10_topp_1.0_total_counts.json"
# naive: "AAA_last/Qwen_72B_0.5B_naive_gamma_10_topp_1.0_total_counts.json"

# top_p=1.0, temp=0.6:
# backward_clever: "AAA_last/Qwen_72B_0.5B_backward_clever_gamma_10_topp_1_total_counts.json"
# block: "AAA_last/Qwen_72B_0.5B_blockwise_gamma_10_topp_1.0_total_counts.json"
# naive: "AAA_last/Qwen_72B_0.5B_naive_gamma_10_topp_1.0_total_counts.json"

with open(f"AAA_final_t_6/Qwen_72B_0.5B_backward_clever_gamma_10_t0.6_topp_1.0_total_counts.json", "rb") as f:
# with open(f"{models}naive_gamma_{gamma}_total_counts.json", "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    counts_1 = orjson.loads(mm[:])   # mm[:] gives you a bytes object
    mm.close()

with open(f"AAA_final_t_6/Qwen_72B_0.5B_blockwise_gamma_10_t0.6_topp_1.0_total_counts.json", "rb") as f:
# with open(f"{models}naive_gamma_{gamma}_total_counts.json", "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    counts_2 = orjson.loads(mm[:])   # mm[:] gives you a bytes object
    mm.close()

with open(f"AAA_final_t_6/Qwen_72B_0.5B_naive_gamma_10_t0.6_topp_1.0_total_counts.json", "rb") as f:
# with open(f"{models}naive_gamma_{gamma}_total_counts.json", "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    counts_3 = orjson.loads(mm[:])   # mm[:] gives you a bytes object
    mm.close()


end = time.time()

print(end-start)


counts = [
    counts_1,
    # counts_backward, counts_backward_recursive,
    counts_2,
          counts_3,
          # counts_backward_clever_approxi,
          ]


draft_eval = []

target_eval = []

total_step = []

sample_length = []

times = []

lens = []



for count in counts:
    draft = 0
    target = 0
    step = 0
    sample = 0
    time_ = 0
    len_ = 0
    for n in range(len(count["draft_eval"])):
        # exclude the draft lengths<10 cases for a fair comparison
        # count["draft_eval"][n][count["draft_eval"][n]==10]
        draft_list = np.array(count["draft_eval"][n])
        target_list = np.array(count["target_eval"][n])
        step_list = np.array(count["total_step"][n])
        sample_list = np.array(count["sample_length"][n])
        print("check length")
        print(len(sample_list))
        print(len(step_list))
        print(count["time"][n])

        draft += draft_list[draft_list==gamma].sum()
        target += target_list[draft_list==gamma].sum()
        step += step_list[draft_list==gamma].sum()
        sample += sample_list[draft_list==gamma].sum()
        time_ += float(count["time"][n])
        len_ += len(sample_list)

    lens.append(len_)
    draft_eval.append(draft/len_)
    target_eval.append(target/len_)
    total_step.append(step/len_)
    sample_length.append(sample/len_)
    times.append(time_/len_)


    # times.append(sample/time_)



print("total decoding steps")
print(times)
# print(lens)
draft_eval = np.array(draft_eval)
target_eval = np.array(target_eval)
total_step = np.array(total_step)
sample_length = np.array(sample_length)
times = np.array(times)
speed = []
for l in range(len(times)):
    speed.append(sample_length[l].sum() / times[l].sum())

speed = np.array(speed)

# top_p=0.8, temp=1.0:
# backward_clever: "AAA_last/Qwen_72B_0.5B_backward_clever_gamma_10_topp_0.8_total_counts.json"
# blockwise: "new_AAA_last/Qwen_72B_0.5B_blockwise_gamma_10_topp_0.8_total_counts.json"
# naive: "new_AAA_last/Qwen_72B_0.5B_naive_gamma_10_topp_0.8_total_counts.json"

x = np.arange(len(counts))  # [0, 1, 2, 3]

width = 0.2  # Width of the bars

# Create plot
fig, ax = plt.subplots()
bar1 = ax.bar(x - width-0.05, [a.sum() for a in speed], width, label='speed')  # should be 6.687 for t1_topp_1
# bar1 = ax.bar(x - width-0.05, [a.sum() for a in times], width, label='time')  # should be 6.687 for t1_topp_1
bar2 = ax.bar(x, [a.sum() for a in target_eval], width, label='target_eval')
bar3 = ax.bar(x + width+0.05, [a.sum() for a in sample_length], width, label='sample_length')

# Add labels, title, and legend
ax.set_ylabel('Scores')
ax.set_title('top_p=1.0, temp=0.6')
ax.set_xticks([0,1,2],['HSD', 'blockwise', 'Naive'])
# ax.set_xticklabels(labels)
ax.legend()

# Optional: Add bar labels
ax.bar_label(bar1, padding=2)
ax.bar_label(bar2, padding=2)
ax.bar_label(bar3, padding=2)

# top_p=1.0, temp=0.6:
plt.tight_layout()
plt.savefig(f"AAA_final_t_6/0730_2027_compare_efficiency_topp_1_t_06_final.png")
