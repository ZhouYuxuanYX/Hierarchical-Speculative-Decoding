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

gamma = 7  # Max draft length in EAGLE JSONL data

start =time.time()

# Load JSONL files (one JSON object per line)
# Each line has: accept_length (list), draft_length (list), generate_time (float)
jsonl_files = [
    '/path/to/your/file/llama38b2_40-temperature-1.0_ea_info.jsonl',
]

# Map each JSON file to a core index
file_to_core = {path: core for core, path in enumerate(jsonl_files)}

print("JSON file -> core mapping:")
for path, core in file_to_core.items():
    print(f"core {core}: {path}")

counts = []

for json_path in jsonl_files:
    with open(json_path, "r") as f:
        lines = [orjson.loads(line) for line in f if line.strip()]
        # Convert to the expected format - flatten the nested structure
        counts_single = {"draft_eval": [], "target_eval": [], "total_step": [], "sample_length": [], "time": []}
        for line_data in lines:
            # draft_length and accept_length are nested: [[val1, val2, ...]]
            # Extract the inner list and store as individual arrays per sample
            draft_lengths = line_data.get("draft_length", [[]])[0]
            accept_lengths = line_data.get("accept_length", [[]])[0]

            # Store as arrays matching the GSM8K format
            counts_single["draft_eval"].append(np.array(draft_lengths))
            counts_single["target_eval"].append(np.array([1] * len(accept_lengths)))
            counts_single["total_step"].append(np.array([1] * len(accept_lengths)))
            counts_single["sample_length"].append(np.array(accept_lengths))
            counts_single["time"].append(line_data.get("generate_time", [0])[0])

    counts.append(counts_single)


end = time.time()

print(end-start)


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
        draft_list = np.array(count["draft_eval"][n])
        target_list = np.array(count["target_eval"][n])
        step_list = np.array(count["total_step"][n])
        sample_list = np.array(count["sample_length"][n])
        
        # Filter by gamma (only include steps where draft_length == gamma)
        draft += draft_list[draft_list == gamma].sum()
        target += target_list[draft_list == gamma].sum()
        step += step_list[draft_list == gamma].sum()
        sample += sample_list.sum()
        time_ += float(count["time"][n])
        len_ += len(sample_list)

    # Calculate per-step averages (same as eval.py)
    draft_eval.append(draft / len_)
    target_eval.append(target / len_)
    total_step.append(step / len_)
    sample_length.append(sample / len_)  # block efficiency
    times.append(time_ / len_)
    
    # Fix speed calculation to match eval.py
    # Speed should be: total_sample_length / total_time
    lens.append(sample / time_)


# print(lens)
draft_eval = np.array(draft_eval)
target_eval = np.array(target_eval)
total_step = np.array(total_step)
sample_length = np.array(sample_length)
times = np.array(times)
# Speed is already calculated as sample/time_ (tokens/s)
speed = np.array(lens)

print("\n=== Summary Statistics ===")
print(f"Speed (tokens/s): {speed}")
print(f"BE (Block Efficiency): {sample_length}")



