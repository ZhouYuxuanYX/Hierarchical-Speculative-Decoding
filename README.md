# Hierarchical Speculative Decoding (HSD)

**Hierarchical Speculative Decoding (HSD)** is a state-of-the-art lossless verification algorithm for accelerating large language model (LLM) inference. HSD verifies draft outputs using the joint probability of multiple draft tokens and reconstructs the target model’s joint distribution through a hierarchy of resampling distributions across different branches. This hierarchical verification enables HSD to **accept more tokens in expectation** than both token-wise and block-wise verification methods, while **strictly preserving the target model’s output distribution**—i.e., without any degradation in model performance.

Moreover, its strong explainability and generality make it readily integrable into a wide range of speculative decoding frameworks. Notably, HSD serves as a **drop-in replacement** for token-wise verification. For example, by directly replacing the token-wise verification in a pre-fine-tuned EAGLE-3 model, HSD achieves over a 12% performance improvement **without requiring any additional fine-tuning**. 

## Table of Contents

- [Hierarchical Speculative Decoding (HSD)](#hierarchical-speculative-decoding-hsd)
  - [Table of Contents](#table-of-contents)
  - [Experiment Settings](#experiment-settings)
    - [Experiments Setup](#experiments-setup)
    - [Baselines and Metrics](#baselines-and-metrics)
  - [Reproduce Experiment Results on GSM8K](#reproduce-experiment-results-on-gsm8k)
    - [Requirements](#requirements)
    - [Single-Draft Verification](#single-draft-verification)
    - [Multi-Draft Verification with Recursive Reject Sampling](#multi-draft-verification-with-recursive-reject-sampling)
    - [Compute Metrics](#compute-metrics)
    - [GSM8K Performance Results](#gsm8k-performance-results)
      - [Accuracy Comparison (Table 2)](#accuracy-comparison-table-1)
      - [Block Efficiency \& Speed (Table 3)](#block-efficiency--speed-table-2)
  - [EAGLE Integration](#eagle-integration)
    - [EAGLE + HSD Results](#eagle--hsd-results)
  - [Citation](#citation)
  - [References](#references)

## Experiment Settings
### Experiments Setup
Experiments are conducted with the widely adopted GPTQ-quantized 8-bit instruction-tuned Qwen2.5 series [Bai et al., 2023](#). By default, we employ the 0.5B model as the draft model and 72B as the target model, with a temperature of 1. All experiments were conducted on a single NVIDIA H20 GPU with 96 GB of memory, unless otherwise specified.

### Baselines and Metrics
We compare two lossless verification methods—Token-wise [Leviathan et al., 2023](https://arxiv.org/abs/2211.17192) and Block-wise [Sun et al., 2024](https://arxiv.org/abs/2403.10444)—using two metrics: *Block Efficiency* (tokens/step) and *Decoding Speed* (tokens/second).
- **Block Efficiency** measures the average tokens generated per serial call to the target model, reflecting intrinsic efficiency independent of hardware.  
- **Decoding Speed** indicates tokens produced per second for practical reference, though it depends on hardware and implementation.


## Reproduce Experiment Results on GSM8K

### Requirements
Please install `python=3.10` and `transformers=4.46.3`, then copy the provided files under the `transformers` directory to:

```
anaconda3/envs/your_environment_name/lib/python3.10/site-packages/transformers/
```
Then navigate to `chain-of-thought-hub/gsm8k`:
```bash
cd chain-of-thought-hub/gsm8k
```

### Single-Draft Verification

Baseline methods:
- **Tokenwise** verification
  ```bash
  bash eval_speculative_qwen.sh
  ```

- **Blockwise** verification
  ```bash
  bash eval_speculative_qwen_blockwise.sh
  ```

Our methods:
- **NaiveHSD** - basic hierarchical verification
  ```bash
  bash eval_speculative_qwen_backward.sh
  ```

- **HSD (clever)** - optimized with smart capping mechanism
  ```bash
  bash eval_speculative_qwen_backward_clever.sh
  ```

### Multi-Draft Verification with Recursive Reject Sampling

Extends HSD to use multiple independent draft models for even better performance.

- **Tokenwise** multidraft
  ```bash
  bash eval_speculative_qwen_multidraft_11.sh
  ```

- **HSD** multidraft
  ```bash
  bash eval_speculative_qwen_backward_clever_multidraft_11.sh
  ```

### Compute Metrics

Finally, evaluate Block Efficiency and Decoding Speed:
```bash
python compute_speculative_stats.py
```

### GSM8K Performance Results

Using Qwen2.5-0.5B as draft model on GSM8K:

#### **Table 1**: Accuracy Comparison

<table>
  <thead>
    <tr>
      <th>Target Model</th>
      <th>Tokenwise</th>
      <th>HSD (ours)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Qwen2.5-72B</td>
      <td>0.8213</td>
      <td><strong>0.8517</strong></td>
    </tr>
    <tr>
      <td>Qwen2.5-32B</td>
      <td>0.8213</td>
      <td><strong>0.8479</strong></td>
    </tr>
    <tr>
      <td>Qwen2.5-14B</td>
      <td>0.8327</td>
      <td><strong>0.8327</strong></td>
    </tr>
  </tbody>
</table>

HSD maintains or slightly improves accuracy (both Token-wise and HSD are lossless methods, so any deviations are due to randomness) while offering significantly faster inference.
#### **Table 2**: Block Efficiency & Speed 
| Method         | BE 14B | BE 32B | BE 72B | DS 14B | DS 32B | DS 72B |
|----------------|--------|--------|--------|--------|--------|--------|
| Tokenwise      | 5.99   | 6.14   | 6.44   | 82.28  | 53.87  | 31.49  |
| Blockwise      | 6.13 (+2.3%) | 6.26 (+2.0%) | 6.53 (+1.4%) | 86.06 (+4.6%) | 54.91 (+1.9%) | 31.79 (+1.0%) |
| **HSD (Ours)** | **6.30 (+5.2%)** | **6.47 (+5.4%)** | **6.65 (+3.3%)** | **91.05 (+10.7%)** | **57.12 (+6.0%)** | **32.52 (+3.3%)** |


HSD consistently improves both Block Efficiency (BE) and Decoding Speed (DS) relative to Tokenwise and Blockwise verification. For **GSM8K**, the gains are stable across scales, with BE improvements of **5.2%--5.4%** at 14B/32B and **3.3%** at 72B, accompanied by DS increases of up to **10.7%**.

## EAGLE Integration

HSD can be integrated with [EAGLE-3](https://github.com/SforAiD/LLM-Eagle) to further improve performance. EAGLE-3 fine-tunes an additional component to predict draft tokens from the target model's hidden features, eliminating the need for a separate draft model and thereby improving decoding speed. 

To demonstrate HSD’s compatibility with advanced decoding systems, we integrated it into the state-of-the-art **EAGLE-3-LLaMa3.1-Instruct-8B** (with a default draft length of 7) by replacing its token-wise verifier, as shown in **Table 3**. Importantly, HSD operates orthogonally to the drafting phase, so it can be used directly with a pre-fine-tuned EAGLE-3 model **without requiring any additional fine-tuning**.

### **Table 3**: EAGLE-3 + HSD (EAGLE-3H) Results
| Method               | Block Eff. | Decoding Speed |
|----------------------|------------|----------------|
| EAGLE-3              | 3.40       | 71.59          |
| Blockwise            | N/A        | N/A            |
| **EAGLE-3H (Ours)**  | **3.55 (+4.4%)** | **80.49 (+12.4%)** |


For EAGLE integration setup, see the [EAGLE repository](https://github.com/SforAiD/LLM-Eagle) and the `EAGLE-hsd/` directory in this repo.

```bash
cd EAGLE-3H
bash 1114_eagle_eval.sh
```

## Citation

```bibtex
@misc{zhou2026overcomingjointintractabilitylossless,
      title={Overcoming Joint Intractability with Lossless Hierarchical Speculative Decoding}, 
      author={Yuxuan Zhou and Fei Huang and Heng Li and Fengyi Wu and Tianyu Wang and Jianwei Zhang and Junyang Lin and Zhi-Qi Cheng},
      year={2026},
      eprint={2601.05724},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2601.05724}, 
}
```

## References

[1] Sun, Ziteng, et al. "Block verification accelerates speculative decoding." arXiv preprint arXiv:2403.10444 (2024).

[2] For EAGLE integration: See the [EAGLE repository](https://github.com/SforAiD/LLM-Eagle)
