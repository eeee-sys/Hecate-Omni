# Hecate-Omni: State-Aligned Hierarchical Credit Assignment for Multimodal Social Reasoning
<p align="center">
  <a href="https://huggingface.co/datasets/Harry-1234/Hecate-Omni" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue">
</p>
## 👀 Hecate-Omni Overview
<p align="center">
    <img src="./assets/method.png" width="100%" height="100%">
</p>

<p align="center">
    <img src="./assets/visualization.png" width="100%" height="100%">
</p>

#### 🌟 Contributions in Hecate-Omni

1. We develop and release **Hecate-Omni**, a model for multimodal social intelligence reasoning, designed to handle diverse social behavior tasks spanning emotion recognition, mental-state analysis, sentiment understanding, and pathological behavior recognition. Hecate-Omni achieves SOTA on the evaluated benchmarks, demonstrating effective and robust reasoning across different social behavior scenarios.

2. We introduce **Hecate-PO**. Hecate-PO adaptively identifies latent reasoning units using predictive entropy and establishes correspondences among equivalent reasoning states through internal state representations, enabling reasoning decisions to be evaluated under semantically comparable contexts.

## 📈 Experimental Results

#### 📍 Results
<p align="center">
    <img src="./assets/model.png" width="100%" height="100%">
</p>

<p align="center">
    <img src="./assets/comparison.png" width="100%" height="100%">
</p>

## ⭐ Training detail and evaluation

### 🔮 Evaluation
```
MODEL_PATH=/path/to/base_model \
DATA_PATH=/path/to/human_behavior_atlas \
ADAPTER=/path/to/checkpoint \
EVAL_SPLIT=test EVAL_PER_DATASET=0 RUN_NAME=hecatepo_test \
bash scripts/run_hecatepo.sh evaluate
```

#### 📖 Prepare
Download the base model [OmniSapiens2.0](https://huggingface.co/HumanBehaviorAtlas/OmniSapiens2.0) and [Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) from [huggingface](https://huggingface.co/)

Prepare Hecate-Omni LoRA weights from [huggingface](https://huggingface.co/datasets/Harry-1234/Hecate-Omni)

Prepare human-behavior-atlas data from [huggingface]([https://huggingface.co/datasets/Harry-1234/Hecate-Omni](https://huggingface.co/datasets/HumanBehaviorAtlas/human_behavior_atlas)
### 🔥 Training
```
MODEL_PATH=/path/to/base_model \
DATA_PATH=/path/to/human_behavior_atlas \
RUN_NAME=hecatepo_train \
bash scripts/run_hecatepo.sh train
```
### Examples

https://github.com/user-attachments/assets/000a936d-2834-4798-88f5-3d7d52c58610
```
Question:
How would you describe the emotional state of the speaker in the recording? Choose the most appropriate emotion from:
anger,
disgust,
fear,
happy,
surprise,
neutral,
sad

First reason briefly from the provided evidence. Enclose your reasoning in <think>...</think>. Then provide exactly one final answer in \boxed{...}. For classification, use exactly one of the requested labels. For open questions, give a concise, direct answer.

Hecate-Omni:
<think> The speaker sounds sad because they are expressing that everything is ruined, which is a sad situation. </think> sad \boxed{sad}
```
## 📜 License
Our project is under [MIT License](https://github.com/eeee-sys/Hecate-Omni/blob/main/LICENSE)
