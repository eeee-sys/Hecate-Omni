# Hecate-Omni: State-Aligned Hierarchical Credit Assignment for Multimodal Social Reasoning
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

Prepare human-behavior-atlas data from [huggingface]([https://huggingface.co/datasets/Harry-1234/Hecate-Omni](https://huggingface.co/datasets/HumanBehaviorAtlas/human_behavior_atlas))
### 🕹️ Training
```
MODEL_PATH=/path/to/base_model \
DATA_PATH=/path/to/human_behavior_atlas \
RUN_NAME=hecatepo_train \
bash scripts/run_hecatepo.sh train
```

#### 🔥 Training
