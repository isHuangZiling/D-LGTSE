# D-LGTSE

Lightweight speech enhancement guided target speech extraction in noisy scenarios.


## Requirements

```bash
pip install -r requirements.txt

```

## Dataset

SCP files for WHAM! dataset are provided in dataset_scp/:

dataset_scp/whamscp/
├── tr/          # training set
├── cv/          # validation set
└── tt/          # test set

## Data Preparation for Distortion-Aware Training
Before Stage 2 training, generate denoised training data using the pre-trained GTCRN from Stage 1:

```bash
bash separate_se.sh
```

This produces the denoised SCP files (prefixed with D_) used for distortion-aware training.

## Training

multi-stage training:

```bash
# Stage 1: GTCRN pre-training
bash train_baseline.sh

# Stage 2: TSE backbone training (configure stage1 checkpoint path in script)
bash train_stage2.sh

# Stage 3: Joint fine-tuning (resume from stage2 checkpoint)
bash train_stage3.sh
```
GTCRN code adapted from: [Xiaobin-Rong/gtcrn](https://github.com/Xiaobin-Rong/gtcrn)

## Evaluation

```bash
# Baseline models (SEF-PNet, CIE-mDPTNet)
bash eval_baseline.sh

# Multi-stage models (D-LGTSE, D-LGTSE-mDPTNet)
bash eval_multistage.sh
```

## Model Weights Download

The model weights are available on Hugging Face:

🔗 https://huggingface.co/zilinghuang/D-LGTSE
