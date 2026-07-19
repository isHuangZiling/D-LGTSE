# D-LGTSE

**Paper:** [Lightweight speech enhancement guided target speech extraction in noisy scenarios](https://www.sciencedirect.com/science/article/pii/S0885230826000896)


## Requirements

```bash
pip install -r requirements.txt

```

## Dataset

SCP files for WHAM! dataset are provided in `dataset_scp/whamscp/`:

- `tr/` - training set
- `cv/` - validation set  
- `tt/` - test set

## Data Preparation for Distortion-Aware Training
Before Stage 2 training, generate denoised training data using the pre-trained GTCRN from Stage 1:

```bash
bash separate_se.sh
```
After the denoised speech is generated, create the corresponding `D_*.scp` files and use them for Stage 2 training.

## Training

Multi-stage training:

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

## Experimental Results

Results on the WHAM! dataset (8 kHz).

| ID | Method | SI-SDR ↑ | PESQ ↑ | STOI ↑ |
|:--:|:-------|---------:|--------:|--------:|
| W0 | Unprocessed | -4.49 | 1.43 | 62.77 |
| W1 | SEF-PNet | 8.28 | 2.26 | 86.09 |
| **W2** | **D-LGTSE (Offline)** | **8.78** | **2.29** | **86.68** |
| W3 | CIE-mDPTNet | 10.44 | 2.50 | 89.85 |
| **W4** | **D-LGTSE-mDPTNet (Offline)** | **11.74*** | **2.62*** | **91.53*** |

\* *p* < 0.001 (paired *t*-test vs. CIE-mDPTNet).

## Model Weights Download

The model weights are available on Hugging Face:

🔗 https://huggingface.co/zilinghuang/D-LGTSE

## Contact

For any questions, please contact:

📧 hzlkycg111@163.com

