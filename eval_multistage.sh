#!/bin/bash
set -eu

checkpoints=(
  "/node/hzl/expriment2025/exp_twospk_noise/dlgtsewhamcpt/exp_wham/ep3_D_LGTSE_stage3"
)

gpuid=2
num_workers=1

data_root=/node/hzl/expriment2025/exp_twospk_noise/dlgtsewhamcpt/whamscp/tt
mix_scp=$data_root/mix.scp
aux_scp=$data_root/aux.scp
ref_scp=$data_root/ref.scp
cal_sdr=1

for checkpoint in "${checkpoints[@]}"; do
  echo "Evaluating checkpoint: $checkpoint"

  python nnet/evaluate_2stage_gpu.py \
    --checkpoint "$checkpoint" \
    --gpuid "$gpuid" \
    --mix_scp "$mix_scp" \
    --aux_scp "$aux_scp" \
    --ref_scp "$ref_scp" \
    --cal_sdr "$cal_sdr" \
    --num_workers "$num_workers" \
  > "$checkpoint/eval.log" 2>&1

  echo "Evaluation for $checkpoint done!"
done

echo "All evaluations completed!"
