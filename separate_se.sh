#!/bin/bash 
set -eu
checkpoint=/node/hzl/expriment2025/D-LGTSE/checkpoint_wham!/cpt_gtcrn
gpuid=1
data_root=/node/hzl/expriment2025/exp_twospk_noise/whamscp/tr

mix_scp=$data_root/mix.scp
ref_scp=$data_root/mix_clean.scp

fs=8000
dump_dir=/node/hzl/expriment2025/D-LGTSE/checkpoint_wham!/cpt_gtcrn/train_results
mkdir -p $checkpoint
./nnet/separate_se.py \
  --checkpoint $checkpoint \
  --gpuid $gpuid \
  --mix_scp $mix_scp \
  --ref_scp $ref_scp \
  --fs $fs \
  --dump-dir $dump_dir \
 > $checkpoint/separate.log 2>&1

echo "Separate done!"