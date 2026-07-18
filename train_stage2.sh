#!/usr/bin/env bash 
set -eu  
epochs=120
batch_size=64
gpuid=2
num_workers=24
cpt_dir=/node/hzl/expriment2025/exp_twospk_noise/exp_wham/ep2_D-LGTSE_stage2
pre_pse_cpt=none # do not write path
pre_se_cpt=/node/hzl/expriment2025/exp_twospk_noise/exp_wham/ep1_gtcrn_se/best.pt.tar
#[ $# -ne 1 ] && echo "Script error: $0 <gpuid> <cpt-id>" && exit 1

mkdir -p $cpt_dir

python ./nnet/train_dlgtse_stage2.py \
  --gpu $gpuid \
  --epochs $epochs \
  --batch-size $batch_size \
  --num-workers $num_workers \
  --checkpoint $cpt_dir \
  --pretrained_pse_cpt $pre_pse_cpt \
  --pretrained_se_cpt $pre_se_cpt \
> $cpt_dir/train.log 2>&1