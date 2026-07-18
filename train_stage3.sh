#!/usr/bin/env bash 
set -eu  
epochs=200
# constrainted by GPU number & memory
batch_size=64
gpuid=0
num_workers=12
cpt_dir=/node/hzl/expriment2025/exp_twospk_noise/exp_wham/ep3_D-LGTSE_stage3
resume=/node/hzl/expriment2025/exp_twospk_noise/exp_wham/ep2_D-LGTSE_stage2/best.pt.tar
#[ $# -ne 1 ] && echo "Script error: $0 <gpuid> <cpt-id>" && exit 1

mkdir -p $cpt_dir

python ./nnet/train_dlgtse_stage3.py \
  --gpu $gpuid \
  --epochs $epochs \
  --batch-size $batch_size \
  --num-workers $num_workers \
  --resume $resume \
  --checkpoint $cpt_dir \
> $cpt_dir/train.log 2>&1