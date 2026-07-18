#!/usr/bin/env bash 
set -eu  
epochs=120
# constrainted by GPU number & memory
batch_size=64
gpuid=0
num_workers=24
cpt_dir=/node/hzl/expriment2025/exp_twospk_noise/exp_wham/ep0_sefpnet
# resume=/node/hzl/expriment2025/exp_twospk_noise/exp_wham/ep0_sefpnet/last.pt.tar
#[ $# -ne 1 ] && echo "Script error: $0 <gpuid> <cpt-id>" && exit 1
# **确保 cpt_dir 存在**
mkdir -p $cpt_dir

python ./nnet/train.py \
  --gpu $gpuid \
  --epochs $epochs \
  --batch-size $batch_size \
  --num-workers $num_workers \
  --checkpoint $cpt_dir \
> $cpt_dir/train_resume.log 2>&1