#!/usr/bin/env bash
data=MKG-Y
dim=128
hidden_dim=1024
num_head=8
num_layer_enc_ent=1
num_layer_enc_rel=1
num_layer_dec=2
max_img_num=3
max_vis_token=8
max_txt_token=25
score_function=tucker
cuda_device=0

# Usage: bash test_mkgy.sh /path/to/checkpoint.ckpt [limit]
if [ -z "$1" ]; then
  echo "Usage: bash test_mkgy.sh /path/to/checkpoint.ckpt [limit]"
  exit 1
fi

ckpt="$1"
limit_arg=""
if [ ! -z "$2" ]; then
  limit_arg="--limit $2"
fi

CUDA_VISIBLE_DEVICES=${cuda_device} python test.py \
  --data ${data} --dim ${dim} --hidden_dim ${hidden_dim} --num_head ${num_head} \
  --num_layer_enc_ent ${num_layer_enc_ent} --num_layer_enc_rel ${num_layer_enc_rel} --num_layer_dec ${num_layer_dec} \
  --max_img_num ${max_img_num} --max_vis_token ${max_vis_token} --max_txt_token ${max_txt_token} \
  --score_function ${score_function} \
  --checkpoint "$1" ${2:+--limit $2}