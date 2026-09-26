#!/bin/bash
# Evaluate QViD on LLaVA-1.5-7B with lmms-eval.
#
#   bash scripts/eval_llava15.sh [NUM_TOKENS] [TASKS]
#
#   CUDA_VISIBLE_DEVICES=0 bash scripts/eval_llava15.sh 64
#   CUDA_VISIBLE_DEVICES=0 bash scripts/eval_llava15.sh 576 pope,gqa   # vanilla baseline
#
# Set MODEL_PATH to use a local checkpoint instead of the Hugging Face id.
set -euo pipefail

NUM_TOKENS=${1:-64}
TASKS=${2:-pope,gqa,mmbench_en_dev,hallusion_bench_image,chartqa,ai2d,scienceqa_img,textvqa_val,mmstar,ocrbench,mme}
MODEL_PATH=${MODEL_PATH:-liuhaotian/llava-v1.5-7b}
OUT=${OUT:-results/llava15_qvid_n${NUM_TOKENS}}

python -m lmms_eval \
    --model qvid_llava \
    --model_args "pretrained=${MODEL_PATH},num_tokens=${NUM_TOKENS},scoring_layer=3,num_components=16" \
    --tasks "${TASKS}" \
    --batch_size 1 \
    --log_samples \
    --output_path "${OUT}"
