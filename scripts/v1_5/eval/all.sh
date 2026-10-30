#!/bin/bash
# QViD on LLaVA-1.5-7B — all 11 benchmarks in one lmms-eval run.
#
#   CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/all.sh [NUM_TOKENS]
set -euo pipefail

NUM_TOKENS=${1:-64}
MODEL_PATH=${MODEL_PATH:-$([ -d ./liuhaotian/llava-v1.5-7b ] && echo ./liuhaotian/llava-v1.5-7b || echo liuhaotian/llava-v1.5-7b)}
TASKS=gqa,scienceqa_img,textvqa_val,pope,mme,mmbench_en_dev,mmstar,ai2d,chartqa,ocrbench,hallusion_bench_image

python -m lmms_eval \
    --model qvid_llava \
    --model_args pretrained=${MODEL_PATH},num_tokens=${NUM_TOKENS},scoring_layer=3,num_components=16 \
    --tasks ${TASKS} \
    --batch_size 1 \
    --log_samples \
    --output_path ./results/llava-v1.5-7b-qvid${NUM_TOKENS}/all
