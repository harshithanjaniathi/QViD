#!/bin/bash
# Evaluate QViD on LLaVA-1.5-7B with lmms-eval.
#
#   CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval.sh [NUM_TOKENS] [TASKS]
#
#   bash scripts/v1_5/eval.sh 64                      # all 11 benchmarks, 64 visual tokens
#   bash scripts/v1_5/eval.sh 64 textvqa_val,mme      # selected benchmarks
#   bash scripts/v1_5/eval.sh 576                     # vanilla LLaVA-1.5 (no pruning)
#
# NUM_TOKENS: visual tokens kept from LLM layer 3 on (0-based index 2) (default 64; 576 = no pruning).
# MODEL_PATH: Hugging Face id or local path (default ./liuhaotian/llava-v1.5-7b if present).
set -euo pipefail

NUM_TOKENS=${1:-64}
TASKS=${2:-gqa,scienceqa_img,textvqa_val,pope,mme,mmbench_en_dev,mmstar,ai2d,chartqa,ocrbench,hallusion_bench_image}
MODEL_PATH=${MODEL_PATH:-$([ -d ./liuhaotian/llava-v1.5-7b ] && echo ./liuhaotian/llava-v1.5-7b || echo liuhaotian/llava-v1.5-7b)}

python -m lmms_eval \
    --model qvid_llava \
    --model_args pretrained=${MODEL_PATH},num_tokens=${NUM_TOKENS},scoring_layer=2,num_components=16 \
    --tasks ${TASKS} \
    --batch_size 1 \
    --log_samples \
    --output_path ./results/llava-v1.5-7b-qvid${NUM_TOKENS}
