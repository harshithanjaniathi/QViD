#!/bin/bash
# QViD on LLaVA-1.5-7B — HallusionBench (image)
#
#   CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/hallusionbench.sh [NUM_TOKENS]
#
# NUM_TOKENS: visual tokens kept from LLM layer 3 on (default 64; 576 = vanilla LLaVA-1.5).
# MODEL_PATH: Hugging Face id or local path (default ./liuhaotian/llava-v1.5-7b if present).
set -euo pipefail

NUM_TOKENS=${1:-64}
MODEL_PATH=${MODEL_PATH:-$([ -d ./liuhaotian/llava-v1.5-7b ] && echo ./liuhaotian/llava-v1.5-7b || echo liuhaotian/llava-v1.5-7b)}

python -m lmms_eval \
    --model qvid_llava \
    --model_args pretrained=${MODEL_PATH},num_tokens=${NUM_TOKENS},scoring_layer=3,num_components=16 \
    --tasks hallusion_bench_image \
    --batch_size 1 \
    --log_samples \
    --output_path ./results/llava-v1.5-7b-qvid${NUM_TOKENS}/hallusionbench
