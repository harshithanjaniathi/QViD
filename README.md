<div align=center>

<h1>QViD: Vision Token Pruning via Query–Vision Interaction Decomposition</h1>

<!-- TODO: authors, affiliations, arXiv / project links -->

</div>

<!-- TODO: <p align="center"><img src="images/teaser.png" width="900"></p> -->

> **TL;DR:** QViD is a training-free, plug-and-play visual token pruning method for large vision-language models. At an early LLM layer, it scores each visual token by its leverage in the top singular subspace of the text–visual query–key affinity, computed per attention head, and keeps only the most query-relevant tokens for the remaining layers.

## 💥 Core Code

- [`qvid/scoring.py`](qvid/scoring.py) — QViD scoring (~20 lines, depends only on PyTorch).
- [`qvid/llava.py`](qvid/llava.py) — in-LLM pruning for LLaVA-1.5 via forward hooks; no changes to the LLaVA source.

## 🔍 Method

At the input of LLM layer `L` (default `L = 3`), let `Q` be the query projections of the text tokens after the image (the user instruction) and `K` the key projections of the visual tokens, both from layer `L`. For every attention head `h`:

1. **Affinity:** `M_h = Q_h K_hᵀ` with shape `N_q × N_v`, where `N_q ≪ N_v`.
2. **Right singular vectors via QR:** `M_hᵀ = Q R` (reduced QR), then `Rᵀ = U S Vᵀ` (SVD of a small `N_q × N_q` matrix). The right singular vectors of `M_h` are exactly `Q V`, obtained without decomposing the full `N_q × N_v` matrix.
3. **Leverage score:** the score of visual token `j` is the squared norm of row `j` of the top-`r` right singular vectors (`r = 16`).

Scores are averaged over heads and the top-`k` visual tokens are kept. Layers `0 … L−1` see all visual tokens; layers `L …` see only the kept ones. Kept tokens keep their original position ids, so RoPE is unchanged.

## 🛠 Installation

```bash
git clone https://github.com/harshithanjaniathi/QViD.git
cd QViD

conda create -n qvid python=3.11 -y
conda activate qvid

# 1. PyTorch (choose the wheel for your CUDA version)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# 2. LLaVA, without its pinned dependencies (it pins torch==2.1.2)
pip install --no-deps git+https://github.com/haotian-liu/LLaVA.git

# 3. lmms-eval v0.7.1 from source (for benchmarking; skip if you only need the demo).
#    The PyPI wheel of 0.7.1 is missing task template files, so install the tag.
git clone --branch v0.7.1 --depth 1 https://github.com/EvolvingLMMs-Lab/lmms-eval.git
pip install -e lmms-eval

# 4. QViD. This pins transformers==4.37.2, which LLaVA-1.5 needs; pip will warn
#    that lmms-eval asks for a newer transformers. The warning is expected and harmless.
pip install -e .
```

The LLaVA-1.5-7B weights ([liuhaotian/llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b)) are downloaded automatically on first use. To use a local copy, pass its path instead of the model id.

## 🚀 Usage

### Demo

```bash
python scripts/demo.py --image path/to/image.jpg \
    --question "What is written on the sign?" --num-tokens 64
```

### In your own code

```python
from llava.model.builder import load_pretrained_model
from qvid import apply_qvid

tokenizer, model, image_processor, _ = load_pretrained_model(
    "liuhaotian/llava-v1.5-7b", None, "llava-v1.5-7b", attn_implementation="sdpa")
apply_qvid(model, num_tokens=64)   # keep 64 of 576 visual tokens from layer 3 on

# Use model.generate(...) exactly as in LLaVA; pruning happens inside.
```

| Argument | Default | Description |
|---|---|---|
| `num_tokens` | 64 | Visual tokens kept from `scoring_layer` onward (576 = no pruning). |
| `scoring_layer` | 3 | LLM layer at whose input tokens are scored and pruned. |
| `num_components` | 16 | Rank `r` of the singular subspace. |

### Evaluation

Evaluation uses [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). Installing QViD registers the model `qvid_llava`.

```bash
# All 11 benchmarks at 64 tokens
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_llava15.sh 64

# Selected benchmarks
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_llava15.sh 128 pope,gqa,textvqa_val

# Vanilla LLaVA-1.5 (no pruning)
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_llava15.sh 576
```

Or call lmms-eval directly:

```bash
python -m lmms_eval --model qvid_llava \
    --model_args pretrained=liuhaotian/llava-v1.5-7b,num_tokens=64 \
    --tasks pope --batch_size 1 --output_path results/
```

## 📊 Results

<!-- TODO: main results table (LLaVA-1.5-7B, 11 benchmarks, token budgets) -->

## ⚠️ Notes

- Supported: LLaVA-1.5-7B (13B should work, since head counts are read from the config, but has not been verified), batch size 1, `attn_implementation="sdpa"`, `transformers==4.37.2`. `apply_qvid` raises an error for other settings.
## 📌 Citation

```bibtex
<!-- TODO -->
```

## 🙏 Acknowledgements

Built on [LLaVA](https://github.com/haotian-liu/LLaVA) and evaluated with [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).
