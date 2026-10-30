<div align=center>

<h1>Vision Token Pruning via Query–Vision Interaction Decomposition</h1>

<h4 align="center">

Harshithanjani Athi<sup>1</sup>,
Sravan Kumar Ankireddy<sup>1</sup>,
Jianzhong Charlie Zhang<sup>2</sup>,
Hyeji Kim<sup>1</sup>

<sup>1</sup>The University of Texas at Austin, <sup>2</sup>Samsung Research America

</h4>

[![arXiv](https://img.shields.io/badge/arXiv-2610.33215-b31b1b.svg)](https://arxiv.org/abs/2610.33215)

</div>

<!-- TODO: <p align="center"><img src="images/overview.png" width="1000"></p> -->

> **TL;DR:** QViD is a training-free, plug-and-play visual token pruning method for large vision-language models. At an early LLM layer, it scores each visual token by its leverage in the top singular subspace of the text–visual query–key affinity, computed per attention head, and keeps only the most query-relevant tokens for the remaining layers.

## 💥 Core Codes

The core implementation is in [`llava/model/language_model/QViD.py`](llava/model/language_model/QViD.py).
It provides `QViDLlamaModel`, which replaces `LlamaModel` as LLaVA's language model
([`llava_llama.py`](llava/model/language_model/llava_llama.py)). Everything else in
[`llava/`](llava) is [LLaVA](https://github.com/haotian-liu/LLaVA) at commit `c121f04`, unmodified.

## 🔍 Method

At the input of LLM layer `L` (default `L = 3`), let `Q` be the query projections of the text tokens after the image and `K` the key projections of the visual tokens, both from layer `L`. For every attention head `h`:

1. **Affinity:** `M_h = Q_h K_hᵀ` with shape `N_q × N_v`, where `N_q ≪ N_v`.
2. **Right singular vectors via QR:** `M_hᵀ = Q R` (reduced QR), then `Rᵀ = U S Vᵀ` (SVD of a small `N_q × N_q` matrix). The right singular vectors of `M_h` are exactly `Q V`, obtained without decomposing the full `N_q × N_v` matrix.
3. **Leverage score:** the score of visual token `j` is the squared norm of row `j` of the top-`r` right singular vectors (`r = 16`).

Scores are averaged over heads and the top-`k` visual tokens are kept. Layers `0 … L−1` see all 576 visual tokens; layers `L …` see only the kept ones. Kept tokens keep their original position ids, so RoPE is unchanged.

## 🛠 Preparation

1. Clone this repository.

```bash
git clone https://github.com/harshithanjaniathi/QViD.git
cd QViD
```

2. Environment setup.

```bash
conda create -n qvid python=3.11 -y
conda activate qvid

# PyTorch (pick the wheel for your CUDA version)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# lmms-eval v0.7.1, with our evaluation patch
git clone --branch v0.7.1 --depth 1 https://github.com/EvolvingLMMs-Lab/lmms-eval.git
git -C lmms-eval apply ../patches/lmms-eval-v0.7.1.patch
pip install -e lmms-eval

# QViD (this LLaVA fork). Installed last: it pins transformers==4.37.2, which
# LLaVA-1.5 needs. pip warns that lmms-eval asks for a newer one; that is expected.
pip install -e .
```

3. Download [LLaVA-1.5-7B](https://huggingface.co/liuhaotian/llava-v1.5-7b) and put it under `./liuhaotian/llava-v1.5-7b` (the scripts fall back to the Hugging Face id if it is not there). The CLIP vision tower is downloaded automatically.

```bash
huggingface-cli download liuhaotian/llava-v1.5-7b --local-dir ./liuhaotian/llava-v1.5-7b
```

4. Benchmarks are downloaded from the Hugging Face Hub by lmms-eval on first use. Several `lmms-lab` datasets require a Hugging Face login:

```bash
huggingface-cli login
```

## 🚀 Evaluation

[`scripts/v1_5/eval.sh`](scripts/v1_5/eval.sh) runs [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). The first argument is the number of visual tokens kept (default 64; 576 = vanilla LLaVA-1.5), the optional second a comma-separated list of lmms-eval tasks (default: all 11 benchmarks).

```bash
# All 11 benchmarks with 64 visual tokens
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval.sh 64

# Selected benchmarks
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval.sh 64 textvqa_val,mme
```

The 11 benchmarks: `gqa`, `scienceqa_img`, `textvqa_val`, `pope`, `mme`, `mmbench_en_dev`, `mmstar`, `ai2d`, `chartqa`, `ocrbench`, `hallusion_bench_image`. Results are written to `./results/llava-v1.5-7b-qvid<NUM_TOKENS>/`.

## 💻 Usage in your own code

```python
from llava.model.builder import load_pretrained_model

tokenizer, model, image_processor, _ = load_pretrained_model(
    "liuhaotian/llava-v1.5-7b", None, "llava-v1.5-7b")
model.get_model().set_qvid(num_tokens=64, layer=3, num_components=16)

# Use model.generate(...) exactly as in LLaVA; pruning happens inside.
```

| Argument | Default | Description |
|---|---|---|
| `num_tokens` | 64 | Visual tokens kept from `layer` onward (`None` or 576 = no pruning). |
| `layer` | 3 | LLM layer at whose input tokens are scored and pruned. |
| `num_components` | 16 | Rank `r` of the singular subspace. |

A single-image demo:

```bash
python scripts/demo.py --image path/to/image.jpg --question "What is written on the sign?" --num-tokens 64
```

## 📌 Citation

If you find QViD useful, please cite:

```bibtex
@misc{athi2026qvid,
      title={Vision Token Pruning via Query-Vision Interaction Decomposition},
      author={Harshithanjani Athi and Sravan Kumar Ankireddy and Jianzhong Charlie Zhang and Hyeji Kim},
      year={2026},
      eprint={2610.33215},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2610.33215},
}
```
