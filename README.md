<div align=center>

<h1>QViD: Vision Token Pruning via Query–Vision Interaction Decomposition</h1>

<!-- TODO: authors, affiliations -->

<!-- TODO: arXiv / project links -->

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

# lmms-eval v0.7.1 + the QViD evaluation patch (see "Evaluation protocol" below)
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

Each benchmark has a script under [`scripts/v1_5/eval/`](scripts/v1_5/eval). The argument is the number of visual tokens kept (default 64; 576 = vanilla LLaVA-1.5).

Example for evaluating TextVQA results:
```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/textvqa.sh 64
```
Example for evaluating MME results:
```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mme.sh 64
```
All 11 benchmarks in one run:
```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/all.sh 64
```

| Script | lmms-eval task | Script | lmms-eval task |
|---|---|---|---|
| `gqa.sh` | `gqa` | `mmstar.sh` | `mmstar` |
| `sqa.sh` | `scienceqa_img` | `ai2d.sh` | `ai2d` |
| `textvqa.sh` | `textvqa_val` | `chartqa.sh` | `chartqa` |
| `pope.sh` | `pope` | `ocrbench.sh` | `ocrbench` |
| `mme.sh` | `mme` | `hallusionbench.sh` | `hallusion_bench_image` |
| `mmbench.sh` | `mmbench_en_dev` | `all.sh` | all of the above |

Results are written to `./results/llava-v1.5-7b-qvid<NUM_TOKENS>/<benchmark>/`.

### Evaluation protocol

[`patches/lmms-eval-v0.7.1.patch`](patches/lmms-eval-v0.7.1.patch) makes lmms-eval v0.7.1 score QViD the way our reported numbers were obtained:

| Benchmark | Change |
|---|---|
| TextVQA | OCR tokens are included in the prompt (`ocr: true`) for `qvid_llava`. |
| MMBench-dev | Scored locally with circular accuracy; no GPT judge, and unparsable answers count as wrong instead of being filled in at random. |
| HallusionBench | Accuracies are aggregated with the `*_intern` metric functions. |

Without the patch, TextVQA and MMBench scores will differ.

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

## ⚠️ Notes

- Supported: LLaVA-1.5 (7B verified), batch size 1, `transformers==4.37.2`, SDPA attention (the transformers default; `set_qvid()` raises an error if the model was loaded with `use_flash_attn=True` or `attn_implementation="eager"`).
- Images with a variable number of visual tokens (LLaVA-1.6 `anyres`) are passed through unpruned.

## 📊 Results

<!-- TODO: main results table (LLaVA-1.5-7B, 11 benchmarks, token budgets) -->

## 📌 Citation

```bibtex
<!-- TODO -->
```

## 🙏 Acknowledgements

Built on [LLaVA](https://github.com/haotian-liu/LLaVA) and evaluated with [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). The repository layout follows [V<sup>2</sup>Drop](https://github.com/xuyang-liu16/V2Drop).
