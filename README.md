# Designed to Spread: A Generative Approach to Enhance Information Diffusion

<p align="center">
  <a href="https://ojs.aaai.org/index.php/AAAI/article/view/37063"><img src="https://img.shields.io/badge/AAAI%202026-Paper-blue?style=flat-square" alt="Paper"></a>
  <a href="https://doi.org/10.1609/aaai.v40i2.37063"><img src="https://img.shields.io/badge/DOI-10.1609%2Faaai.v40i2.37063-green?style=flat-square" alt="DOI"></a>
  <a href="https://github.com/idvxlab/designed-to-spread"><img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License"></a>
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.4-orange?style=flat-square" alt="PyTorch">
</p>

<p align="center">
  <b>Ziqing Qian*, Jiaying Lei*, Shengqi Dang, Nan Cao</b><br>
  Tongji University
</p>

---

## Overview

Social media has fundamentally transformed how people access information and form social connections, with **content expression** playing a critical role in driving information diffusion. While prior research has focused largely on network structures and tipping point identification, it provides limited tools for automatically generating content tailored for virality within a specific audience.

To fill this gap, we propose the novel task of **Diffusion-Oriented Content Generation (DOCG)** and introduce an information enhancement algorithm for generating content optimized for diffusion. Our method includes:

- 📊 **Influence Indicator** — enables content-level diffusion assessment *without* requiring access to network topology
- ✏️ **Information Editor** — employs reinforcement learning to explore interpretable editing strategies, leveraging generative models to produce semantically faithful, audience-aware textual or visual content

Experiments on real-world social media datasets and a user study demonstrate that our approach significantly improves diffusion effectiveness while preserving the core semantics of the original content.

---

## Paper

> **Designed to Spread: A Generative Approach to Enhance Information Diffusion**  
> Ziqing Qian, Jiaying Lei, Shengqi Dang, Nan Cao  
> *Proceedings of the AAAI Conference on Artificial Intelligence*, 2026, 40(2): 944–952  
> 📄 [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/37063) | 🔗 [DOI](https://doi.org/10.1609/aaai.v40i2.37063)

---

## Repository Structure

```
designed-to-spread/
├── Data/                    # Raw inputs & pipeline outputs (posts/accounts JSONL, features, models, …)
├── Data_preprocess/         # End-to-end preprocessing (PySpark + Python; entry: main.py)
├── Long_CLIP/               # Long-CLIP model code (used by image / CLIP features)
├── baselines/               # llm_baseline.py, in_context_learning.py
├── influence_indicator/     # train_pijc_model.py, Evaluation.ipynb
├── information_editor/      # text_editor.py, visual_editor.py, base_editor.py (imported by scripts/)
├── scripts/                 # Runnable entry points: text/visual editors, tweet→image preprocess
└── requirements.txt         # Python dependencies
```

**Note:** There is **no** `preprocess.ipynb` or `run_pipeline.sh` in this repo. Preprocessing is driven by **`Data_preprocess/main.py`** (and its modules). The **only** notebook is **`influence_indicator/Evaluation.ipynb`** (evaluation / analysis). Text and image editing are run via **`scripts/run_text_editor.py`** and **`scripts/run_visual_editor.py`**.

---

## Getting Started

### Step 1 — Prerequisites

Make sure you have the following installed:

- Python >= 3.8
- CUDA 12.1 (required for GPU acceleration; the dependencies are built against `cu121`)
- [Conda](https://docs.conda.io/en/latest/) (recommended for environment management)

### Step 2 — Clone the Repository

```bash
git clone https://github.com/idvxlab/designed-to-spread.git
cd designed-to-spread
```

### Step 3 — Create a Virtual Environment

It is strongly recommended to use a dedicated virtual environment to avoid dependency conflicts:

```bash
# Using conda (recommended)
conda create -n docg python=3.10
conda activate docg

# Or using venv
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### Step 4 — Install Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies include:

| Package | Version | Purpose |
|---|---|---|
| `torch` | 2.4.1 | Core deep learning framework |
| `transformers` | 4.46.3 | Pre-trained language models |
| `diffusers` | 0.36.0 | Diffusion-based image generation |
| `openai` | 2.2.0 | LLM API calls |
| `scikit-learn` | 1.3.2 | Machine learning utilities |
| `pyspark` | 3.5.8 | Large-scale data processing |

> **Note**: If you do not have a CUDA 12.1 GPU, you may need to install a CPU-only version of PyTorch separately. See the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

### Step 5 — Prepare the Data & Run Preprocessing

1. Place (or symlink) your social data under `Data/` as expected by **`Data_preprocess/extract_tables.py`**:
   - **`posts.jsonl`** (required), optionally **`posts_all.jsonl`**
   - **`accounts.jsonl`** (required for the user table)

   The released sample under `Data/raw_data/` only contains ID lists and small `*_example.jsonl` snippets; for the full pipeline you need full **`posts.jsonl` / `accounts.jsonl`** (or pass explicit paths).

2. Run the **eight-step** preprocessing pipeline from the repo root (uses **PySpark** in step 1):

```bash
cd Data_preprocess
python main.py
# Optional: custom Data dir or JSONL paths
# python main.py --base_dir /path/to/Data --posts_path ... --accounts_path ...
```

`main.py` runs, in order: **`extract_tables`** → **`extract_network`** → **`split_dataset`** → **`generate_samples`** → **`filter_texts`** → **`compute_message_features`** → **`compute_user_features`** → **`compute_features_and_labels`**.  
You can also import and run any single module’s `main()` for debugging. Intermediate artifacts are written under `Data/` (e.g. features, splits, `outputs/` for `.pt` training packs).

### Step 6 — Train the Influence Indicator (PIJC)

Train the cascade / diffusion predictor on the preprocessed `.pt` samples:

```bash
cd influence_indicator
python train_pijc_model.py
```

By default this reads **`Data/outputs/outputs_train`** and **`Data/outputs/outputs_test`**, and saves checkpoints under **`Data/outputs/model/`** (e.g. `model_epoch_100.pth` referenced elsewhere in the repo).

**Optional — Jupyter evaluation:** open and run **`influence_indicator/Evaluation.ipynb`**.  
(If cells fail to import Long-CLIP weights, note the repo directory is **`Long_CLIP/`** (underscore); align `sys.path` and `model_path` in the notebook with that folder and your weight file, e.g. `longclip-B.pt`.)

### Step 7 — Run the Information Editor (RL)

Core logic lives in **`information_editor/`** (`text_editor.py`, `visual_editor.py`). **Entry points** are under **`scripts/`** from the repository root:

**Text editing**

```bash
python scripts/run_text_editor.py
# See defaults and overrides:
python scripts/run_text_editor.py --help
```

**Image / visual editing** (expects CSVs with `tweet_id`, `tweet_text`, `image_prompt`, plus images under `--orig_image_folder`):

```bash
python scripts/run_visual_editor.py
python scripts/run_visual_editor.py --help
```

Defaults assume a trained PIJC model at **`Data/outputs/model/model_epoch_100.pth`**, user features at **`Data/features/user_features_dict.pt`**, and tweet splits under **`Data/tweet2image/`** (see script `--help`).

**Tweet → image preprocessing & train/test split** (uses **`LLM_API_KEY`** / **`LLM_BASE_URL`** from a `.env` file; generates prompts, images, and `train_tweets.csv` / `test_tweets.csv`):

```bash
python scripts/tweet2image_preprocess_and_split.py
# Edit the paths in the `if __name__ == "__main__"` block if your CSV/output locations differ.
```

### Step 8 — Run Baselines (Optional)

Two Python scripts (no notebooks):

```bash
# LLM-style baseline
python baselines/llm_baseline.py --help

# In-context learning baseline
python baselines/in_context_learning.py --help
```

Use `--text` and/or `--image` to select modalities; see argparse defaults for CSVs, model paths, and output dirs.

### Step 9 — Pipeline Summary

There is **no** single **`scripts/run_pipeline.sh`**. A typical order is:

1. **`Data_preprocess/main.py`** → features & `Data/outputs/` tensors  
2. **`influence_indicator/train_pijc_model.py`** → **`Data/outputs/model/*.pth`**  
3. (Optional) **`scripts/tweet2image_preprocess_and_split.py`** → **`Data/tweet2image/`** for the image track  
4. **`scripts/run_text_editor.py`** / **`scripts/run_visual_editor.py`** (and/or **`baselines/*.py`**)

Configure paths via each script’s CLI flags or the hard-coded defaults in **`tweet2image_preprocess_and_split.py`**’s `main` block where applicable.

---

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{qian2026designed,
  title     = {Designed to Spread: A Generative Approach to Enhance Information Diffusion},
  author    = {Qian, Ziqing and Lei, Jiaying and Dang, Shengqi and Cao, Nan},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {40},
  number    = {2},
  pages     = {944--952},
  year      = {2026},
  doi       = {10.1609/aaai.v40i2.37063},
  url       = {https://ojs.aaai.org/index.php/AAAI/article/view/37063}
}
```

---

## Contact

If you have any questions, feel free to open an issue or contact us at [2411920@tongji.edu.cn](mailto:2411920@tongji.edu.cn).

---

## Acknowledgments

This project is from the [Intelligent Big Data Visualization Lab (iDVX Lab)](https://idvxlab.com/) at Tongji University. The projects released here are associated with publications from the lab.
