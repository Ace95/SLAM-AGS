# SLAM-AGS: Slide-Label Aware Multi-Task Pretraining Using Adaptive Gradient Surgery in Computational Cytology

*Marco Acerbis, Swarnadip Chatterjee, Christophe Avenel, Joakim Lindblad*

[![ArXiv](https://img.shields.io/badge/arXiv-2511.14639-b31b1b.svg)](https://arxiv.org/pdf/2511.14639) [![ISBI 2026](https://img.shields.io/badge/ISBI_2026-Accepted-green)]()

---

## Overview

**SLAM-AGS** is a multi-task self-supervised pretraining framework for computational cytology. It leverages slide-level weak labels alongside adaptive gradient surgery to learn discriminative patch-level representations, which are then used to drive a Multiple Instance Learning (MIL) pipeline for whole-slide classification.

## Repository Structure

```
SLAM-AGS/
├── SLAM-AGS.py        # Main pretraining script
├── PAMIL.py           # Prototype-Aware MIL training and evaluation
├── embeddings.py      # Feature extraction using a pretrained encoder
├── bags_gen.py        # Bag generation from raw dataset
├── model.py           # Model architecture definitions
├── utils.py           # Utility functions
└── prerequisites.txt  # Required dependencies
```

## Dataset

This codebase uses the [Bone Marrow Cytomorphology dataset](https://www.cancerimagingarchive.net/collection/bone-marrow-cytomorphology_mll_helmholtz_fraunhofer/) from The Cancer Imaging Archive (TCIA).

## Setup

### Installation

```bash
pip install -r requirements.txt
```

### Data Preparation

1. Download the Bone Marrow Cytomorphology dataset from the link above.
2. Run `bags_gen.py` to generate train/test bags from the raw data.
3. For the **training set only**, create two folders:
   - `positive/` — copy all patches from **positive** bags
   - `negative/` — copy all patches from **negative** bags

> The `positive/` and `negative/` directories are used by `SLAM-AGS.py`. For `PAMIL.py`, you will use the folder containing the bags directly.

## Usage

### Step 1 — Pretrain the Encoder

```bash
python SLAM-AGS.py \
  --positive_dir /path/to/positive/dir \
  --negative_dir /path/to/negative/dir \
  --wr <witness_rate>
```

`--wr` sets the witness rate and is used only for naming the saved checkpoint file.

### Step 2 — Extract Embeddings

```bash
python embeddings.py \
  --split train   \          # or 'test'
  --data_dir /path/to/bags/dir \
  --dim <encoder_output_dim> \
  --model /path/to/pretrained/model.pth \
  --pre <weakly|self|wcs>
```

### Step 3 — Train & Evaluate the MIL Model

```bash
python PAMIL.py \
  --emb_train /path/to/train/embeddings/dir \
  --emb_test  /path/to/test/embeddings/dir  \
  --nproto <number_of_prototypes>            \
  --dim <encoder_output_dim>
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{acerbis2025slamags,
  title     = {SLAM-AGS: Slide-Label Aware Multi-Task Pretraining Using Adaptive Gradient Surgery in Computational Cytology},
  author    = {Acerbis, Marco and Chatterjee, Swarnadip and Avenel, Christophe and Lindblad, Joakim},
  booktitle = {IEEE International Symposium on Biomedical Imaging (ISBI)},
  year      = {2026}
}
```

## Contact

Corresponding author: **Marco Acerbis** — [marco.acerbis@it.uu.se](mailto:marco.acerbis@it.uu.se)
