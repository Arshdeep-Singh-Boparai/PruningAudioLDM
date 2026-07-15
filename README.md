# Efficient Text-to-Audio Generation via  Pruning

> **Official implementation** of our pruning framework for compressing the **AudioLDM-M-Full** text-to-audio (TTA) generative model.

[![Project Page](https://img.shields.io/badge/🌐-Project_Page-blue)](https://arshdeep-singh-boparai.github.io/EfficientAudioLDM/)
[![arXiv](https://img.shields.io/badge/arXiv-coming--soon-b31b1b.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📚 Table of Contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [Installation and AudioLDM Framework](#installation-and-audioldm-framework)
- [Download Checkpoints](#download-checkpoints)
- [1. Generate Layer-wise Sorted Channel Indexes](#1-generate-layer-wise-sorted-channel-indexes)
- [2. Generate Pruned U-Net Checkpoint](#2-generate-pruned-u-net-checkpoint)
- [3. Merge the Pruned Checkpoint](#3-merge-the-pruned-checkpoint)
- [4. Finetuning the Pruned AudioLDM-M-Full](#4-finetuning-the-pruned-audioldm-m-full)
- [5. Semantic Quality Analysis](#5-semantic-quality-analysis)
- [Citation](#citation)
- [Acknowledgment](#acknowledgment)

---


## Overview

This repository presents a **pruning framework** for compressing the **AudioLDM-M-Full** text-to-audio (TTA) generative model.

Our approach reduces the computational complexity, memory footprint, and inference cost of the diffusion U-Net by removing redundant convolutional filters while maintaining high audio generation quality.

The compressed models are evaluated on the **AudioCaps** test dataset using:

- **Frechet Audio Distance (FAD)**
- **KL Divergence (KL)**

---

## Pipeline

The proposed framework consists of five stages:

1. **Filter Importance Estimation**
   - Compute layer-wise channel importance rankings from a pretrained AudioLDM U-Net.

2. **Structured U-Net Pruning**
   - Generate compact U-Net architectures using block-wise channel scaling factors.

3. **Checkpoint Reconstruction**
   - Merge the pruned U-Net weights into the original AudioLDM checkpoint for inference or further finetuning.

4. **Finetuning**
   - Finetune the pruned AudioLDM-M-Full model using the AudioCaps training dataset.

5. **Semantic Quality Analysis**
   - Evaluate semantic preservation using PANNs by analysing the Top-10 predicted sound events before and after pruning and finetuning.

---

## Installation and AudioLDM Framework

This implementation is built upon the official **AudioLDM training, fine-tuning, inference, and evaluation framework**.

Please follow the official repository to install dependencies, prepare datasets, and configure the training environment.

**Official Repository**

https://github.com/haoheliu/AudioLDM-training-finetuning

Many thanks to **Haohe Liu** for making the AudioLDM framework publicly available.

---

## Download Checkpoints

Please download the pretrained checkpoints before running the pruning pipeline.

| Checkpoint | Status |
|------------|--------|
| AudioLDM-M-Full, Pretrained U-Net, Pruned Models | [Zenodo link](https://doi.org/10.5281/zenodo.21376822) |

---

# 1. Generate Layer-wise Sorted Channel Indexes

Compute channel importance rankings for every convolutional layer in the pretrained U-Net.

The generated index dictionary is used during structured pruning.

```bash
python layerwise_sorted_index_generation.py \
    --ckpt checkpoints/Unet_model-m.ckpt \
    --output pruned_indexes/B3_B4/sorted_indexes_dict.pkl
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--ckpt` | Path to the pretrained AudioLDM U-Net checkpoint |
| `--output` | Output path for the sorted channel index dictionary |

---

# 2. Generate Pruned U-Net Checkpoint

Generate a compact U-Net using structured channel pruning.

The pruning configuration is controlled by two block-wise scaling factors:

- **p** → Block-4 channel scaling factor
- **dp** → Block-3 channel scaling factor

```bash
python pruned_unet_dict_creation.py \
    --ckpt checkpoints/Unet_model-m.ckpt \
    --idx-dict pruned_indexes/B3_B4/sorted_indexes_dict.pkl \
    --output checkpoints/pruned/l1_unet_pruned_p2_dp2.pt \
    --p 2 \
    --dp 2
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--ckpt` | Pretrained U-Net checkpoint |
| `--idx-dict` | Layer-wise pruning index dictionary |
| `--output` | Output pruned checkpoint |
| `--p` | Block-4 scaling factor |
| `--dp` | Block-3 scaling factor |

---

# 3. Merge the Pruned Checkpoint

The generated checkpoint contains only the diffusion U-Net weights.

Merge these weights into the complete AudioLDM checkpoint while preserving all remaining model components.

```bash
python merge_pruned_checkpoint.py \
    --pruned-ckpt checkpoints/pruned/l1_unet_pruned_p2_dp2.pt \
    --full-ckpt checkpoints/original/audioldm-m-full.ckpt \
    --output checkpoints/l1_audioldm-m-full_p2_dp2.ckpt
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--pruned-ckpt` | Path to the pruned U-Net checkpoint |
| `--full-ckpt` | Original AudioLDM checkpoint |
| `--output` | Output merged checkpoint |
| `--prefix` | Parameter prefix (default: `model.diffusion_model.`) |

---

# 4. Finetuning the Pruned AudioLDM-M-Full

Finetuning follows exactly the same training pipeline as the official AudioLDM repository.

Before training, update

```text
audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium.yaml
```

Replace

```yaml
channel_mult: [1,2,3,5]
```

with

```yaml
channel_mult: [1,2,dp,p]
```

where `dp` (b3) and `p` (b4) correspond to channel scaling parameter for the third and fourth block of U-Net.

Then launch finetuning:

```bash
python3 audioldm_train/train/latent_diffusion.py \
    -c audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium.yaml \
    --reload_from_ckpt checkpoints/l1_audioldm-m-full_p2_dp2.ckpt
```

Please follow the official AudioLDM repository for evaluation of the generated audio.

---

# 5. Semantic Quality Analysis

We evaluate semantic preservation using a pretrained **PANNs** classifier.

For every generated audio clip, we obtain the **Top-10 predicted sound events** and analyse:

- Capture rate after pruning
- Missed sound events
- Recovery after finetuning
- Category-wise semantic preservation

Audio samples and results are available on the project website.

🌐 **Project Page**

https://arshdeep-singh-boparai.github.io/EfficientAudioLDM/

---

## Citation

If you find this repository useful in your research, please consider citing:

```bibtex
soon will be uploaded on arXiv.
```

---

## Acknowledgment

This repository builds upon the excellent open-source implementation of **AudioLDM** by Haohe Liu and collaborators.

We sincerely thank the authors for making their training and inference framework publicly available.

This work was supported by the **Engineering and Physical Sciences Research Council (EPSRC)** under Grant **EP/Y028805/1**.

For the purpose of Open Access, the authors have applied a Creative Commons Attribution (CC BY) licence to any Author Accepted Manuscript arising from this submission.
