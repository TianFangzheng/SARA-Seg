<div align="center">

# SARA-Seg

### Semantic Anchoring and Reliability-Aware Adaptation for Medical Image Segmentation

**Jia Gu · Fangzheng Tian · Yanli Liu**

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://sara-seg.netlify.app)
![Code Status](https://img.shields.io/badge/Code-Core%20Implementation%20Available-brightgreen)
![Framework](https://img.shields.io/badge/Framework-PyTorch-ee4c2c)

</div>

## Overview

SARA-Seg is a medical image segmentation framework that integrates **semantic anchoring**, **reliability-aware adaptation**, and **structure-consistent refinement**. The current repository provides the core model implementation, dataset loader, loss functions, training pipeline, and an example configuration.

## Repository Structure

```text
sara_seg_final/
├── sara_seg/
│   ├── __init__.py
│   ├── datasets.py
│   ├── losses.py
│   ├── model.py
│   ├── tokenizer.py
│   └── utils.py
├── configs/
│   └── sara_seg_busi.yaml
├── train_sara_seg.py
└── README.md
```

## Main Components

- **Semantic Anchoring:** constructs image-specific semantic anchors from internally distilled representations and calibrates textual semantics.
- **Reliability-Aware Adaptation:** performs multi-source semantic interaction and probabilistic gate-controlled feature updating.
- **Structure-Consistent Refinement:** coordinates region and boundary prediction and feeds boundary cues back into the feature representation.

## Installation

```bash
conda create -n sara-seg python=3.10 -y
conda activate sara-seg
pip install torch torchvision numpy pandas pillow pyyaml openpyxl
```

## Data Preparation

Each dataset should follow the structure below:

```text
DATASET_ROOT/
├── img/
├── label/
├── Train_text.xlsx
├── Val_text.xlsx
└── Test_text.xlsx        # optional
```

Each spreadsheet should contain:

- an image identifier column, such as `image`, `image_id`, `filename`, or `name`;
- a text column, such as `text`, `prompt`, `caption`, or `description`.

The image and mask files should use the same stem. A mask file may optionally use the suffix `_mask`.

## Training

Update the dataset path and training settings in `configs/sara_seg_busi.yaml`, and then run:

```bash
python train_sara_seg.py --config configs/sara_seg_busi.yaml
```

The training script saves:

```text
best.pt
last.pt
history.json
resolved_config.json
tokenizer.json
```

## Reproducibility Notes

The current implementation includes:

- fixed random seed support;
- online and EMA teacher branches;
- soft semantic-anchor assignment;
- anchor-guided token representation `A_m = P A`;
- probabilistic adaptation gating;
- region and boundary prediction branches;
- region-boundary consistency loss;
- boundary-aware feature feedback.

Additional dataset configurations, evaluation scripts, and pretrained checkpoints will be organized and released progressively.

## Citation

The citation information will be updated after publication.

## Contact

For questions, please contact:

**Yanli Liu**  
School of Electronic Information Engineering, Shanghai Dianji University  
Email: liuyl@sdju.edu.cn
