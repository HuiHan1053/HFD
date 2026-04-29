## Overview

This repository contains PyTorch training code for **learning with noisy labels**. For CIFAR-100, the code also includes an **ID/OOD-style split** setup implemented in `dataloader_cifar.py`.

Key ideas implemented in the training scripts:

- Two-network co-training pipeline (with EMA teacher in some parts)
- Prototype / center-based loss modeling (GMM over per-sample losses)
- Supervised contrastive loss (`SupConLoss`)

For GitHub friendliness, `train_cifar100.py` was split: helper utilities were extracted into `our_cifar/` and the entry script now imports them.

---

## Quick start

### Environment

```bash
pip install -r requirements.txt
```

### Run (examples)

#### CIFAR-10

```bash
python train_cifar10.py --dataset cifar10 --data_path /path/to/cifar-10-batches-py --gpuid 0
```

#### CIFAR-100 (ID/OOD-style setup implemented in `dataloader_cifar.py`)

```bash
python train_cifar100.py --dataset cifar100 --data_path /path/to/cifar-100-python --gpuid 0
```

---

## Repository structure

- **Training entrypoints**
  - `train_cifar10.py`: CIFAR-10 noisy-label training entry.
  - `train_cifar100.py`: CIFAR-100 training entry (includes ID/OOD-style split logic and many experimental utilities).
  - `train_animal10n.py`, `train_clothing1m.py`, `train_clothing1mall.py`, `train_tinyImageNet.py`: dataset-specific training scripts.
  - `绘图测试.py`: local plotting/testing script (outputs are ignored by `.gitignore`).

- **Models**
  - `ResNet.py`: ResNet backbone (returns `(feature, logits)`).
  - `dnn7.py`: CNN backbone `DNN7` (feature extractor + head), used by `train_cifar100.py`.

- **Loss**
  - `contrastive_loss.py`: `SupConLoss` (Supervised Contrastive Learning).

- **Data**
  - `dataloader_cifar.py`: CIFAR dataset loader and transformations; includes noise injection and (for CIFAR-100) ID/OOD style label mapping and masking utilities.
  - `augmentation_cifar.py`, `asymmetric_noise.py`: augmentation policies and asymmetric noise helpers.

---

## Split helpers (`our_cifar/`)

Utilities extracted from `train_cifar100.py`:

- `our_cifar/cache.py`: `DataCache` (OOD removal + clean intersection + high/low uncertain partition)
- `our_cifar/prototypes.py`: prototype initialization and EMA update
- `our_cifar/vis.py`: visualization helpers (GMM histogram, t-SNE feature collection/plot)
- `our_cifar/metrics.py`: small metrics helpers used by training logs

---

## Outputs (not tracked by git)

The training scripts will create folders like:

- `checkpoint/`: model checkpoints and logs
- `figure_his/`, `figure_vis/`: figures

These are intentionally ignored by `.gitignore`.
