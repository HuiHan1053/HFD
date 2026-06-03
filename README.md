Noisy-Label Learning with Dual-Net Co-training & SupCon + GMM Prototype
> PyTorch Implementation for Robust Learning under Label Noise | Support CIFAR-10/100, Animal10N, Clothing1M, TinyImageNet | Built-in ID/OOD partition for CIFAR-100

📖 Overview
This repo provides complete PyTorch source code for **noisy label robust learning**, featuring dual-network co-training framework combined with prototype-based GMM loss and supervised contrastive learning.
Specially, CIFAR-100 pipeline integrates customized **ID/OOD sample split mechanism** encapsulated in `dataloader_cifar.py`.

Core Technical Contributions
- ✅ Dual-network co-training framework (partial pipeline equipped with EMA momentum teacher model)
- ✅ GMM-driven prototype/centroid loss (per-sample loss distribution fitting for clean/noisy sample discrimination)
- ✅ Supervised Contrastive Loss (`SupConLoss`) for discriminative feature representation

For code readability and maintenance, auxiliary functions of original `train_cifar100.py` are decoupled into standalone module `our_cifar/`.

⚡ Quick Start
1. Environment Setup
```bash
# Install all required dependencies
pip install -r requirements.txt
```

2. Run Training Commands
```bash
 CIFAR-10 noisy label training
python train_cifar10.py --dataset cifar10 --data_path /path/to/cifar-10-batches-py --gpuid 0

CIFAR-100 (built-in ID/OOD split strategy)
python train_cifar100.py --dataset cifar100 --data_path /path/to/cifar-100-python --gpuid 0
```

📁 Repository File Structure
```
├── train_*.py                # Entry scripts for all datasets
├── Models/
│   ├── ResNet.py             # ResNet backbone (return feature + logits)
│   └── dnn7.py               # DNN7 CNN backbone for CIFAR-100
├── Loss/
│   └── contrastive_loss.py  # Supervised Contrastive Loss implementation
├── Data/
│   ├── dataloader_cifar.py  # CIFAR dataloader + noise injection + ID/OOD split
│   ├── augmentation_cifar.py# Data augmentation policies
│   └── asymmetric_noise.py  # Asymmetric label noise generation tool
└── our_cifar/               # Decoupled helper modules for CIFAR-100
    ├── cache.py             # Data caching: OOD filter / clean split / uncertainty partition
    ├── prototypes.py        # Prototype init & EMA dynamic update
    ├── vis.py               # Visualization: GMM histogram, t-SNE feature plot
    └── metrics.py           # Evaluation metric calculator for training log
```
Entry Script Description
| Script | Function |
|--------|----------|
| train_cifar10.py | Training pipeline on CIFAR-10 with noisy labels |
| train_cifar100.py | CIFAR-100 training + ID/OOD partition + abundant experimental tools |
| train_animal10n.py / train_clothing1m*.py / train_tinyImageNet.py | Training for corresponding real-world noisy datasets |
| 绘图测试.py | Local-only visualization debug script (excluded from git tracking) |

📦 our_cifar Module Details
All utilities migrated from original `train_cifar100.py`:
- `cache.py`: `DataCache` module to eliminate OOD samples, screen clean samples, divide high/low uncertainty subsets
- `prototypes.py`: Class prototype initialization and EMA iterative update logic
- `vis.py`: Draw GMM loss distribution histogram & collect features for t-SNE visualization
- `metrics.py`: Lightweight metric computation for training log recording

📌 Output & Git Ignore Rules
Auto-generated runtime directories are added to `.gitignore`,
- `checkpoint/`: Saved model weights + training log files
- `figure_his/` & `figure_vis/`: Generated histogram & feature visualization figures

