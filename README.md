# HFD
We'll soon update the code and data.

# Project / 项目

PyTorch code for noisy-label learning with **two networks, **GMM-based clean/noisy split**, and optional **OOD filtering**.  
PyTorch 噪声标签学习代码：**双网络**、**GMM 划分干净/噪声**，可选 **OOD 过滤**。

## Structure / 结构

```text
.
├─ train_cifar10.py            # CIFAR-10 training entry (two-net + GMM + proto/contrastive)
├─ train_cifar100.py           # Main entry used in this project (CIFAR80N-O / CIFAR80-O style pipeline)
├─ train_animal10n.py          # Animal10N entry (if needed)
├─ train_clothing1m.py         # Clothing1M entry (if needed)
├─ train_clothing1mall.py      # Clothing1M (all) entry (if needed)
├─ train_tinyImageNet.py       # TinyImageNet entry (if needed)
│
├─ dataloader_cifar.py         # Dataset + DataLoader; noisy-label injection; ID/OOD mapping; train split by masks
├─ augmentation_cifar.py       # Data augmentation policies
├─ asymmetric_noise.py         # Asymmetric noise utilities
│
├─ PreResNet.py                # ResNet backbone with projector head (feature, logits)
├─ dnn7.py                     # DNN7 backbone used by `train_cifar100.py`
│
├─ contrastive_loss.py         # Supervised contrastive loss (SupConLoss)
└─ 绘图测试.py                 # Plot/visualization experiments (optional)
```

Quickstart / 快速开始

Adjust `--data_path` to your dataset folder.  
把 `--data_path` 改成你本机数据集路径。

```bash
# CIFAR80N-O / CIFAR80-O style
python train_cifar100.py --dataset cifar80o --data_path "PATH_TO/cifar-100-python" --noise_mode sym --r 0.2

# CIFAR-10
python train_cifar10.py --dataset cifar10 --data_path "PATH_TO/cifar-10-batches-py" --noise_mode asym --r 0.4
```

