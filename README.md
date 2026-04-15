# HFD
We'll soon update the code and data.

# Project 

PyTorch code for noisy-label learning with **two networks, **GMM-based clean/noisy split**, and optional **OOD filtering**.  

## Structure 

```text
├─ train_cifar10.py            # CIFAR-10 training entry (two-net + GMM + proto/contrastive)
├─ train_cifar100.py           # Main entry used in this project (CIFAR80N-O / CIFAR80-O style pipeline)
├─ train_animal10n.py          # Animal10N entry 
├─ train_clothing1m.py         # Clothing1M entry 
├─ train_clothing1mall.py      # Clothing1M (all) entry 
├─ train_tinyImageNet.py       # TinyImageNet entry 
│
├─ dataloader_cifar.py         # Dataset + DataLoader; noisy-label injection; ID/OOD mapping; train split by masks
├─ augmentation_cifar.py       # Data augmentation policies
├─ asymmetric_noise.py         # Asymmetric noise utilities
```
```bash
# CIFAR80N-O / CIFAR80-O style
python train_cifar100.py --dataset cifar80o --data_path "PATH_TO/cifar-100-python" --noise_mode sym --r 0.2

# CIFAR-10
python train_cifar10.py --dataset cifar10 --data_path "PATH_TO/cifar-10-batches-py" --noise_mode asym --r 0.4
```

