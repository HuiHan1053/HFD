from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random
import torchvision.transforms as transforms


class SubPolicy(object):
    """子策略类，实现单组增强操作组合（论文中每组包含2个操作）"""
    def __init__(
        self,
        p1,
        operation1,
        magnitude_idx1,
        p2,
        operation2,
        magnitude_idx2,
        fillcolor=(128, 128, 128),
    ):
        # 增强操作的强度范围（与原论文一致，针对CIFAR优化）
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),  # 基于331尺寸归一化，适配小尺寸图像
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),  # 最大旋转30度
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,  # 无参数操作
            "equalize": [0] * 10,
            "invert": [0] * 10,
        }

        # 旋转时填充背景色（避免黑边，与CIFAR图像风格适配）
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(
                rot, Image.new("RGBA", rot.size, (128,) * 4), rot
            ).convert(img.mode)

        # 增强操作函数映射（完整实现论文中所有变换）
        self.func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC,
                fillcolor=fillcolor,
            ),
            "shearY": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC,
                fillcolor=fillcolor,
            ),
            "translateX": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor,
            ),
            "translateY": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor,
            ),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(
                1 + magnitude * random.choice([-1, 1])
            ),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])
            ),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])
            ),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])
            ),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img),
        }

        # 初始化当前子策略的两个操作参数
        self.p1 = p1  # 第一个操作的执行概率
        self.operation1 = self.func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]  # 第一个操作的强度
        self.p2 = p2  # 第二个操作的执行概率
        self.operation2 = self.func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]  # 第二个操作的强度

    def __call__(self, img):
        # 按概率执行第一个操作
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1)
        # 按概率执行第二个操作
        if random.random() < self.p2:
            img = self.operation2(img, self.magnitude2)
        return img


class CIFAR10Policy(object):
    """
    论文中针对CIFAR系列（CIFAR10/CIFAR100）优化的AutoAugment策略
    包含25组子策略，与原论文完全一致
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),
            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),
            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),
            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.6, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),
            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor),
        ]

    def __call__(self, img):
        # 随机选择一组子策略执行
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR Policy (适配CIFAR10/CIFAR100)"


def get_cifar10_weak_augmentation():
    """CIFAR10弱增强（论文中w(x)）：随机裁剪+水平翻转+归一化"""
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],  # CIFAR10官方均值
            std=[0.2470, 0.2435, 0.2616]     # CIFAR10官方标准差
        )
    ])


def get_cifar10_strong_augmentation():
    """CIFAR10强增强（论文中s(x)）：弱增强+AutoAugment"""
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFARPolicy(),  # 加入完整AutoAugment策略
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])


def get_cifar100_weak_augmentation():
    """CIFAR100弱增强（论文中w(x)）：随机裁剪+水平翻转+归一化"""
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],  # CIFAR100官方均值
            std=[0.2675, 0.2565, 0.2761]     # CIFAR100官方标准差
        )
    ])


def get_cifar100_strong_augmentation():
    """CIFAR100强增强（论文中s(x)）：弱增强+AutoAugment"""
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFARPolicy(),  # 加入完整AutoAugment策略
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        )
    ])