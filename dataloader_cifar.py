from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch

# 核心修改：替换为CIFAR100官方超类-子类映射（语义相关分组）
# 每个超类包含5个语义相近的子类，非对称噪声在同超类内翻转
cifar100_superclass_subclasses = [
    [4, 30, 55, 72, 95],  # 超类0：水生哺乳动物（beaver, dolphin, otter, seal, whale）
    [1, 32, 67, 73, 91],  # 超类1：鱼类（aquarium_fish, flatfish, ray, shark, trout）
    [54, 62, 70, 82, 92],  # 超类2：花卉（orchid, poppy, rose, sunflower, tulip）
    [9, 10, 16, 28, 61],  # 超类3：食品容器（bottle, bowl, can, cup, plate）
    [0, 51, 53, 57, 83],  # 超类4：大型食肉动物（bear, leopard, lion, tiger, wolf）
    [22, 39, 40, 86, 87],  # 超类5：两栖动物（frog, newt, salamander, toad, turtle）
    [5, 20, 25, 84, 94],  # 超类6：中型哺乳动物（fox, porcupine, possum, raccoon, skunk）
    [6, 7, 14, 18, 24],  # 超类7：昆虫（bee, beetle, butterfly, caterpillar, cockroach）
    [3, 42, 43, 88, 97],  # 超类8：树木（oak, palm, pine, willow, maple）
    [12, 17, 37, 68, 76],  # 超类9：草本植物（grass, hedgehog, mushroom, cactus, fern）
    [23, 33, 49, 60, 71],  # 超类10：中型鸟类（hamster, mouse, rabbit, shrew, squirrel）
    [15, 19, 21, 31, 38],  # 超类11：大型鸟类（chicken, pigeon, seagull, sparrow, turkey）
    [34, 63, 64, 66, 75],  # 超类12：爬行动物（crocodile, dinosaur, lizard, snake, turtle）
    [26, 45, 77, 79, 99],  # 超类13：小型哺乳动物（hamster, mouse, rabbit, shrew, squirrel）
    [2, 11, 35, 46, 98],  # 超类14：灌木（maple, oak, palm, pine, willow）
    [27, 29, 44, 78, 93],  # 超类15：无脊椎动物（crab, lobster, snail, spider, worm）
    [36, 50, 65, 74, 80],  # 超类16：中型鱼类（eel, flatfish, ray, shark, trout）
    [47, 52, 56, 59, 96],  # 超类17：水果和蔬菜（apple, banana, cherry, orange, pear）
    [8, 13, 48, 58, 90],  # 超类18：小型鸟类（bluejay, cardinal, crow, dove, sparrow）
    [41, 69, 81, 85, 89]   # 超类19：蠕虫（earthworm, leech, slug, snail, worm）
]

def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


class cifar_dataset(Dataset):
    def __init__(self, dataset, root_dir, r, mode, noise_mode, noise_file='',
                 transform=None, test_form=None, pred=None, probability=None, log=None, **kwargs):
        """
        适配论文CIFAR80-O数据集的自定义Dataset类
        核心：OOD样本标注伪装成ID类标签，确保模型无信息泄露
        """
        self.transform = transform
        self.noise_mode = noise_mode
        self.r = r
        self.noise_file = noise_file
        self.mode = mode
        self.test_form = test_form
        self.pred = pred  # 样本可信性标记（与全量样本长度一致）
        self.probability = probability  # 样本可信概率（与全量样本长度一致）
        self.dataset = dataset
        self.log = log
        self.root_dir = root_dir

        # -------------------------- 1. 核心：ID/OOD子类划分（仅定义一次） --------------------------
        self.id_subclasses = list(range(80))  # ID类：0-79（模型需学习的目标类别）
        self.ood_subclasses = list(range(80, 100))  # OOD类：80-99（需检测的未知类别）

        # -------------------------- 2. 加载原始数据（train/test） --------------------------
        if mode == 'test':
            data_file = os.path.join(root_dir, "test")
            dataset_data = unpickle(data_file)
            self.data = dataset_data['data'].reshape((10000, 3, 32, 32)).transpose((0, 2, 3, 1))
            raw_labels = np.array(dataset_data['fine_labels'])  # 原始标签（0-99）
        else:
            data_file = os.path.join(root_dir, "train")
            dataset_data = unpickle(data_file)
            self.data = dataset_data['data'].reshape((50000, 3, 32, 32)).transpose((0, 2, 3, 1))
            raw_labels = np.array(dataset_data['fine_labels'])  # 原始标签（0-99）

        # -------------------------- 3. 标签映射：ID→0-79，OOD→80-99（全量样本） --------------------------
        orig2new = {}
        for i, orig in enumerate(self.id_subclasses):
            orig2new[orig] = i  # ID类原始标签→0-79（模型输出空间）
        for orig in self.ood_subclasses:
            orig2new[orig] = orig  # OOD类原始标签→80-99（仅内部使用）

        # 全量样本映射（长度=总样本数：训练集50000/测试集10000）
        mapped_labels = np.full_like(raw_labels, -1)
        for orig, new in orig2new.items():
            mapped_labels[raw_labels == orig] = new

        # 初始化干净标签和噪声标签（全量样本，长度一致）
        self.clean_label = mapped_labels.copy()  # 真实标签：ID→0-79，OOD→80-99（内部评估用）
        self.noise_label = mapped_labels.copy()  # 噪声标签：初始与干净标签一致，后续处理

        # -------------------------- 4. 测试集无需噪声处理，直接返回 --------------------------
        if mode == 'test':
            self.index = np.arange(len(self.data))  # 测试集全局索引
            return

        # -------------------------- 5. 样本权重与全局索引（全量样本） --------------------------
        if mode in ['labeled', 'unlabeled']:
            if self.probability is not None:
                self.probability = np.asarray(self.probability, dtype=np.float32)
                assert len(self.probability) == len(self.data), "probability长度必须为50000"
                self.w_x = torch.tensor(self.probability, dtype=torch.float32)
            else:
                self.w_x = torch.ones(len(self.data), dtype=torch.float32)
        else:
            self.w_x = None

        self.index = np.arange(len(self.data))  # 训练集全局索引（50000）
        # ========= 依据 pred 掩码做真切分（关键修复） =========
        if self.mode in ['labeled', 'unlabeled']:
            if self.pred is None:
                print("[警告] 未提供 pred 掩码，默认全量样本，这会使半监督失效。")
            else:
                mask = np.asarray(self.pred, dtype=np.bool_)
                assert mask.shape[0] == len(self.index), "pred 掩码长度必须为50000"
                sub_idx = self.index[mask] if self.mode == 'labeled' else self.index[~mask]
                self.index = sub_idx
                if self.w_x is not None:
                    idx_tensor = torch.from_numpy(self.index.copy()).long()
                    self.w_x = self.w_x.index_select(0, idx_tensor)
        # =======================================================

        # -------------------------- 6. 验证标签映射正确性 --------------------------
        print(f"[DataLoader检查] 干净标签范围 {self.clean_label.min()}..{self.clean_label.max()} | "
              f"ID(<80)={(self.clean_label < 80).sum()} | OOD(>=80)={(self.clean_label >= 80).sum()}")
        assert (self.clean_label >= 80).any(), "未检测到OOD样本（标签≥80），请检查ID/OOD划分"

        # -------------------------- 7. 核心：生成噪声标签（含OOD伪装） --------------------------
        if os.path.exists(self.noise_file):
            self.noise_label = np.array(json.load(open(self.noise_file, "r")), dtype=np.int64)
            assert len(self.noise_label) == 50000, "噪声标签必须为50000个样本"
            print(f"[CIFAR80N-O] 已加载噪声文件: {self.noise_file}")
        else:
            # 论文定义的噪声率
            n_all = self.r  # 整体噪声率
            n_c = max(0.0, (n_all - 0.2) / 0.8)  # ID区域噪声率
            print(f"[CIFAR80N-O] 整体噪声率 n_all={n_all:.2f}, ID区域噪声率 n_c={n_c:.2f}")

            # 划分ID/OOD样本索引（基于clean_label）
            id_mask = (self.clean_label < 80)  # 真实ID样本（0-79）
            ood_mask = (self.clean_label >= 80)  # 真实OOD样本（80-99）
            id_indices = np.where(id_mask)[0]  # ID样本全局索引（≈40000）
            ood_indices = np.where(ood_mask)[0]  # OOD样本全局索引（≈10000）

            # 核心修改1：OOD样本标注伪装成ID类标签（0-79），模型无法从标注区分
            for i in ood_indices:
                self.noise_label[i] = random.choice(self.id_subclasses)  # 伪装为ID类标签

            # 对ID样本注入噪声（对称/非对称）
            num_id_noise = int(n_c * len(id_indices))
            noise_id_idx = np.random.choice(id_indices, num_id_noise, replace=False)
            noise_id_set = set(noise_id_idx)

            if self.noise_mode == 'sym':
                for i in id_indices:
                    if i in noise_id_set:
                        possible = [c for c in range(80) if c != self.noise_label[i]]
                        self.noise_label[i] = random.choice(possible)
            elif self.noise_mode == 'asym':
                # 超类映射（确保在0-99范围内）
                sub_to_super = {}
                for sc, subs in enumerate(cifar100_superclass_subclasses):
                    for sub in subs:
                        sub_to_super[sub] = sc
                for i in id_indices:
                    if i in noise_id_set:
                        orig_sub = self.noise_label[i]
                        sc = sub_to_super[orig_sub]
                        same_super = [sub for sub in cifar100_superclass_subclasses[sc]
                                     if sub < 80 and sub != orig_sub]
                        if same_super:
                            self.noise_label[i] = random.choice(same_super)

            # 保存噪声标签（长度50000）
            json.dump(self.noise_label.tolist(), open(self.noise_file, "w"))
            print(f"[CIFAR80N-O] 已保存噪声标签至 {self.noise_file}")

        # 核心检查：确保clean_label和noise_label长度均为50000
        assert len(self.clean_label) == len(self.noise_label) == 50000, \
            f"标签长度不匹配：clean={len(self.clean_label)}, noise={len(self.noise_label)}"
        # 核心检查：确保所有noise_label均在ID类空间（0-79），OOD伪装生效
        assert (self.noise_label < 80).all(), "OOD样本未成功伪装，噪声标签存在≥80的值"

    def __getitem__(self, index):
        real_index = self.index[index]  # ✅ 子集索引（防止全量访问）
        image = Image.fromarray(self.data[real_index])
        global_index = real_index

        # 多视图增强（4个视图）
        if self.transform is not None and isinstance(self.transform, list) and len(self.transform)>=4:
            inputs_x1 = self.transform[0](image)
            inputs_x2 = self.transform[1](image)
            inputs_x3 = self.transform[2](image)
            inputs_x4 = self.transform[3](image)
        else:
            t = self.transform(image) if self.transform else image
            inputs_x1 = inputs_x2 = inputs_x3 = inputs_x4 = t

        # 核心修改2：训练模式返回noise_label（含OOD伪装），测试模式返回clean_label
        if self.mode in ['labeled', 'unlabeled', 'all']:
            label = self.noise_label[real_index]  # 训练用：OOD已伪装成0-79
        else:
            label = self.clean_label[real_index]  # 测试用：真实标签（ID→0-79，OOD→80-99）

        # 根据模式返回数据
        if self.mode == 'unlabeled':
            return inputs_x1, inputs_x2, inputs_x3, inputs_x4
        elif self.mode == 'labeled':
            w_x = self.w_x[index] if self.w_x is not None else torch.tensor(1.0)
            return inputs_x1, inputs_x2, inputs_x3, inputs_x4, label, global_index, w_x
        elif self.mode == 'all':
            return inputs_x1, label, global_index
        else:  # test模式
            return inputs_x1, label

    def __len__(self):
        return len(self.index)

    def if_noise(self):
        """判断噪声样本（噪声标签≠干净标签），长度均为50000，无报错"""
        noise_mask = self.noise_label != self.clean_label
        noise_indices = np.where(noise_mask)[0]
        clean_indices = np.where(~noise_mask)[0]
        return noise_indices, clean_indices


class cifar_dataloader():
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, log, noise_file=''):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file

        # 定义 CIFAR80N-O 数据集的标准化和数据增强操作
        transform_cifar80no = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),  # CIFAR-100的均值和标准差
        ])

        # 这里删除了 CIFAR-10 和 CIFAR-100 的处理，直接使用 CIFAR80N-O 的设置
        self.transform = {
            "warmup": transform_cifar80no,
            "unlabeled": [transform_cifar80no, transform_cifar80no, transform_cifar80no, transform_cifar80no],
            "labeled": [transform_cifar80no, transform_cifar80no, transform_cifar80no, transform_cifar80no]
        }

        # 对测试集的标准化
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),  # CIFAR-100的均值和标准差
        ])

    def run(self, mode, pred=[], prob=[]):
        if mode == 'warmup':
            all_dataset = cifar_dataset(
                dataset=self.dataset,
                noise_mode=self.noise_mode,  # 仅占位，测试模式不使用
                r=self.r,  # 仅占位，测试模式不使用
                root_dir=self.root_dir,
                transform=self.transform["warmup"], mode="all",
                noise_file=self.noise_file
            )
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            return trainloader

        elif mode == 'train':
            # === 修改 1：防止 CUDA 张量传入 DataLoader ===
            if torch.is_tensor(pred):
                pred_cpu = pred.detach().cpu().numpy().astype(np.bool_)
            else:
                pred_cpu = np.asarray(pred, dtype=np.bool_)

            if torch.is_tensor(prob):
                prob_cpu = prob.detach().cpu().numpy().astype(np.float32)
            else:
                prob_cpu = np.asarray(prob, dtype=np.float32)
            # ================================================

            labeled_dataset = cifar_dataset(
                dataset=self.dataset,
                noise_mode=self.noise_mode,  # 仅占位，测试模式不使用
                r=self.r,  # 仅占位，测试模式不使用
                root_dir=self.root_dir,
                transform=self.transform["labeled"], mode="labeled",
                noise_file=self.noise_file, pred=pred_cpu, probability=prob_cpu, log=self.log
            )
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)

            unlabeled_dataset = cifar_dataset(
                dataset=self.dataset,
                noise_mode=self.noise_mode,  # 仅占位，测试模式不使用
                r=self.r,  # 仅占位，测试模式不使用
                root_dir=self.root_dir,
                transform=self.transform["unlabeled"], mode="unlabeled",
                noise_file=self.noise_file, pred=pred_cpu,  log=self.log
            )
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            return labeled_trainloader, unlabeled_trainloader

        elif mode == 'test':
            test_dataset = cifar_dataset(
                dataset=self.dataset,
                root_dir=self.root_dir,
                r=self.r,  # 补充传递r参数（从dataloader的初始化参数获取）
                noise_mode=self.noise_mode,  # 补充传递noise_mode参数（从dataloader的初始化参数获取）
                transform=self.transform_test, mode='test')
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return test_loader

        elif mode == 'eval_train':
            eval_dataset = cifar_dataset(
                dataset=self.dataset,
                noise_mode=self.noise_mode,  # 仅占位，测试模式不使用
                r=self.r,  # 仅占位，测试模式不使用
                root_dir=self.root_dir,
                transform=self.transform_test, mode='all', noise_file=self.noise_file)
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return eval_loader
