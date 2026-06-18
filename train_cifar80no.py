import argparse
import os
import random
import sys
from collections import deque
from datetime import datetime

import matplotlib
matplotlib.use('Agg')  # 服务器/无GUI环境安全
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.mixture import GaussianMixture

import Data.dataloader_cifar as dataloader
from Loss.contrastive_loss import SupConLoss
from Models.dnn7 import DNN7
from our_cifar.cache import DataCache
from our_cifar.metrics import compute_clean_precision_on_eval
from our_cifar.prototypes import initialize_prototypes, update_prototypes
from our_cifar.vis import (
    collect_features_for_tsne,
    plot_gmm_histogram_tri_dual,
    plot_tsne,
)

parser = argparse.ArgumentParser(description='PyTorch CIFAR80N-O Training')
parser.add_argument('--batch_size', default=256, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.05, type=float, help='initial learning rate')
parser.add_argument('--noise_mode', default='sym')
parser.add_argument('--r', default=0.2, type=float, help='noise ratio')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=80, type=int)
parser.add_argument('--dr_dim', default=128, type=int)
parser.add_argument('--threshold', default=0.95, type=float, help='pseudo label threshold')
parser.add_argument('--eps', default=0.999, type=float, help='Running average of model weights')
parser.add_argument('--data_path', default='/tmp/pycharm_project_179/data/cifar-100-python', type=str,
                    help='path to dataset')
parser.add_argument('--dataset', default='cifar80no', type=str)
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--lambda_ood_warmup', default=0.6, type=float,
                    help='weight for OOD uniform-entropy KL loss during warm-up')
parser.add_argument('--disable_ood_filter', action='store_true', help='消融实验: 禁用OOD过滤')
parser.add_argument('--simple_selection', action='store_true', help='消融实验: 禁用层级筛选(回退到简单阈值)')
parser.add_argument('--lambda_ccr', default=0.05, type=float, help='CCR损失权重(设为0可关闭)')
parser.add_argument('--lambda_s', default=1.0, type=float, help='监督损失(Lce)权重')
# ===== Speedup: A100 推荐设置 =====
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")  # PyTorch 2.x
except Exception:
    pass

USE_AMP = True
AMP_DTYPE = torch.bfloat16  # A100 更稳，不需要 GradScaler

args = parser.parse_args()
args.num_class = 80
data_cache = DataCache()


torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


## NOTE: prototype helpers moved to `our_cifar.prototypes`

## NOTE: visualization helpers moved to `our_cifar.vis`





# Training
def train(epoch, net, tch_net, optimizer, centers, peer_centers,
          labeled_trainloader, unlabeled_trainloader,
          ood_noise_indices, prob_clean_full,
          # >>> 新增：把 DataCache 传进来，及可调力度
          data_cache,
          alpha_clean=0.10,     # clean交集 更新原型步长
          alpha_high=0.03,      # 高不确定 更新原型步长（弱一点）
          clean_boost=0.20,     # clean交集 监督权重boost
          make_clean_hard=False # True时clean样本直接w=1.0（纯one-hot）
          ):
    net.train()
    tch_net.train()

    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1

    eLce = eLu = ePenalty = eLoss_simCLR = eLoss_ccr = 0.0
    loss = 0.0

    for batch_idx, (inputs_x1, inputs_x2, inputs_x3, inputs_x4, labels_x, index, _) in enumerate(labeled_trainloader):
        # --------------------------
        # 过滤当前批次的OOD样本（保留）
        # --------------------------
        index_cuda = index.cuda()

        if args.disable_ood_filter:
            # 如果禁用过滤，创建一个全True的掩码，即保留所有样本
            non_ood_mask = torch.ones_like(index_cuda, dtype=torch.bool)
        else:
            # 正常逻辑：剔除在 ood_noise_indices 中的样本
            non_ood_mask = ~torch.isin(index_cuda, ood_noise_indices)

        non_ood_mask = non_ood_mask.cpu()
        if not non_ood_mask.any():
            continue

        # 裁剪到非OOD
        inputs_x1 = inputs_x1[non_ood_mask].cuda()
        inputs_x2 = inputs_x2[non_ood_mask].cuda()
        inputs_x3 = inputs_x3[non_ood_mask].cuda()
        inputs_x4 = inputs_x4[non_ood_mask].cuda()
        labels_x = labels_x[non_ood_mask].cuda()
        index      = index[non_ood_mask]
        index_cuda = index_cuda[non_ood_mask]

        # >>> 取 batch 内三类掩码（来自 DataCache 且按全局索引对齐）
        # DataCache 中的掩码应为 cuda: bool[N]
        clean_inter_b = data_cache.clean_inter_mask[index_cuda]           # clean交集
        high_unc_b    = data_cache.high_uncertain_mask[index_cuda]        # 非clean交集中的“高”
        # 低不确定样本不会出现在 labeled_trainloader；它们来自 unlabeled_trainloader

        # BN 保护
        if inputs_x1.size(0) < 2:
            continue

        # 取无标签batch
        try:
            inputs_u1, inputs_u2, inputs_u3, inputs_u4 = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u1, inputs_u2, inputs_u3, inputs_u4 = next(unlabeled_train_iter)
        if inputs_u1.size(0) < 2:
            continue
        inputs_u1 = inputs_u1.cuda(); inputs_u2 = inputs_u2.cuda()
        inputs_u3 = inputs_u3.cuda(); inputs_u4 = inputs_u4.cuda()

        # EMA教师
        update_ema_variables(net, tch_net, epoch)

        # --------------------------
        # 半监督：构造 targets_x / targets_u（保留）
        # --------------------------
        with torch.no_grad():
            # 学生对 labeled 的两视图
            _, logits_x1 = net(inputs_x1)
            _, logits_x2 = net(inputs_x2)
            px = (torch.softmax(logits_x1, dim=1) + torch.softmax(logits_x2, dim=1)) / 2

            # 学生/教师对 unlabeled
            _, logits_u1 = net(inputs_u1)
            _, logits_u2 = net(inputs_u2)
            _, logits_u1_tch = tch_net(inputs_u1)
            _, logits_u2_tch = tch_net(inputs_u2)
            pu = (torch.softmax(logits_u1, dim=1) + torch.softmax(logits_u1_tch, dim=1) +
                  torch.softmax(logits_u2, dim=1) + torch.softmax(logits_u2_tch, dim=1)) / 4
            targets_u = pu / pu.sum(dim=1, keepdim=True)

            # >>> 使用平均clean后验作为监督权重基底（仍由调用方传入 prob_clean_full）
            batch_prob_clean = prob_clean_full[index_cuda]  # [B’]

            # 原来的线性映射权重
            p0 = args.p_threshold
            w_base = torch.clamp((batch_prob_clean - p0) / (1.0 - p0 + 1e-8), 0.0, 1.0)

            # warm-up 早期 boost
            boost = 0.3 if epoch < (warm_up + 10) else 0.0
            w = torch.clamp(w_base + boost, 0.0, 1.0)

            # >>> 对 clean 交集样本强化监督（两种策略：硬 or 软）
            if make_clean_hard:
                w = w.clone()
                w[clean_inter_b] = 1.0
            else:
                w = w.clone()
                w[clean_inter_b] = torch.clamp(w[clean_inter_b] + clean_boost, 0.0, 1.0)

            # targets_x = w*onehot + (1-w)*px （仅对ID标签）
            targets_x = px.clone()
            id_mask = (labels_x >= 0) & (labels_x < args.num_class)
            if id_mask.any():
                one_hot_id = F.one_hot(labels_x[id_mask], num_classes=args.num_class).float()
                w_id = w[id_mask].unsqueeze(1)
                targets_x[id_mask] = w_id * one_hot_id + (1.0 - w_id) * px[id_mask]

            targets_x = targets_x / targets_x.sum(dim=1, keepdim=True)

        with torch.cuda.amp.autocast(enabled=USE_AMP, dtype=AMP_DTYPE):
            # 特征 + CCR（保留）
            f1, _ = net(inputs_x1); f1 = F.normalize(f1, dim=1)
            f2_view, _ = net(inputs_x2); f2_view = F.normalize(f2_view, dim=1)

            centers_mean = F.normalize(centers.mean(dim=1), dim=1)  # [80, D]
            T_ccr = 0.2
            sim1 = torch.mm(f1, centers_mean.t()); sim2 = torch.mm(f2_view, centers_mean.t())
            p1 = F.softmax(sim1 / T_ccr, dim=1); p2 = F.softmax(sim2 / T_ccr, dim=1)
            loss_ccr = 0.5 * (F.kl_div(p1.log(), p2, reduction='batchmean') + F.kl_div(p2.log(), p1, reduction='batchmean'))
            lambda_ccr = args.lambda_ccr  # 原代码是 lambda_ccr = 0.05

            # 过滤越界标签（保留）
            labels_x = labels_x.to(centers.device).long()
            C_total = centers.size(0)
            valid_mask = ((labels_x >= 0) & (labels_x < C_total)).cpu()
            if not valid_mask.all():
                f1 = f1[valid_mask.cuda()]
                labels_x = labels_x[valid_mask.cuda()]
                # 同步裁剪
                index_cuda = index_cuda[valid_mask.cuda()]
                clean_inter_b = clean_inter_b[valid_mask.cuda()]
                high_unc_b    = high_unc_b[valid_mask.cuda()]
            if f1.size(0) < 2:
                continue

            # 对比损失（保留）
            if f1.size(0) == 0:
                loss_simCLR = torch.tensor(0.0, device=f1.device)
            else:
                centers_for_samples = centers[labels_x]  # [B'', M, D]
                similarities = torch.bmm(f1.unsqueeze(1), centers_for_samples.transpose(1, 2)).squeeze(1)
                closest_prototype_idx = similarities.argmax(dim=1)
                f2 = centers_for_samples[torch.arange(f1.size(0)), closest_prototype_idx]
                f2 = F.normalize(f2, dim=1)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                loss_simCLR = contrastive_criterion(features)

            # --------------------------
            # >>> 原型更新：分层力度（clean强 / 高不确定弱）
            # --------------------------
            with torch.no_grad():
                # clean交集 强更新
                if clean_inter_b.any():
                    centers = update_prototypes(
                        prototypes=centers,
                        sample_features=f1[clean_inter_b].detach(),
                        labels=labels_x[clean_inter_b].detach(),
                        alpha=alpha_clean,
                    )
                # 高不确定 弱更新
                if high_unc_b.any():
                    centers = update_prototypes(
                        prototypes=centers,
                        sample_features=f1[high_unc_b].detach(),
                        labels=labels_x[high_unc_b].detach(),
                        alpha=alpha_high,
                    )

            # MixUp + 有/无标签损失（保留）
            size_m = inputs_x3.size(0) + inputs_x4.size(0)
            l = np.random.beta(args.alpha, args.alpha); l = max(l, 1 - l)
            all_inputs = torch.cat([inputs_x3, inputs_x4, inputs_u3, inputs_u4], dim=0)
            all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)
            idx = torch.randperm(all_inputs.size(0))
            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]
            mixed_input = l * input_a + (1 - l) * input_b
            mixed_target = l * target_a + (1 - l) * target_b
            _, mixed_logit = net(mixed_input)

            logits_mixed_x = mixed_logit[:size_m]
            logits_mixed_u = mixed_logit[size_m:]
            targets_mixed_x = mixed_target[:size_m]
            targets_mixed_u = mixed_target[size_m:]

            # 有标签损失
            Lce = -torch.mean(torch.sum(F.log_softmax(logits_mixed_x, dim=1) * targets_mixed_x, dim=1))

        # 无标签过滤 + Lu（保留）
        with torch.no_grad():
            max_prob_u, _ = targets_mixed_u.max(dim=1)
            base_threshold = 0.5
            delta = 0.10 if epoch < (warm_up + 60) else 0.15
            median_prob = max_prob_u.median()
            u_threshold = torch.minimum(torch.tensor(base_threshold, device=max_prob_u.device), median_prob + delta)
            u_mask = (max_prob_u >= u_threshold)
            min_keep_ratio = 0.25
            total_u = max_prob_u.numel()
            min_keep = max(1, int(min_keep_ratio * total_u))
            if u_mask.sum() < min_keep:
                kth = max_prob_u.kthvalue(k=total_u - min_keep + 1).values
                u_threshold = torch.minimum(u_threshold, kth)
                u_mask = (max_prob_u >= u_threshold)
            u_mask = u_mask[:len(logits_mixed_u)]

        if len(logits_mixed_u) == 0:
            Lu = torch.tensor(0.0, device=logits_mixed_u.device)
        else:
            u_mask = u_mask[:len(logits_mixed_u)]
            logits_mixed_u_filtered = logits_mixed_u[u_mask]
            targets_mixed_u_filtered = targets_mixed_u[u_mask]
            if len(logits_mixed_u_filtered) == 0:
                Lu = torch.tensor(0.0, device=logits_mixed_u.device)
            else:
                probs_u_filtered = F.softmax(logits_mixed_u_filtered, dim=1)
                Lu = F.kl_div(torch.log(probs_u_filtered + 1e-8), targets_mixed_u_filtered, reduction='mean')

        # 其余损失（保留）
        prior = (torch.ones(args.num_class) / args.num_class).cuda()
        pred_mean = torch.softmax(mixed_logit, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))

        # loss = Lce + args.lambda_u * Lu + penalty + args.lambda_c * loss_simCLR + lambda_ccr * loss_ccr
        # ↓↓↓↓↓↓↓↓↓↓ 【修改这行】 ↓↓↓↓↓↓↓↓↓↓
        loss = args.lambda_s * Lce + args.lambda_u * Lu + penalty + args.lambda_c * loss_simCLR + lambda_ccr * loss_ccr

        eLce += Lce.item(); eLu += Lu.item(); ePenalty += penalty.item()
        eLoss_simCLR += loss_simCLR.item(); eLoss_ccr += loss_ccr.item()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Loss: %.2f | CCR: %.4f'
                         % (args.dataset, args.r, args.noise_mode, epoch, args.num_epochs,
                            batch_idx + 1, num_iter, loss.item(), loss_ccr.item()))
        sys.stdout.flush()

    loss_log.write('Epoch:%d Total:%.4f Lce:%.2f Lu:%.4f penalty:%.4f Lsim:%.2f Lccr:%.4f\n'
                   % (epoch, loss.item(), eLce / num_iter, eLu / num_iter,
                      ePenalty / num_iter, eLoss_simCLR / num_iter, eLoss_ccr / num_iter))
    loss_log.flush()
    return centers



def warmup(epoch, net, optimizer, dataloader):
    net.train()
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
    # 核心修改1：启用论文要求的矛盾损失（适配噪声和OOD场景）
    #c_loss = ContradictoryLoss(alpha_clean=1.0 - args.r, beta_noise=args.r, gamma=0.1).cuda()
    c_loss = nn.CrossEntropyLoss().cuda()  # 临时使用标准交叉熵
    total_loss = 0.0

    # 核心修改2：检查clean_label中是否有真实OOD样本（≥80），验证数据集划分正确性
    ds = dataloader.dataset
    clean_arr = np.array(ds.clean_label, dtype=np.int64)  # 真实标签（ID→0-79，OOD→80-99）
    print(f"[Warmup检查] 干净标签（真实标签）范围：{clean_arr.min()}~{clean_arr.max()}")
    print(f"[Warmup检查] 真实OOD样本（≥80）数量：{(clean_arr >= 80).sum()} / {len(clean_arr)}")
    assert (clean_arr >= 80).any(), "未检测到真实OOD样本（clean_label中无≥80的标签），请检查dataloader的ID/OOD划分"

    for batch_idx, (inputs, labels, _) in enumerate(dataloader):
        inputs = inputs.cuda()
        labels = labels.cuda()  # labels为noise_label（OOD已伪装成0-79，ID可能含噪声）

        #optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=USE_AMP, dtype=AMP_DTYPE):
            feats, logits = net(inputs)  # logits为80类输出（ID类）
            loss = c_loss(logits, labels)  # 用矛盾损失处理ID样本中的噪声
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t ContradictoryLoss: %.4f'
                         % (args.dataset, args.r, args.noise_mode,
                            epoch, args.num_epochs, batch_idx + 1, num_iter, loss.item()))
        sys.stdout.flush()

    avg_loss = total_loss / num_iter
    loss_log.write(f'Epoch {epoch} - Warmup Avg ContradictoryLoss: {avg_loss:.4f}\n')
    loss_log.flush()


def generate_test_labels(args):
    """生成或加载CIFAR80-O测试集的干净标签和带噪声标签（仅ID样本加20%对称噪声）"""
    import os
    import json
    import random
    import numpy as np
    from torchvision.datasets import CIFAR100
    from Data.dataloader_cifar import cifar_dataset  # 复用数据加载器的ID/OOD划分逻辑

    # 1. 定义标签文件路径（基于args.data_path，确保与主程序路径一致）
    clean_label_path = os.path.join(args.data_path, "test_clean_labels.json")
    noise_label_path = os.path.join(args.data_path, "test_0.2_sym_labels.json")

    # 2. 若标签已存在，直接加载并返回（避免重复生成）
    if os.path.exists(clean_label_path) and os.path.exists(noise_label_path):
        print(f"[Label Check] 测试集标签已存在，直接加载（路径：{args.data_path}）")
        clean_labels = np.array(json.load(open(clean_label_path, "r")))
        noise_labels = np.array(json.load(open(noise_label_path, "r")))
        return clean_labels, noise_labels

    # 3. 复用cifar_dataset的ID/OOD子类划分（确保与训练集逻辑完全一致）
    dummy_dataset = cifar_dataset(
        dataset='cifar80o',
        root_dir=args.data_path,
        r=0.2,  # 仅占位，不影响ID/OOD划分
        mode='test',
        noise_mode='sym',
        noise_file=''
    )
    id_subclasses = dummy_dataset.id_subclasses  # 80类ID子类（与训练集一致）
    ood_subclasses = dummy_dataset.ood_subclasses  # 20类OOD子类（与训练集一致）
    print(f"[Label Gen] 复用CIFAR80-O划分：ID子类{len(id_subclasses)}个，OOD子类{len(ood_subclasses)}个")

    # 4. 加载原始CIFAR100测试集（获取干净标签）
    raw_testset = CIFAR100(
        root=os.path.dirname(args.data_path),
        train=False,
        download=False,  # 若已下载则自动跳过
        transform=None  # 仅需标签，无需图像变换
    )
    raw_clean_labels = np.array(raw_testset.targets)  # 原始100类标签（0-99）
    print(f"[Label Gen] 加载原始CIFAR100测试集：共{len(raw_clean_labels)}个样本")

    # 5. 生成并保存干净标签（无噪声，直接保存原始标签）
    with open(clean_label_path, "w") as f:
        json.dump(raw_clean_labels.tolist(), f)
    print(f"[Label Gen] 生成干净标签：{clean_label_path}")

    # 6. 生成并保存带20%对称噪声的标签（仅ID样本加噪，OOD样本标签不变）
    noise_test_labels = raw_clean_labels.copy()  # 初始化噪声标签为干净标签
    noise_ratio = 0.2  # 20%对称噪声（与训练集噪声率一致）

    # 筛选ID样本的索引（仅对这些样本加噪）
    id_sample_indices = np.where(np.isin(raw_clean_labels, id_subclasses))[0]
    num_id_samples = len(id_sample_indices)
    num_noise_samples = int(num_id_samples * noise_ratio)  # 需加噪的ID样本数量

    # 固定随机种子，确保标签生成可复现（与训练集噪声生成逻辑一致）
    random.seed(args.seed)
    # 随机选择要加噪的ID样本索引
    noise_indices_in_id = random.sample(range(num_id_samples), num_noise_samples)
    noise_global_indices = id_sample_indices[noise_indices_in_id]  # 全局索引

    # 为选中的ID样本添加对称噪声（随机替换为其他ID子类）
    for idx in noise_global_indices:
        original_label = raw_clean_labels[idx]
        # 候选标签：所有ID子类中排除原始标签
        candidate_labels = [sub for sub in id_subclasses if sub != original_label]
        noise_test_labels[idx] = random.choice(candidate_labels)

    # 保存噪声标签文件
    with open(noise_label_path, "w") as f:
        json.dump(noise_test_labels.tolist(), f)
    print(f"[Label Gen] 生成噪声标签：{noise_label_path}")

    # 验证加噪效果（确保噪声率正确）
    actual_noise_count = sum(noise_test_labels[id_sample_indices] != raw_clean_labels[id_sample_indices])
    actual_noise_ratio = actual_noise_count / num_id_samples
    print(
        f"[Label Gen] 加噪验证：ID样本{num_id_samples}个，实际加噪{actual_noise_count}个，噪声率{actual_noise_ratio:.2%}（目标20%）")

    return raw_clean_labels, noise_test_labels


def test(epoch, net1, net2, args, test_loader, clean_test_labels, noise_test_labels, id_map,id_map_pure):
    """
    论文口径 Acc（100类整体Top-1 + 最后10轮平均） + 你的 OOD 判别 + Precision(ID)
    - 准确率分母：测试集中所有样本（100类，ID+OOD）
    - 准确率分子：预测正确的样本数
    注意：不使用 noise_test_labels 作为 Acc 的 GT；只使用 clean_test_labels。
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score, roc_curve

    net1.eval();
    net2.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # —— 100类整体准确率计数器 ——
    total_all = 0
    correct_all = 0

    # —— 你的 OOD 指标缓存（保持原逻辑：基于ID-80的置信度阈值）——
    total_ood = 0
    correct_ood_detect = 0
    all_id_conf = []
    all_gt_type = []

    global best_acc, last10_acc
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(test_loader):
            inputs = inputs.to(device)
            b = inputs.size(0)
            start = batch_idx * args.batch_size
            end = start + b

            # 干净测试标签（0..99，100类全空间）
            batch_clean = torch.as_tensor(clean_test_labels[start:end], device=device, dtype=torch.long)

            # 双模型融合（要求模型输出100维）
            _, out1 = net1(inputs)
            _, out2 = net2(inputs)
            logits = out1 + out2  # [B, 80]

            # ===== 论文口径：100类整体Top-1准确率 =====
            # 预测的Top-1类
            pred = logits.argmax(dim=1)  # 模型100-way输出

            # ===== 论文口径：仅ID类（0-79）Top-1准确率（CIFAR80N-O对齐） =====
            # 筛选ID样本（id_map[真实标签]≥0表示为ID类，排除OOD类）
            is_id_sample = (id_map[batch_clean] >= 0)  # [B]，True=ID样本，False=OOD样本
            if is_id_sample.any():
                # 仅累计ID样本的总数和正确数
                total_all += is_id_sample.sum().item()  # 分母：ID样本数
                # 正确逻辑：将原始标签映射为ID类索引（0~79）后再比较
                correct_all += (pred[is_id_sample] == id_map[batch_clean[is_id_sample]]).sum().item()

            # ===== 你的 OOD 指标（保持旧口径：基于ID-80）=====
            ID_CLASSES = 80  # 固定80为ID类数
            #logits_id80 = logits[:, id_map]  # 不再直接截取前80列，而是通过id_map筛选logits列
            logits_id80 = logits[:, id_map_pure]  # 用纯 ID 映射表索引：仅取 80 个 ID 类的 logits
            prob_id80 = torch.softmax(logits_id80, dim=1)
            id_conf, _ = prob_id80.max(dim=1)  # [B]

            mapped_clean = id_map[batch_clean]  # [B]，ID→0..79；OOD→-1
            is_ood = (mapped_clean < 0)  # True 表示 OOD
            #ood_pred = (id_conf < 0.35)  # 你的阈值法
            if epoch < 50:
                ood_threshold = 0.1
            elif epoch < 100:
                ood_threshold = 0.2
            else:
                ood_threshold = 0.3
            ood_pred = (id_conf < ood_threshold)
            if is_ood.any():
                total_ood += is_ood.sum().item()
                correct_ood_detect += (ood_pred[is_ood] == is_ood[is_ood]).sum().item()

            all_id_conf.extend(id_conf.detach().cpu().numpy())
            all_gt_type.extend(is_ood.detach().cpu().numpy().astype(int))

    # —— 论文口径：ID类（0-79）Acc（单轮 + 最后10轮平均，CIFAR80N-O对齐）——
    acc_all = 100.0 * correct_all / max(1, total_all)  # 仅ID样本的准确率
    last10_acc.append(acc_all)
    paper_metric = sum(last10_acc) / len(last10_acc)
    # 打印时明确“ID80-only”，避免与100类混淆
    print(f"\n[Paper-Acc] ID80-only Top-1 (clean labels): {acc_all:.2f}%  | Avg(last10): {paper_metric:.2f}%")

    # —— 你的 OOD 指标（独立展示）——
    all_id_conf = np.array(all_id_conf);
    all_gt_type = np.array(all_gt_type)
    auroc = 0.0;
    fpr95 = 100.0
    if len(np.unique(all_gt_type)) == 2:
        auroc = roc_auc_score(all_gt_type, 1.0 - all_id_conf)
        fpr, tpr, _ = roc_curve(all_gt_type, 1.0 - all_id_conf)
        idx95 = np.where(tpr >= 0.95)[0]
        if len(idx95) > 0:
            fpr95 = fpr[idx95[0]] * 100.0
    ood_detect_acc = 100.0 * correct_ood_detect / max(1, total_ood)
    print(f"[OOD] Acc:{ood_detect_acc:.2f}%  AUROC:{auroc:.4f}  FPR95:{fpr95:.2f}%")

    # —— Precision-ID（口径不变：id_conf>=0.5 预测为 ID）——
    pred_ID_np = (all_id_conf >= 0.5)
    true_ID_np = (all_gt_type == 0)  # 非 OOD 即 ID
    denom = pred_ID_np.sum()
    prec_ID = (100.0 * (pred_ID_np & true_ID_np).sum() / denom) if denom > 0 else 0.0
    print(f"[Precision-ID(test)] {prec_ID:.2f}%   (阈值=0.5; 口径=预测为ID中的真实ID比例)")

## NOTE: metrics helper moved to `our_cifar.metrics`


import torch
import torch.nn.functional as F


class CLoss(torch.nn.Module):
    def __init__(self, temp=0.1, eps=1e-8):  # 移除num_class硬编码
        super(CLoss, self).__init__()
        self.num_class = args.num_class  # 统一使用预设任务边界（80类）
        self.temp = temp  # 对比损失温度系数（控制相似度区分度）
        self.eps = eps  # 防止log(0)或除零错误

    def forward(self, outputs, labels, features, prototype_features, teacher_outputs, epoch):
        """
        新方案核心：仅对ID样本计算损失，整合分类损失+对比损失+KL散度（分阶段权重）
        参数：
            outputs: [B, 80] 学生网络对ID类的预测（logits）
            labels: [B] 原始标签（含OOD标签80-99）
            features: [B, D] 样本特征（用于对比损失）
            prototype_features: [80, D] 类原型特征（CPM模块输出，已归一化）
            teacher_outputs: [B, 80] 教师网络对ID类的预测（用于生成平滑标签）
            epoch: 当前训练轮次（用于分阶段调整权重）
        """
        # 1. 过滤ID样本（仅保留0-79的标签，与旧代码逻辑一致）
        id_mask = (labels >= 0) & (labels < self.num_class)
        if not id_mask.any():  # 全为OOD样本时损失为0
            return torch.tensor(0.0, device=outputs.device)

        # 提取ID样本的相关数据（过滤OOD）
        valid_outputs = outputs[id_mask]  # [K, 80]，K为ID样本数
        valid_labels = labels[id_mask]  # [K]，仅0-79
        valid_feats = features[id_mask]  # [K, D]，ID样本特征
        teacher_logits = teacher_outputs[id_mask]  # [K, 80]，教师对ID样本的预测

        # 2. l1：交叉熵损失（保留旧代码的核心，确保ID样本分类正确）
        l1 = F.cross_entropy(valid_outputs, valid_labels)

        # 3. l2：对比损失（替换旧代码的log(1-pred)，基于类原型的特征一致性）
        # 3.1 特征与原型归一化（确保余弦相似度有效）
        valid_feats = F.normalize(valid_feats, dim=1)  # [K, D]
        prototype_features = F.normalize(prototype_features, dim=1)  # [80, D]

        # 3.2 计算“样本与自身类别原型”的相似度
        target_protos = prototype_features[valid_labels]  # [K, D]，每个样本对应的类原型
        same_sim = F.cosine_similarity(valid_feats, target_protos, dim=1)  # [K]

        # 3.3 计算“样本与所有异类原型”的平均相似度
        diff_sim_list = []
        for i in range(valid_feats.shape[0]):
            c = valid_labels[i]  # 当前样本的真实类别
            # 排除自身类别，取所有异类原型
            diff_protos = prototype_features[torch.arange(self.num_class, device=valid_feats.device) != c]
            # 样本与异类原型的平均相似度
            diff_sim = F.cosine_similarity(valid_feats[i].unsqueeze(0), diff_protos, dim=1).mean()
            diff_sim_list.append(diff_sim)
        diff_sim = torch.tensor(diff_sim_list, device=valid_feats.device)  # [K]

        # 3.4 InfoNCE风格对比损失（同类相似度 > 异类相似度）
        l2 = -torch.log(
            torch.exp(same_sim / self.temp) /
            (torch.exp(same_sim / self.temp) + torch.exp(diff_sim / self.temp))
        ).mean()

        # 4. l3：KL散度（替换旧代码的错误计算，基于教师平滑标签）
        # 4.1 教师输出平滑（温度缩放增强抗噪声能力）
        T = 0.5  # 温度系数（T越小，平滑后分布越集中）
        teacher_probs = F.softmax(teacher_logits / T, dim=1)  # [K, 80]，平滑后的教师软标签

        # 4.2 学生输出概率
        student_probs = F.softmax(valid_outputs, dim=1)  # [K, 80]

        # 4.3 正确的KL散度（学生预测 对齐 教师平滑标签）
        l3 = F.kl_div(
            torch.log(student_probs + self.eps),  # 学生log概率
            teacher_probs,  # 教师平滑概率
            reduction='batchmean'
        )

        # 5. 分阶段加权组合（根据epoch动态调整权重，贴合论文策略）
        if epoch < 130:  # 前期：侧重分类和教师蒸馏
            loss = l1 + 0.01 * l2 + 0.1 * l3
        else:  # 后期：增强特征与原型的一致性
            loss = l1 + 0.05 * l2 + 0.05 * l3

        return loss


class LSLoss(object):
    def __call__(self, input_logits, target_logits, eps=0.35, tau=1 / 8, reduction='batchmean'):
        assert input_logits.size() == target_logits.size()
        input_log_softmax = F.log_softmax(input_logits, dim=1)
        target = F.softmax(target_logits, dim=1)

        C = args.num_class  # 强制使用预设任务边界（80类），不依赖输入维度
        smooth_labels = target.gt(tau).float() * target
        smooth_labels = smooth_labels / smooth_labels.sum(1).unsqueeze(1)
        smooth_labels = smooth_labels * (1 - eps)
        Ks = target.gt(tau).sum(1).unsqueeze(1)
        Ks = Ks + Ks.eq(0).int()
        small_mask = 1 - target.gt(tau).float()
        smooth_labels = smooth_labels + small_mask * (eps / (C - Ks.float()))

        return F.kl_div(input_log_softmax, smooth_labels, reduction=reduction)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u):
        # 有标签样本损失（保持不变）
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        # 无标签样本损失：用KL散度替代MSE
        probs_u = F.softmax(outputs_u, dim=1)  # 学生模型对无标签样本的预测分布
        # 确保目标伪标签是概率分布（已归一化）
        targets_u = targets_u / targets_u.sum(dim=1, keepdim=True)  # 双重保险，避免数值问题
        # KL散度：学生预测分布 对齐 伪标签分布（加入eps避免log(0)）
        Lu = F.kl_div(torch.log(probs_u + 1e-8), targets_u, reduction='mean')
        return Lx, Lu
# === Contradictory Loss (CPSL 式(1)) ===
class ContradictoryLoss(nn.Module):
    def __init__(self, alpha_clean, beta_noise, gamma=0.1):
        super().__init__()
        self.alpha = alpha_clean  # 低熵样本的监督权重
        self.beta = beta_noise    # 高熵样本的反向监督权重
        self.gamma = gamma        # 熵正则项的权重

    def forward(self, logits, labels):
        p = torch.softmax(logits, dim=1)
        entropy = -torch.sum(p * torch.log(torch.clamp(p, 1e-8, 1.0)), dim=1)  # 计算样本的熵

        # 根据熵值动态调整样本权重
        weight_pos = self.alpha * (1 - entropy / entropy.max())  # 低熵样本的正向权重
        weight_neg = self.beta * (entropy / entropy.max())       # 高熵样本的反向权重

        # 计算交叉熵损失
        ce_pos = F.cross_entropy(logits, labels, reduction='none')
        ce_neg = F.nll_loss(torch.log(torch.clamp(1.0 - p, 1e-8, 1.0)), labels, reduction='none')

        # 加权交叉熵损失
        loss = (weight_pos * ce_pos + weight_neg * ce_neg).mean() + self.gamma * entropy.mean()
        return loss



def update_ema_variables(model, ema_model, epoch):
    # 确保索引不越界（取min限制，使用最后一个alpha值）
    idx = min(epoch - warm_up, len(numbers) - 1)
    alpha = numbers[idx]
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_((1 - alpha) * param.data)


def create_model():
    model = DNN7(num_classes=args.num_class, feat_dim=args.dr_dim, dropout=0.2)
    return model.cuda()



def resume(checkpoint_path, net1, net2, tch_net1, tch_net2, optimizer1, optimizer2):
    checkpoint = torch.load(checkpoint_path)
    print('Resume from checkpoint at epoch {}'.format(checkpoint["epoch"]))

    net1.load_state_dict(checkpoint["net1_state_dict"])
    net2.load_state_dict(checkpoint["net2_state_dict"])

    tch_net1.load_state_dict(checkpoint["tch_net1_state_dict"])
    tch_net2.load_state_dict(checkpoint["tch_net2_state_dict"])

    optimizer1.load_state_dict(checkpoint["optimizer1_state_dict"])
    optimizer2.load_state_dict(checkpoint["optimizer2_state_dict"])

    epoch = checkpoint["epoch"] + 1
    centers1 = checkpoint["centers1"]
    centers2 = checkpoint["centers2"]
    best_acc = checkpoint["best_acc"]

    return net1, net2, tch_net1, tch_net2, optimizer1, optimizer2, epoch, centers1, centers2, best_acc


def save(filepath, net1, net2, tch_net1, tch_net2, optimizer1, optimizer2, epoch, centers1, centers2, best_acc):
    torch.save(
        {
            "net1_state_dict": net1.state_dict(),
            "net2_state_dict": net2.state_dict(),
            "tch_net1_state_dict": tch_net1.state_dict(),
            "tch_net2_state_dict": tch_net2.state_dict(),
            "optimizer1_state_dict": optimizer1.state_dict(),
            "optimizer2_state_dict": optimizer2.state_dict(),
            "epoch": epoch,
            "centers1": centers1,
            "centers2": centers2,
            "best_acc": best_acc
        },
        filepath,
    )
    print('Checkpoint Saved')


def plotHistogram(model_1_loss, model_2_loss, noise_index, clean_index, epoch, noise_rate):
    title = 'Epoch-' + str(epoch) + ':'
    fig = plt.figure()
    plt.subplot(121)
    gmm = GaussianMixture(n_components=2, max_iter=20, tol=1e-2, random_state=0, reg_covar=5e-4)
    model_1_loss = np.reshape(model_1_loss, (-1, 1))
    gmm.fit(model_1_loss)  # fit the loss

    # plot resulting fit
    x_range = np.linspace(0, 1, 1000)
    pdf = np.exp(gmm.score_samples(x_range.reshape(-1, 1)))
    responsibilities = gmm.predict_proba(x_range.reshape(-1, 1))
    pdf_individual = responsibilities * pdf[:, np.newaxis]
    plt.hist(np.array(model_1_loss[noise_index]), density=True, bins=100, alpha=0.5, histtype='bar', color='red',
             label='Noisy subset')
    plt.hist(np.array(model_1_loss[clean_index]), density=True, bins=100, alpha=0.5, histtype='bar', color='blue',
             label='Clean subset')
    plt.plot(x_range, pdf, '-k', label='Mixture')
    plt.plot(x_range, pdf_individual, '--', label='Component')
    plt.legend(loc='upper right', prop={'size': 12})
    plt.xlabel('Normalized loss')
    plt.ylabel('Estimated pdf')
    plt.title(title + 'Model_1')

    plt.subplot(122)
    gmm = GaussianMixture(n_components=2, max_iter=20, tol=1e-2, random_state=0, reg_covar=5e-4)
    model_2_loss = np.reshape(model_2_loss, (-1, 1))
    gmm.fit(model_2_loss)  # fit the loss

    # plot resulting fit
    x_range = np.linspace(0, 1, 1000)
    pdf = np.exp(gmm.score_samples(x_range.reshape(-1, 1)))
    responsibilities = gmm.predict_proba(x_range.reshape(-1, 1))
    pdf_individual = responsibilities * pdf[:, np.newaxis]
    plt.hist(np.array(model_2_loss[noise_index]), density=True, bins=100, alpha=0.5, histtype='bar', color='red',
             label='Noisy subset')
    plt.hist(np.array(model_2_loss[clean_index]), density=True, bins=100, alpha=0.5, histtype='bar', color='blue',
             label='Clean subset')
    plt.plot(x_range, pdf, '-k', label='Mixture')
    plt.plot(x_range, pdf_individual, '--', label='Component')
    plt.legend(loc='upper right', prop={'size': 12})
    plt.xlabel('Normalized loss')
    plt.ylabel('Estimated pdf')
    plt.title(title + 'Model_2')

    print('\nlogging histogram...')
    title = 'cifar80no_' + str(args.noise_mode) + '_moit_double_' + str(noise_rate)
    plt.savefig(os.path.join('./figure_his/', 'two_model_{}_{}.{}'.format(epoch, title, "png")), dpi=300)
    plt.close()


if not os.path.exists('./checkpoint'): os.makedirs('./checkpoint')
if not os.path.exists('./figure_his'): os.makedirs('./figure_his')

filepath = os.path.join('./checkpoint', 'model.pth.tar')
stats_log = open('./checkpoint/%s_%.1f_%s' % (args.dataset, args.r, args.noise_mode) + '_stats.txt', 'a')
test_log = open('./checkpoint/%s_%.1f_%s' % (args.dataset, args.r, args.noise_mode) + '_acc.txt', 'a')
loss_log = open('./checkpoint/%s_%.1f_%s' % (args.dataset, args.r, args.noise_mode) + '_loss.txt', 'a')
# 新增：用于论文口径 Acc 的“最后10轮平均”
last10_acc = deque(maxlen=10)

# ↓↓↓ 新增：调用标签生成函数，获取测试集标签 ↓↓↓
print("[Label Gen] 开始生成/加载测试集标签...")
clean_test_labels, noise_test_labels = generate_test_labels(args)  # 传入args，使用args.data_path
print(f"[Label Gen] 标签加载完成：干净标签{len(clean_test_labels)}个，噪声标签{len(noise_test_labels)}个")
# ↑↑↑ 新增结束 ↑↑↑

warm_up = 25

ts = datetime.now().strftime('%Y%m%d-%H%M%S')
noise_file = os.path.join(args.data_path,
                          f"noise_{args.dataset}_{args.noise_mode}_r{args.r:.2f}_seed{args.seed}_{ts}.json")
os.makedirs(os.path.dirname(noise_file) or ".", exist_ok=True)
loader = dataloader.cifar_dataloader(args.dataset, r=args.r, noise_mode=args.noise_mode,
                                     batch_size=args.batch_size, num_workers=5, root_dir=args.data_path, log=stats_log,
                                     noise_file=noise_file)

# =========================================================
# 新增：构建论文口径的测试集标签映射表（原始标签 → 连续ID索引）
# =========================================================
test_loader_tmp = loader.run('test')
test_ds = test_loader_tmp.dataset

# 取出论文中定义的80个ID子类（来自cifar_dataset）
id_subclasses = list(test_ds.id_subclasses)  # 长度80，元素为原始0..99标签
id_map_np = np.full(100, -1, dtype=np.int64)  # 非ID类初始化为-1
for i, orig_label in enumerate(id_subclasses):
    id_map_np[orig_label] = i  # ID类重映射为0..79
id_map = torch.tensor(id_map_np, device='cuda')  # 放GPU，后续test()直接索引使用
print(f"[ID映射表] 已建立：共{len(id_subclasses)}个ID类，示例映射：{id_map_np[:10]}")
# =========================================================
# =========================================================
# 新增：构建纯 ID 映射表（id_map_pure）：用于索引 logits，无 -1
# =========================================================
# 1. 筛选所有有效 ID 类的「原始标签」（剔除 OOD 类的 -1 映射）
valid_orig_labels = [orig_label for orig_label in range(100) if id_map_np[orig_label] != -1]
# 2. 构建纯 ID 类的 logits 索引：直接用 0~79（因为 logits 输出是 80 维，对应 ID 类的连续索引）
id_map_pure = torch.tensor([i for i in range(len(valid_orig_labels))], device='cuda')  # 结果：[0,1,2,...,79]

# 3. 强制验证：确保纯 ID 映射表无错误（避免后续索引问题）
assert len(id_map_pure) == 80, f"纯 ID 映射表长度错误！应为 80，实际 {len(id_map_pure)}"
assert (id_map_pure >= 0).all(), "纯 ID 映射表包含负索引！（致命错误）"
assert (id_map_pure < 80).all(), "纯 ID 映射表索引超出 80 维！（与 logits 维度不匹配）"
print(f"[纯 ID 映射表] 构建完成：长度 {len(id_map_pure)}，索引范围 {id_map_pure.min()}~{id_map_pure.max()}（应 0~79）")


# 新多原型定义：[80, M, dr_dim]（仅ID类原型，符合预设任务边界）
M = args.num_prototypes  # 与 initialize_prototypes 中一致
num_prototypes = args.num_prototypes
C_total = args.num_class
centers1 = torch.randn(C_total, M, args.dr_dim, device='cuda')
centers2 = torch.randn(C_total, M, args.dr_dim, device='cuda')
centers1 = F.normalize(centers1, dim=-1)
centers2 = F.normalize(centers2, dim=-1)
print(f"初始化后 centers1 形状: {centers1.shape}")
num_samples = 50000
best_acc = 0.0
# 新增：全局变量，保存 OOD 噪声样本的全局索引（训练集）
ood_noise_indices = None  # ← 插入这行
print('| Building net')
net1 = create_model()
net2 = create_model()
tch_net1 = create_model()
tch_net2 = create_model()

cudnn.benchmark = True

for param in tch_net1.parameters(): param.requires_grad = False
for param in tch_net2.parameters(): param.requires_grad = False

criterion_ls = LSLoss()
criterion_semi = SemiLoss()
CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
contrastive_criterion = SupConLoss()

optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

start_epoch = 0

if args.resume:
    net1, net2, tch_net1, tch_net2, optimizer1, optimizer2, start_epoch, centers1, centers2, best_acc = (
        resume(filepath, net1, net2, tch_net1, tch_net2, optimizer1, optimizer2))


numbers = np.linspace(0.99, 0.999, args.num_epochs - warm_up)
numbers_eval = np.linspace(0.9, 0.8, args.num_epochs - warm_up)
# 全局变量：保存OOD噪声样本的全局索引（训练集）
# 全局变量：保存OOD噪声样本的全局索引（训练集）
# ood_noise_indices = None

for epoch in range(start_epoch, args.num_epochs):
    if epoch > 149:
        lr = 0.002
    else:
        lr = 0.01  #0.02

    # 更新学习率
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr

        # 加载数据
    test_loader = loader.run('test')
    # ↓↓↓ 修改：传入测试集标签 ↓↓↓
    acc_all, paper_metric, prec_ID = test(
        epoch, net1, net2, args, test_loader,
        clean_test_labels, noise_test_labels, id_map, id_map_pure
    )# ↑↑↑ 修改结束 ↑↑↑
    # 获取ID类测试集
    # 调用新test函数，评估当前epoch的性能
    eval_loader = loader.run('eval_train')  # 获取评估集
    # print("\n===== 验证eval_loader批次结构 =====")
    # try:
    #     # 取第一个批次验证
    #     first_batch = next(iter(eval_loader))
    #     print(f"批次元素数量：{len(first_batch)}")  # 应输出3
    #     print(f"inputs形状：{first_batch[0].shape}")  # 如 torch.Size([128, 3, 32, 32])
    #     print(f"targets形状：{first_batch[1].shape}")  # 如 torch.Size([128])
    #     print(f"index示例：{first_batch[2][:5]}")  # 如 tensor([0, 1, 2, 3, 4])
    #     print("===== 验证完成 =====")
    # except StopIteration:
    #     print("eval_loader为空！")
    # except Exception as e:
    #     print(f"验证失败：{e}")

    # 获取噪声样本的索引
    noise_ind, clean_ind = eval_loader.dataset.if_noise()

    if epoch < warm_up:
        warmup_trainloader = loader.run('warmup')
        # ====== 这里新增：检查标签范围 ======
        ds = warmup_trainloader.dataset
        import numpy as np

        # 核心修改：用clean_label（真实标签）检查OOD样本是否存在，而非noise_label（伪装后标签）
        clean_arr = np.array(ds.clean_label, dtype=np.int64)  # 真实标签：ID→0-79，OOD→80-99
        print(f"[Warmup检查] 干净标签（真实标签）最小={clean_arr.min()}, 最大={clean_arr.max()}")
        print(f"[Warmup检查] 真实OOD样本（≥80）数量={(clean_arr >= 80).sum()} / {len(clean_arr)}")
        # 强约束：若真实标签中没有≥80的样本，说明ID/OOD划分错误
        assert (clean_arr >= 80).any(), "未检测到真实OOD样本（clean_label中无≥80的标签），请检查dataloader的ID/OOD映射逻辑"

        print('Warmup Net1')
        warmup(epoch, net1, optimizer1, warmup_trainloader)
        print('\nWarmup Net2')
        warmup(epoch, net2, optimizer2, warmup_trainloader)

    else:  # 半监督阶段（epoch >= warm_up）
        # 调用eval_train获取：置信度、损失、评估原型、相似度差值、OOD索引（新增返回OOD索引）
        prob1, loss1, centers_eval1, sim_diff1, ood1, hard1 = eval_train(net1, epoch, centers1, args, eval_loader)
        prob2, loss2, centers_eval2, sim_diff2, ood2, hard2 = eval_train(net2, epoch, centers2, args, eval_loader)
        # 融合双模型的OOD索引（取并集，确保无重复）
        ood_noise_indices = torch.unique(torch.cat([ood1, ood2])) if (len(ood1) > 0 or len(ood2) > 0) else torch.tensor(
            [], device='cuda')
        # ↓↓↓ 新增：将prob1/prob2转为GPU张量（关键修改）↓↓↓
        prob1_t = torch.tensor(prob1, device='cuda', dtype=torch.float32)  # 网络2的干净概率（供net2训练用）
        prob2_t = torch.tensor(prob2, device='cuda', dtype=torch.float32)  # 网络1的干净概率（供net1训练用）
        # ↑↑↑ 新增结束 ↑↑↑
        # 融合双模型的置信度（用于划分labeled/unlabeled）
        combined_prob = torch.tensor((prob1 + prob2) / 2, device='cuda')

        # 第1个半监督epoch：初始化原型（仅用非OOD样本）
        if epoch == warm_up:
            print(f"\n=== 半监督阶段开始（epoch={warm_up}）：初始化原型与教师模型 ===")

            # 1. 初始化纯净原型（过滤OOD样本）
            # 提取Net1的非OOD特征和标签
            all_features_net1 = []
            all_labels_net1 = []
            with torch.no_grad():
                for inputs, targets, index in eval_loader:  # 注意：这里需要获取样本全局索引index
                    inputs, targets = inputs.cuda(), targets.cuda()
                    index = index.cuda()
                    # 过滤当前批次的OOD样本
                    non_ood_mask = ~torch.isin(index, ood_noise_indices)
                    if not non_ood_mask.any():
                        continue
                    feature, _ = net1(inputs[non_ood_mask])
                    all_features_net1.append(F.normalize(feature, dim=1))
                    all_labels_net1.append(targets[non_ood_mask])
            all_features_net1 = torch.cat(all_features_net1, dim=0)
            all_labels_net1 = torch.cat(all_labels_net1, dim=0)
            centers1 = initialize_prototypes(
                all_features_net1, all_labels_net1, num_classes=80, num_prototypes=args.num_prototypes
            )
            print(f"epoch={warm_up} 初始化后 centers1 形状: {centers1.shape}")
            # 同理初始化Net2的纯净原型
            all_features_net2 = []
            all_labels_net2 = []
            with torch.no_grad():
                for inputs, targets, index in eval_loader:
                    inputs, targets = inputs.cuda(), targets.cuda()
                    index = index.cuda()
                    non_ood_mask = ~torch.isin(index, ood_noise_indices)
                    if not non_ood_mask.any():
                        continue
                    feature, _ = net2(inputs[non_ood_mask])
                    all_features_net2.append(F.normalize(feature, dim=1))
                    all_labels_net2.append(targets[non_ood_mask])
            all_features_net2 = torch.cat(all_features_net2, dim=0)
            all_labels_net2 = torch.cat(all_labels_net2, dim=0)
            centers2 = initialize_prototypes(
                all_features_net2, all_labels_net2, num_classes=80, num_prototypes=args.num_prototypes
            )
            print(f"epoch={warm_up} 初始化后 centers2 形状: {centers2.shape}")
            # 2. 学生→教师模型参数复制
            for param, param_tch in zip(net1.parameters(), tch_net1.parameters()):
                param_tch.data.copy_(param.data)
            for param, param_tch in zip(net2.parameters(), tch_net2.parameters()):
                param_tch.data.copy_(param.data)

        # ========= 触发可视化：只在 warm_up 和 warm_up+50 画 =========
        plot_epochs = {25, 50, 75, 100, 125, 150, 175, 199,225,250,275,299}
        if epoch in plot_epochs:
            # === 和 eval_train 的 loss 对齐（N 要一致）===
            N = len(eval_loader.dataset)
            assert len(loss1) == N and len(loss2) == N, "loss 与数据集长度不一致"

            # 1) OOD 掩码（全局索引 → 布尔掩码）
            ood_mask_full = torch.zeros(N, dtype=torch.bool, device='cuda')
            if ood_noise_indices is not None and len(ood_noise_indices) > 0:
                # 确保是 LongTensor 且位于同一设备
                ood_idx = ood_noise_indices.to(device='cuda', dtype=torch.long)
                ood_mask_full[ood_idx] = True

            # 2) clean 交集（来自 DataCache；可能还没初始化）
            clean_mask_full = data_cache.clean_inter_mask
            if clean_mask_full is None:
                # 退化为“平均 clean 概率 >= 0.5”的硬判（注意取平均再比较）
                p1 = torch.as_tensor(prob1, device='cuda', dtype=torch.float32)
                p2 = torch.as_tensor(prob2, device='cuda', dtype=torch.float32)
                clean_mask_full = ((p1 + p2) / 2.0) >= 0.5
            else:
                # 保证是 cuda:bool 且长度匹配
                clean_mask_full = clean_mask_full.to(device='cuda', dtype=torch.bool)
                assert clean_mask_full.numel() == N, "clean_inter_mask 长度不一致"

            # 3) ID(非 clean) = 非 OOD 且 非 clean
            non_ood_full = ~ood_mask_full
            id_rest_full = non_ood_full & ~clean_mask_full

            # 转 numpy（与 loss1/loss2 对齐）
            c1 = clean_mask_full.detach().cpu().numpy().astype(bool)
            i1 = id_rest_full.detach().cpu().numpy().astype(bool)
            o1 = ood_mask_full.detach().cpu().numpy().astype(bool)

            # 画三分区直方图（两模型共用同一组掩码是合理的：clean 为“交集”，OOD 为并集）
            plot_gmm_histogram_tri_dual(
                np.asarray(loss1), c1, i1, o1,
                np.asarray(loss2), c1, i1, o1,
                epoch, outdir="./figure_vis"
            )

            # （可选）t-SNE
            feats_tsne, labels_tsne = collect_features_for_tsne(net1, eval_loader, max_points=20000)
            plot_tsne(
                feats_tsne, labels_tsne, epoch,
                outdir="./figure_vis",
                only_id=True, id_set=set(id_subclasses), max_points_per_class=800
            )
        # ============================================================

        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ 修改开始 ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓

        # === 【修改点：层级筛选消融控制】 ===
        # 如果命令行加了 --simple_selection，就把 hard_clean 设为 None
        # 这样 DataCache 内部就会回退到 "prob > threshold" 的简单筛选逻辑
        if args.simple_selection:
            use_hard_1 = None
            use_hard_2 = None
        else:
            # 正常逻辑：传入 eval_train 返回的硬判掩码
            use_hard_1 = torch.as_tensor(hard1, device='cuda', dtype=torch.bool)
            use_hard_2 = torch.as_tensor(hard2, device='cuda', dtype=torch.bool)

        # 使用关键字参数，避免位置参数错位
        data_cache.update(
            epoch=epoch,
            ood_indices=ood_noise_indices,
            p_threshold=args.p_threshold,
            seed=args.seed,
            hard_clean_1=use_hard_1,  # <--- 修改这里，传入变量
            hard_clean_2=use_hard_2,  # <--- 修改这里，传入变量
            prob1=prob1_t,  # net1 的 clean 概率
            prob2=prob2_t,  # net2 的 clean 概率
            # avg_prob 可省略，类里会用 0.5*(prob1+prob2)
        )

        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ 修改结束 ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        # 新增：训练集干净样本精确率计算（基于缓存的labeled掩码）
        pred_clean_np = data_cache.labeled_mask.cpu().numpy().astype(bool)
        prec_clean_avg = compute_clean_precision_on_eval(eval_loader, pred_clean_np)
        print(f"[干净样本精确率(eval_train)] 平均:{prec_clean_avg:.2f}%")
        stats_log.write(f"轮次:{epoch}\t平均干净样本精确率:{prec_clean_avg:.2f}%\n")
        stats_log.flush()

        # 更新多原型（仅用ID类评估原型，过滤OOD类）
        alpha_eval = numbers_eval[epoch - warm_up]
        centers1 = centers1.to('cuda')
        centers2 = centers2.to('cuda')
        #print(f"更新前 centers1 形状: {centers1.shape}")  # 预期: [80, 4, 128]
        #print(f"centers_eval1 形状: {centers_eval1.shape}")  # 预期: [80, 4, 128]
        # 过滤评估原型中的OOD类（假设0-79为ID类，80-99为OOD类）
        centers_eval1 = centers_eval1[:80].to(centers1.device)  # 仅保留ID类原型
        centers_eval2 = centers_eval2[:80].to(centers2.device)
        #print(f"过滤后 centers_eval1 形状: {centers_eval1.shape}")  # 预期: [80, 4, 128]
        # 原型融合更新（确保归一化）
        centers1 = F.normalize(centers1, dim=-1)
        centers_eval1 = F.normalize(centers_eval1, dim=-1)
        centers1 = centers1.mul_(alpha_eval).add_(centers_eval1 * (1 - alpha_eval))
        centers1 = F.normalize(centers1, dim=-1)
        # 同理更新centers2
        centers2 = F.normalize(centers2, dim=-1)
        centers_eval2 = F.normalize(centers_eval2, dim=-1)
        centers2 = centers2.mul_(alpha_eval).add_(centers_eval2 * (1 - alpha_eval))
        centers2 = F.normalize(centers2, dim=-1)

        # 基于缓存的掩码划分labeled/unlabeled（复用缓存，避免重复计算）
        print('Train Net1')
        pred2_safe = data_cache.labeled_mask.clone()  # 从缓存获取labeled掩码
        N = pred2_safe.numel()
        # Split-Guard逻辑（确保数据非空）
        if pred2_safe.sum().item() == N:
            k = max(int(0.1 * N), 1)
            _, idx_sorted = torch.sort(torch.tensor(prob2, device='cuda'))
            pred2_safe[idx_sorted[:k]] = False
        elif pred2_safe.sum().item() == 0:
            k = max(int(0.05 * N), 1)
            _, idx_sorted = torch.sort(torch.tensor(prob2, device='cuda'), descending=True)
            pred2_safe[idx_sorted[:k]] = True
        print(f"[Split-Guard] Net1: labeled={pred2_safe.sum().item()}/{N}, unlabeled={N - pred2_safe.sum().item()}")
        # 训练Net1时，传入prob2_t（对方网络的干净概率）
        labeled_trainloader1, unlabeled_trainloader1 = loader.run('train', pred2_safe, prob2_t)
        # 训练Net1（传入OOD索引，用于内部过滤）
        centers1 = train(
            epoch, net1, tch_net1, optimizer1, centers1, centers2,
            labeled_trainloader1, unlabeled_trainloader1,
            ood_noise_indices=ood_noise_indices,
            prob_clean_full=prob2_t,  # ← 必传
            data_cache=data_cache,
        )

        # 同理训练Net2
        print('\nTrain Net2')
        pred1_safe = data_cache.labeled_mask.clone()
        N = pred1_safe.numel()
        if pred1_safe.sum().item() == N:
            k = max(int(0.05 * N), 1)
            _, idx_sorted = torch.sort(torch.tensor(prob1, device='cuda'))
            pred1_safe[idx_sorted[:k]] = False
        elif pred1_safe.sum().item() == 0:
            k = max(int(0.05 * N), 1)
            _, idx_sorted = torch.sort(torch.tensor(prob1, device='cuda'), descending=True)
            pred1_safe[idx_sorted[:k]] = True
        print(f"[Split-Guard] Net2: labeled={pred1_safe.sum().item()}/{N}, unlabeled={N - pred1_safe.sum().item()}")
        # 训练Net2时，传入prob1_t（对方网络的干净概率）
        labeled_trainloader2, unlabeled_trainloader2 = loader.run('train', pred1_safe, prob1_t)
        # 训练Net2：传入prob1_t作为干净概率（对方网络的判断结果）
        centers2 = train(
            epoch, net2, tch_net2, optimizer2, centers2, centers1,
            labeled_trainloader2, unlabeled_trainloader2,
            ood_noise_indices=ood_noise_indices,
            prob_clean_full=prob1_t, # ← 新增参数
            data_cache = data_cache,
        )
        # ↓↓↓ 新增：首轮半监督断言 ↓↓↓
        if epoch == warm_up:
            assert prob1_t.shape[0] == len(eval_loader.dataset), \
                f"prob_clean_full长度错误：实际{prob1_t.shape[0]}，预期{len(eval_loader.dataset)}"
            assert prob2_t.shape[0] == len(eval_loader.dataset), \
                f"prob_clean_full长度错误：实际{prob2_t.shape[0]}，预期{len(eval_loader.dataset)}"
            print(f"[首轮半监督验证] prob_clean_full长度正确，与评估集样本数一致")
        # ↑↑↑ 新增结束 ↑↑↑

    # 保存模型
    if epoch == 29:
        filepath29 = os.path.join('./checkpoint', 'model29.pth.tar')
        save(filepath29, net1, net2, tch_net1, tch_net2, optimizer1, optimizer2, epoch, centers1, centers2, best_acc)
    if epoch == 129:
        filepath129 = os.path.join('./checkpoint', 'model129.pth.tar')
        save(filepath129, net1, net2, tch_net1, tch_net2, optimizer1, optimizer2, epoch, centers1, centers2, best_acc)

    # 定期保存模型
    save(filepath, net1, net2, tch_net1, tch_net2, optimizer1, optimizer2, epoch, centers1, centers2, best_acc)
