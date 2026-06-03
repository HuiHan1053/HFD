from __future__ import print_function
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from PreResNet import *
from contrastive_loss import *
from sklearn.mixture import GaussianMixture
import dataloader_cifar as dataloader
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=128, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode',  default='asym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--lambda_c', default=0.025, type=float, help='weight for contrastive loss')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--r', default=0.4, type=float, help='noise ratio')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--dr_dim', default=128, type=int)
parser.add_argument('--threshold', default=0.95, type=float, help='pseudo label threshold')
parser.add_argument('--eps', default=0.999, type=float, help='Running average of model weights')
parser.add_argument('--data_path', default='/root/autodl-tmp/cifar-10-batches-py', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Training
def train(epoch,net,tch_net,optimizer,centers,labeled_trainloader,unlabeled_trainloader):      
    net.train()
    tch_net.train()
    
    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    
    eLce = 0
    eLu = 0
    ePenalty = 0
    eLfix = 0
    eLls = 0
    eLoss_simCLR = 0

    for batch_idx, (inputs_x1, inputs_x2, inputs_x3, inputs_x4, labels_x, index, w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u1, inputs_u2, inputs_u3, inputs_u4 = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u1, inputs_u2, inputs_u3, inputs_u4 = unlabeled_train_iter.next()

        update_ema_variables(net, tch_net, epoch)

        w_x = w_x.view(-1, 1).type(torch.FloatTensor)
        
        inputs_x1 = inputs_x1.cuda()
        inputs_x2 = inputs_x2.cuda()
        inputs_x3 = inputs_x3.cuda()
        inputs_x4 = inputs_x4.cuda()
        inputs_u1 = inputs_u1.cuda()
        inputs_u2 = inputs_u2.cuda()
        inputs_u3 = inputs_u3.cuda()
        inputs_u4 = inputs_u4.cuda()
        labels_x = labels_x.cuda()
        w_x = w_x.cuda()

        if epoch > 129:
            with torch.no_grad():
                _, logits_x1 = net(inputs_x1)                                         
                _, logits_x2 = net(inputs_x2)               
                
                _, logits_u1 = net(inputs_u1)
                _, logits_u2 = net(inputs_u2)
                _, logits_u1_tch = tch_net(inputs_u1)
                _, logits_u2_tch = tch_net(inputs_u2)
                
                logits_x1 = logits_x1.detach()
                logits_x2 = logits_x2.detach()
            
                logits_u1 = logits_u1.detach()
                logits_u2 = logits_u2.detach()
                logits_u1_tch = logits_u1_tch.detach()
                logits_u2_tch = logits_u2_tch.detach()
                
                
                #构噪干净标签集
                labels_x_m01 = torch.nn.functional.one_hot(labels_x.cuda(), num_classes=args.num_class).float()           
                px = (torch.softmax(logits_x1, dim=1) + torch.softmax(logits_x2, dim=1)) / 2
                px = w_x * labels_x_m01 + (1 - w_x) * px
                ptx = px
                #ptx = px ** (1 / args.T)
                targets_x = ptx / ptx.sum(dim=1, keepdim=True)
        
                #构造噪声标签集
                pu = (torch.softmax(logits_u1, dim=1) + torch.softmax(logits_u1_tch, dim=1) + torch.softmax(logits_u2, dim=1) + torch.softmax(logits_u2_tch, dim=1)) / 4
                #ptu = pu ** (1 / args.T)
                ptu = pu
                targets_u = ptu / ptu.sum(dim=1, keepdim=True)
                
            f1, _ = net(inputs_x1)
            f2 = torch.stack([centers[labels_x[i]] for i in range(labels_x.size(0))], dim=0).cuda()
            
            f1 = F.normalize(f1, dim=1)
            f2 = F.normalize(f2, dim=1)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss_simCLR = contrastive_criterion(features)
            
            size_m = inputs_x3.size(0) + inputs_x4.size(0)
            l = np.random.beta(args.alpha, args.alpha)        
            l = max(l, 1-l)  
            all_inputs  = torch.cat([inputs_x3,inputs_x4,inputs_u3,inputs_u4], dim=0)
            all_targets = torch.cat([targets_x,targets_x,targets_u,targets_u], dim=0)
            idx = torch.randperm(all_inputs.size(0))
            input_a, input_b   = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]
            mixed_input  = l * input_a  + (1 - l) * input_b        
            mixed_target = l * target_a + (1 - l) * target_b
            _, mixed_logit = net(mixed_input) 
            logits_mixed_x = mixed_logit[:size_m]
            logits_mixed_u = mixed_logit[size_m:]
            targets_mixed_x = mixed_target[:size_m]
            targets_mixed_u = mixed_target[size_m:]
            
            Lce,Lu=criterion_semi(logits_mixed_x,targets_mixed_x,logits_mixed_u,targets_mixed_u)
            
            prior = (torch.ones(args.num_class)/args.num_class).cuda()   
            pred_mean = torch.softmax(mixed_logit, dim=1).mean(0)
            penalty = torch.sum(prior*torch.log(prior/pred_mean))
            
            loss = Lce + args.lambda_u * Lu + penalty + args.lambda_c * loss_simCLR
            
            eLce += Lce.item()
            eLu += Lu.item()
            ePenalty += penalty.item()
            eLoss_simCLR += loss_simCLR.item()
            
        else:
            #噪声标签数据剔除标签FixMatch
            inputs_u = torch.cat([inputs_u3, inputs_u3, inputs_u4, inputs_u4], dim=0)
            inputs_u_l = torch.cat([inputs_u1, inputs_u1, inputs_u2, inputs_u2], dim=0)
            _, logits_u = net(inputs_u)
            _, logits_u_l = tch_net(inputs_u_l)
            logits_u_l = logits_u_l.detach()

            pseudo_label = torch.softmax(logits_u_l/args.T, dim=-1)               
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()
            Lfix = (F.cross_entropy(logits_u, targets_u, reduction='none') * mask).mean()

            #干净标签数据标签平滑
            inputs_x = torch.cat([inputs_x3, inputs_x4], dim=0)
            inputs_x_l = torch.cat([inputs_x1, inputs_x2], dim=0)
            labels_x_d = torch.cat([labels_x, labels_x], dim=0)
            _, logits_x = net(inputs_x)
            _, logits_x_l = tch_net(inputs_x_l)
            logits_x_l = logits_x_l.detach()
            Lls = criterion_ls(logits_x, logits_x_l)
            Lce = CEloss(logits_x, labels_x_d)

            loss = Lce + Lfix + Lls
            
            eLce += Lce.item()
            eLfix += Lfix.item()
            eLls += Lls.item()
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Loss: %.2f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()
        
    loss_log.write('Epoch:%d Lce:%.2f Lu:%.2f penalty:%.2f Lfix:%.2f Lls:%.2f Lsim:%.2f\n'%(epoch,eLce/num_iter,eLu/num_iter,ePenalty/num_iter,eLfix/num_iter,eLls/num_iter,eLoss_simCLR/num_iter))
    loss_log.flush()

def warmup(epoch,net,optimizer,dataloader):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    eLoss = 0
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        _, outputs = net(inputs)               
        loss = CLoss(a=(1-args.r), b=args.r, c=0.1)(outputs, labels)
        loss.backward()  
        optimizer.step()
        
        eLoss += loss.item()

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()
        
    loss_log.write('Epoch:%d Loss:%.2f\n'%(epoch,eLoss/num_iter))
    loss_log.flush()

def test(epoch,net1,net2):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0

    global best_acc
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            _, outputs1 = net1(inputs)
            _, outputs2 = net2(inputs)           
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)            
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                 
    acc = 100.*correct/total
    
    if acc > best_acc:
        best_acc = acc
        torch.save(net1.state_dict(), './checkpoint/net1_best.pth.tar')
        torch.save(net2.state_dict(), './checkpoint/net2_best.pth.tar')
        
    print("\n| Test Epoch #%d\t Accuracy: %.2f%% Best_Accuracy: %.2f%%\n" %(epoch,acc,best_acc))  
    test_log.write('Epoch:%d   Accuracy:%.2f   Best_Accuracy:%.2f\n'%(epoch,acc,best_acc))
    test_log.flush()  

def eval_train(model, epoch, centers):    
    model.eval()
    all_targets = torch.zeros(num_samples)
    normalized_features = torch.zeros((num_samples, args.dr_dim))
    predictions = torch.zeros(num_samples)
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            feature, outputs = model(inputs)
            normalized_fea = F.normalize(feature.detach(), dim=1)
            _, predicted = torch.max(outputs, 1)

            for b in range(inputs.size(0)):
                all_targets[index[b]] = targets[b]
                normalized_features[index[b]] = normalized_fea[b]
                predictions[index[b]] = predicted[b]

    losses_proto = torch.zeros((num_samples, ))
    centers_eval = torch.zeros((args.num_class, args.dr_dim))
    for cls_ in range(args.num_class):
        centers_eval[cls_] = F.normalize(normalized_features[predictions==cls_].mean(dim=0), dim=0)
        
    if epoch == warm_up:
        centers = centers_eval
        
    normalized_centers = F.normalize(centers, dim=1)

    for cls_ in range(args.num_class):
        logits_proto = torch.mm(normalized_features[all_targets==cls_], normalized_centers[cls_].reshape(-1, 1)) / 0.1
        loss_proto = - torch.log_softmax(logits_proto, dim = 0)
        losses_proto[all_targets==cls_] = loss_proto.flatten()

    losses_proto = (losses_proto-losses_proto.min())/(losses_proto.max()-losses_proto.min())

    input_loss = losses_proto.reshape(-1,1).detach().numpy()
    
    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    prob = prob[:,gmm.means_.argmin()]
    return prob, input_loss, centers_eval

class CLoss(torch.nn.Module):
    def __init__(self, a, b, c):
        super(CLoss, self).__init__()
        self.a = a
        self.b = b
        self.c = c
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, outputs, labels):
        
        l1 = self.cross_entropy(outputs, labels)
       
        pred = F.softmax(outputs, dim=1)
        pred = torch.log(1-pred)      
        l2 = F.nll_loss(pred,labels)

        pred = F.softmax(outputs, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, args.num_class).float().cuda()
        label_one_hot = torch.clamp(label_one_hot, min=1e-10, max=1.0) #min=1e-10forCifar100;min=1e-1foranimal10N;min=1e-45forClothing1M
        l3 = -torch.mean(torch.sum(pred * torch.log(label_one_hot), dim=1))
        
        loss = self.a * l1 + self.b * l2 + self.c * l3
        return loss

class LSLoss(object):
    def __call__(self, input_logits, target_logits, eps=0.35, tau=1/8, reduction='batchmean'):
        assert input_logits.size() == target_logits.size()
        input_log_softmax = F.log_softmax(input_logits, dim=1)
        target = F.softmax(target_logits, dim=1)

        _, C = target_logits.shape
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
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)
        return Lx, Lu

def update_ema_variables(model, ema_model, epoch):
    alpha = numbers[epoch-warm_up]
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_((1 - alpha) * param.data)

def create_model():
    model = ResNet18(num_classes=args.num_class)
    model = model.cuda()
    return model

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

def save(filepath,net1,net2,tch_net1,tch_net2,optimizer1,optimizer2,epoch,centers1,centers2,best_acc):
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
    title = 'Epoch-' + str(epoch)+':'
    fig = plt.figure()
    plt.subplot(121)
    gmm = GaussianMixture(n_components=2, max_iter=20, tol=1e-2, random_state=0, reg_covar=5e-4)
    model_1_loss = np.reshape(model_1_loss, (-1, 1))
    gmm.fit(model_1_loss)  # fit the loss

    # plot resulting fit
    x_range = np.linspace(0, 1, 1000)
    pdf = np.exp(gmm.score_samples(x_range.reshape(-1, 1)))
    responsibilities = gmm. predict_proba(x_range.reshape(-1, 1))
    pdf_individual = responsibilities * pdf[:, np.newaxis]
    plt.hist(np.array(model_1_loss[noise_index]), density=True, bins=100, alpha=0.5,histtype='bar', color='red', label='Noisy subset')
    plt.hist(np.array(model_1_loss[clean_index]), density=True, bins=100, alpha=0.5,histtype='bar', color='blue', label='Clean subset')
    plt.plot(x_range, pdf, '-k', label='Mixture')
    plt.plot(x_range, pdf_individual, '--', label='Component')
    plt.legend(loc='upper right', prop={'size': 12})
    plt.xlabel('Normalized loss')
    plt.ylabel('Estimated pdf')
    plt.title(title+'Model_1')

    plt.subplot(122)
    gmm = GaussianMixture(n_components=2, max_iter=20, tol=1e-2, random_state=0, reg_covar=5e-4)
    model_2_loss = np.reshape(model_2_loss, (-1, 1))
    gmm.fit(model_2_loss)  # fit the loss

    # plot resulting fit
    x_range = np.linspace(0, 1, 1000)
    pdf = np.exp(gmm.score_samples(x_range.reshape(-1, 1)))
    responsibilities = gmm.predict_proba(x_range.reshape(-1, 1))
    pdf_individual = responsibilities * pdf[:, np.newaxis]
    plt.hist(np.array(model_2_loss[noise_index]), density=True, bins=100, alpha=0.5,histtype='bar', color='red', label='Noisy subset')
    plt.hist(np.array(model_2_loss[clean_index]), density=True, bins=100, alpha=0.5,histtype='bar', color='blue', label='Clean subset')
    plt.plot(x_range, pdf, '-k', label='Mixture')
    plt.plot(x_range, pdf_individual, '--', label='Component')
    plt.legend(loc='upper right', prop={'size': 12})
    plt.xlabel('Normalized loss')
    plt.ylabel('Estimated pdf')
    plt.title(title+'Model_2')

    print('\nlogging histogram...')
    title = 'cifar10_' + str(args.noise_mode) + '_moit_double_' + str(noise_rate)
    plt.savefig(os.path.join('./figure_his/', 'two_model_{}_{}.{}'.format(epoch, title, "png")), dpi=300)
    plt.close()

if not os.path.exists('./checkpoint'): os.makedirs('./checkpoint')
if not os.path.exists('./figure_his'): os.makedirs('./figure_his')
        
filepath = os.path.join('./checkpoint', 'model.pth.tar')
stats_log=open('./checkpoint/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_stats.txt','a') 
test_log=open('./checkpoint/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_acc.txt','a')
loss_log=open('./checkpoint/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_loss.txt','a')     

if args.dataset=='cifar10': warm_up = 30
elif args.dataset=='cifar100': warm_up = 30

loader = dataloader.cifar_dataloader(args.dataset,r=args.r,noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=5,\
    root_dir=args.data_path,log=stats_log,noise_file='%s/%.1f_%s.json'%(args.data_path,args.r,args.noise_mode))

centers1 = torch.zeros((args.num_class, args.dr_dim))
centers2 = torch.zeros((args.num_class, args.dr_dim))
num_samples = 50000

best_acc = 0.0

print('| Building net')
net1 = create_model()
net2 = create_model()
tch_net1 = create_model()
tch_net2 = create_model()

cudnn.benchmark = True

for param in tch_net1.parameters(): param.requires_grad = False
for param in tch_net2.parameters(): param.requires_grad = False

criterion_ls = LSLoss()
criterion_semi  = SemiLoss()
CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
contrastive_criterion = SupConLoss()

optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

start_epoch = 0

if args.resume:
    net1, net2, tch_net1, tch_net2, optimizer1, optimizer2, start_epoch, centers1, centers2, best_acc= (
        resume(filepath, net1, net2, tch_net1, tch_net2, optimizer1, optimizer2))

numbers = np.linspace(0.99, 0.999, args.num_epochs-warm_up)
numbers_eval = np.linspace(0, 1, args.num_epochs-warm_up)
    
for epoch in range(start_epoch, args.num_epochs):
    if epoch > 149:
        lr = 0.002
    else:
        lr = 0.02

    for param_group in optimizer1.param_groups: param_group['lr'] = lr       
    for param_group in optimizer2.param_groups: param_group['lr'] = lr          
    test_loader = loader.run('test')
    eval_loader = loader.run('eval_train')

    noise_ind, clean_ind = eval_loader.dataset.if_noise()
    
    if epoch<warm_up:       
        warmup_trainloader = loader.run('warmup')
        print('Warmup Net1')
        warmup(epoch,net1,optimizer1,warmup_trainloader)    
        print('\nWarmup Net2')
        warmup(epoch,net2,optimizer2,warmup_trainloader) 
   
    else: 
        prob1, loss1, centers_eval1 = eval_train(net1, epoch, centers1)   
        prob2, loss2, centers_eval2  = eval_train(net2, epoch, centers2)          
               
        pred1 = (prob1 > args.p_threshold)      
        pred2 = (prob2 > args.p_threshold)
        
        plotHistogram(np.array(loss1), np.array(loss2), noise_ind, clean_ind, epoch, args.r)
        
        if epoch == warm_up:
            for param,param_tch in zip(net1.parameters(),tch_net1.parameters()): 
                param_tch.data.copy_(param.data)
            for param,param_tch in zip(net2.parameters(),tch_net2.parameters()): 
                param_tch.data.copy_(param.data)
        
        print('Train Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred2,prob2) # co-divide             
        train(epoch,net1,tch_net1,optimizer1,centers1,labeled_trainloader,unlabeled_trainloader)
        
        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred1,prob1) # co-divide
        train(epoch,net2,tch_net2,optimizer2,centers2,labeled_trainloader,unlabeled_trainloader)
        
        alpha_eval = numbers_eval[epoch-warm_up]
        
        centers1 = F.normalize(centers1, dim=1)
        centers2 = F.normalize(centers2, dim=1)

        centers1 = centers1.mul_(alpha_eval).add_(centers_eval1 * (1-alpha_eval))
        centers2 = centers2.mul_(alpha_eval).add_(centers_eval2 * (1-alpha_eval))

    test(epoch,net1,net2)
    
    if epoch == 29:
        filepath29 = os.path.join('./checkpoint', 'model29.pth.tar')
        save(filepath29,net1,net2,tch_net1,tch_net2,optimizer1,optimizer2,epoch,centers1,centers2,best_acc)
    if epoch == 129:
        filepath129 = os.path.join('./checkpoint', 'model129.pth.tar')
        save(filepath129,net1,net2,tch_net1,tch_net2,optimizer1,optimizer2,epoch,centers1,centers2,best_acc)
        
    save(filepath,net1,net2,tch_net1,tch_net2,optimizer1,optimizer2,epoch,centers1,centers2,best_acc)
