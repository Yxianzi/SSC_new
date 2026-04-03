#!/usr/bin/env python
# encoding: utf-8

import torch
from torch.autograd import Variable
import numpy as np
from Weight import Weight
import torch.nn.functional as F

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)


def mmd_rbf_accelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i+1)%batch_size
        t1, t2 = s1+batch_size, s2+batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / float(batch_size)

def mmd_rbf_noaccelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss

def cmmd(source, target, s_label, t_label, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    s_label = s_label.cpu()
    batch_size = int(source.size()[0])
    s_label = s_label.view(batch_size,1)
    s_label = torch.zeros(batch_size, 7).scatter_(1, s_label.data, 1)
    s_label = Variable(s_label).cuda()

    t_label = t_label.cpu()
    t_label = t_label.view(batch_size, 1)
    t_label = torch.zeros(batch_size, 7).scatter_(1, t_label.data, 1)
    t_label = Variable(t_label).cuda()


    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    loss += torch.mean(torch.mm(s_label, torch.transpose(s_label, 0, 1)) * XX +
                      torch.mm(t_label, torch.transpose(t_label, 0, 1)) * YY -
                      2 * torch.mm(s_label, torch.transpose(t_label, 0, 1)) * XY)
    return loss

#DSAN
def lmmd(source, target, s_label, t_label, kernel_mul=2.0, kernel_num=5, fix_sigma=None, CLASS_NUM=7, BATCH_SIZE=32):
    batch_size = source.size()[0]
    weight_ss, weight_tt, weight_st = Weight.cal_weight(s_label, t_label,batch_size=BATCH_SIZE,CLASS_NUM = CLASS_NUM)
    weight_ss = torch.from_numpy(weight_ss).cuda()
    weight_tt = torch.from_numpy(weight_tt).cuda()
    weight_st = torch.from_numpy(weight_st).cuda()

    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = torch.Tensor([0]).cuda()
    if torch.sum(torch.isnan(sum(kernels))):
        return loss
    SS = kernels[:batch_size, :batch_size]
    TT = kernels[batch_size:, batch_size:]
    ST = kernels[:batch_size, batch_size:]

    loss += torch.sum( weight_ss * SS + weight_tt * TT - 2 * weight_st * ST )
    return loss


def weighted_lmmd(
        source,
        target,
        s_label,
        t_prob,
        target_weights=None,
        kernel_mul=2.0,
        kernel_num=5,
        fix_sigma=None,
        CLASS_NUM=7,
):
    s_batch = source.size(0)
    t_batch = target.size(0)
    device = source.device

    if s_batch == 0 or t_batch == 0:
        return torch.zeros((), device=device, dtype=source.dtype)

    source_one_hot = F.one_hot(s_label.view(-1), num_classes=CLASS_NUM).float().to(device)
    source_class_mass = source_one_hot.sum(dim=0, keepdim=True).clamp(min=1.0)
    source_norm = source_one_hot / source_class_mass

    target_prob = t_prob.float().to(device)
    if target_weights is None:
        target_weights = torch.ones(t_batch, device=device, dtype=source.dtype)
    else:
        target_weights = target_weights.float().to(device)
    target_weights = target_weights.clamp(min=0.0).view(-1, 1)

    target_prob = target_prob * target_weights
    target_class_mass = target_prob.sum(dim=0, keepdim=True).clamp(min=1e-6)
    target_norm = target_prob / target_class_mass

    # 修复核心：使用目标域硬标签（Hard Label）来判定该批次实际存在的类别
    target_hard_labels = target_prob.argmax(dim=1)
    target_hard_one_hot = F.one_hot(target_hard_labels, num_classes=CLASS_NUM).float()

    # 只有源域和目标域同时存在的类别，才参与 MMD 矩阵计算
    valid_classes = (source_one_hot.sum(dim=0) > 0) & (target_hard_one_hot.sum(dim=0) > 0)

    if valid_classes.sum() == 0:
        return torch.zeros((), device=device, dtype=source.dtype)

    weight_ss = torch.matmul(source_norm[:, valid_classes], source_norm[:, valid_classes].t()) / valid_classes.sum()
    weight_tt = torch.matmul(target_norm[:, valid_classes], target_norm[:, valid_classes].t()) / valid_classes.sum()
    weight_st = torch.matmul(source_norm[:, valid_classes], target_norm[:, valid_classes].t()) / valid_classes.sum()

    kernels = guassian_kernel(
        source,
        target,
        kernel_mul=kernel_mul,
        kernel_num=kernel_num,
        fix_sigma=fix_sigma,
    )

    if torch.isnan(kernels).any():
        return torch.zeros((), device=device, dtype=source.dtype)

    SS = kernels[:s_batch, :s_batch]
    TT = kernels[s_batch:, s_batch:]
    ST = kernels[:s_batch, s_batch:]

    return torch.sum(weight_ss * SS + weight_tt * TT - 2 * weight_st * ST)

def mmd_linear(f_of_X, f_of_Y):
    delta = f_of_X - f_of_Y
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    return loss

# def mmd_linear_self(f_of_X,f_of_Y):
#     x_mean = torch
def prob_mmd_linear(input_list,class_num):

    ## input_list的第一个输入是特征，后一个是softmax的输出
    loss = 0
    outer_product_out = torch.bmm(input_list[0].unsqueeze(2), input_list[1].unsqueeze(1))
    batch_size = input_list[0].size(0) // 2
    for i in range(class_num):#class_num
        feat = outer_product_out.narrow(2, i, 1).squeeze(2)
        loss += mmd_linear(feat[:batch_size,:],feat[batch_size:,:])

    return loss


        # print(index)
        # print(location_x,location_y)


def SAN(input_list,ad_net_list,constant):
    loss = 0
    outer_product_out = torch.bmm(input_list[0].unsqueeze(2), input_list[1].unsqueeze(1))
    batch_size = input_list[0].size(0) // 2
    dc_target = (torch.from_numpy(np.array([[0]] * batch_size + [[1]] * batch_size)).type(torch.LongTensor).squeeze(1))
    # print(dc_target.shape)
    domain_criterion = torch.nn.NLLLoss()
    # dc_target = (torch.zeros(source_data.size()[0])).type(torch.LongTensor).cuda()
    if torch.cuda.is_available():
        dc_target = dc_target.cuda()
    for i in range(len(ad_net_list)):
        ad_out = ad_net_list[i](outer_product_out.narrow(2, i, 1).squeeze(2),constant)
        # print(ad_out)
        loss += domain_criterion(ad_out,dc_target)
    return loss


def EntropyLoss(input_):
    mask = input_.ge(0.000001)###与0.000001对比，大于则取1，反之取0
    mask_out = torch.masked_select(input_, mask)##平铺成为一维向量
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))##计算熵
    return entropy / float(input_.size(0))

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    if torch.cuda.device_count() > 1:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cuda')
else:
    device = torch.device('cpu')


def optimizer_scheduler_dann(lr,optimizer, p):
    """
    Adjust the learning rate of optimizer
    :param optimizer: optimizer for updating parameters
    :param p: a variable for adjusting learning rate
    :return: optimizer
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr / (1. + 10 * p) ** 0.75
    return optimizer
