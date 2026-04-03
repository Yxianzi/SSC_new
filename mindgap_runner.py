# -*- coding:utf-8 -*-
"""Common training loop for the three proposed innovations."""

import math
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

import mmd
import utils
from contrastive_loss import SupConLoss, WeightedSupConLoss
from mindgap_model import MindGapModel
from pseudo_labeling import (
    PrototypeBank,
    RobustPseudoLabelEngine,
    build_ema_teacher,
    mc_dropout_predict,
    update_ema,
    weighted_pseudo_ce_loss,
)


def _to_namespace(config_module):
    return config_module


def _source_prior_from_label_map(label_map, class_num, device):
    labels = torch.as_tensor(label_map).reshape(-1).long()
    labels = labels[labels > 0] - 1
    if labels.numel() == 0:
        return torch.ones(class_num, device=device) / class_num
    counts = torch.bincount(labels, minlength=class_num).float()
    return (counts / counts.sum().clamp(min=1.0)).to(device)


def _set_ssc_trainable(model, trainable):
    for parameter in model.ssc.parameters():
        parameter.requires_grad = trainable


def _build_optimizer(model, base_lr, momentum, weight_decay):
    other_params = []
    ssc_params = []
    for name, parameter in model.named_parameters():
        # 这里不要使用 requires_grad 判断，把所有参数都加进优化器，方便后续动态调整
        if name.startswith('ssc.'):
            ssc_params.append(parameter)
        else:
            other_params.append(parameter)

    param_groups = [{'params': other_params, 'lr': base_lr}]
    if ssc_params:
        param_groups.append({
            'params': ssc_params,
            'lr': 0.0, # 初始化时设为 0，在 epoch 循环内动态调整
        })

    return torch.optim.SGD(param_groups, lr=base_lr, momentum=momentum, weight_decay=weight_decay)

def _spectral_weight(epoch, epochs, ssc_unfreeze_epoch, spectral_reg_weight):
    if epoch < ssc_unfreeze_epoch:
        return 0.0
    local_epoch = epoch - ssc_unfreeze_epoch + 1
    if local_epoch <= 5:
        return spectral_reg_weight
    decay_span = max(epochs - ssc_unfreeze_epoch - 4, 1)
    decay_ratio = min((local_epoch - 5) / decay_span, 1.0)
    return spectral_reg_weight * (1.0 - 0.8 * decay_ratio)


def _target_loss_scale(epoch, warmup_epochs, ramp_epochs):
    if epoch <= warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    progress = (epoch - warmup_epochs) / float(ramp_epochs)
    return max(0.0, min(progress, 1.0))


def _sync_teacher(student_model, teacher_model):
    teacher_model.load_state_dict(student_model.state_dict())


@torch.no_grad()
def _evaluate_teacher(
    teacher,
    test_loader,
    prototype_bank,
    device,
    use_ssc=True,
    use_prototype_attention=False,
    prototype_blend=1.0,
):
    teacher.eval()
    total_rewards = 0
    predict = np.array([], dtype=np.int64)
    labels = np.array([], dtype=np.int64)

    eval_source_prototypes = None
    if use_prototype_attention and prototype_bank.num_valid_source() > 0:
        eval_source_prototypes = prototype_bank.get_source()

    for test_datas, test_labels in test_loader:
        batch_size_eval = test_labels.shape[0]
        test_datas = test_datas.float().to(device)
        test_outputs = teacher.forward_target(
            Variable(test_datas),
            source_prototypes=eval_source_prototypes,
            use_ssc=use_ssc,
            use_prototype_attention=eval_source_prototypes is not None,
            prototype_blend=prototype_blend,
        )
        pred = test_outputs['logits'].data.max(1)[1]

        test_labels_np = test_labels.numpy()
        rewards = [1 if pred[j] == test_labels_np[j] else 0 for j in range(batch_size_eval)]
        total_rewards += np.sum(rewards)
        predict = np.append(predict, pred.cpu().numpy())
        labels = np.append(labels, test_labels_np)

    test_accuracy = 100.0 * total_rewards / len(test_loader.dataset)
    return test_accuracy, predict, labels


def _safe_target_contrastive(loss_fn, features, labels, weights, min_weight=0.0, min_count=2):
    if features.size(0) == 0:
        return torch.zeros((), device=weights.device, dtype=weights.dtype)

    valid_mask = weights >= min_weight
    if valid_mask.sum() < min_count:
        return torch.zeros((), device=weights.device, dtype=weights.dtype)

    selected_features = features[valid_mask]
    selected_labels = labels[valid_mask]
    selected_weights = weights[valid_mask]
    if selected_labels.unique().numel() < 2:
        return torch.zeros((), device=weights.device, dtype=weights.dtype)
    return loss_fn(selected_features, labels=selected_labels, sample_weights=selected_weights)


def run_mindgap_experiment(data_s, label_s, data_t, label_t, config):
    cfg = _to_namespace(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class_num = cfg.CLASS_NUM
    batch_size = cfg.BATCH_SIZE
    epochs = cfg.epochs

    source_contrastive = SupConLoss(temperature=0.1).to(device)
    target_contrastive = WeightedSupConLoss(temperature=0.1).to(device)
    cross_entropy = nn.CrossEntropyLoss().to(device)

    ssc_unfreeze_epoch = getattr(cfg, 'ssc_unfreeze_epoch', max(epochs - 30, 1))
    ssc_lr_scale = getattr(cfg, 'ssc_lr_scale', 0.1)
    spectral_reg_weight = getattr(cfg, 'spectral_reg_weight', 1.0)
    teacher_momentum = getattr(cfg, 'teacher_momentum', 0.99)
    prototype_momentum = getattr(cfg, 'prototype_momentum', 0.9)
    base_threshold = getattr(cfg, 'base_threshold', 0.80)
    min_threshold = getattr(cfg, 'min_threshold', 0.40)
    max_threshold = getattr(cfg, 'max_threshold', 0.95)
    da_momentum = getattr(cfg, 'distribution_align_momentum', 0.9)
    class_momentum = getattr(cfg, 'class_threshold_momentum', 0.9)
    mc_dropout_passes = getattr(cfg, 'mc_dropout_passes', 4)
    pseudo_loss_weight = getattr(cfg, 'pseudo_loss_weight', 1.0)
    target_contrastive_weight = getattr(cfg, 'target_contrastive_weight', 0.25)
    lmmd_weight = getattr(cfg, 'lmmd_weight', 0.01)
    source_samples_per_class = getattr(cfg, 'source_samples_per_class', 180)
    inverse_lr_schedule = getattr(cfg, 'inverse_lr_schedule', False)
    target_warmup_epochs = getattr(cfg, 'target_warmup_epochs', 10)
    target_ramp_epochs = getattr(cfg, 'target_ramp_epochs', 20)
    prototype_attention_start_epoch = getattr(
        cfg,
        'prototype_attention_start_epoch',
        max(target_warmup_epochs + target_ramp_epochs + 10, ssc_unfreeze_epoch),
    )
    target_contrastive_start_epoch = getattr(
        cfg,
        'target_contrastive_start_epoch',
        prototype_attention_start_epoch,
    )
    prototype_attention_ramp_epochs = getattr(cfg, 'prototype_attention_ramp_epochs', 15)
    prototype_attention_max_scale = getattr(cfg, 'prototype_attention_max_scale', 0.35)
    target_contrastive_min_weight = getattr(cfg, 'target_contrastive_min_weight', 0.75)
    target_contrastive_min_count = getattr(cfg, 'target_contrastive_min_count', max(class_num, 4))
    pseudo_fallback_topk = getattr(cfg, 'pseudo_fallback_topk', 4)
    pseudo_min_weight = getattr(cfg, 'pseudo_min_weight', 0.05)
    enable_distribution_alignment = getattr(cfg, 'enable_distribution_alignment', False)
    eval_interval = getattr(cfg, 'eval_interval', 5)

    acc = np.zeros([cfg.nDataSet, 1])
    A = np.zeros([cfg.nDataSet, class_num])
    k = np.zeros([cfg.nDataSet, 1])
    best_predict_all = []
    best_G, best_RandPerm, best_Row, best_Column = None, None, None, None

    for iDataSet in range(cfg.nDataSet):
        print('#######################idataset######################## ', iDataSet)
        utils.set_seed(cfg.seeds[iDataSet])

        trainX, trainY = utils.get_sample_data(data_s, label_s, cfg.HalfWidth, source_samples_per_class)
        testID, testX, testY, G, RandPerm, Row, Column = utils.get_all_data(data_t, label_t, cfg.HalfWidth)

        # 使用 torch.from_numpy() 替代 torch.tensor()，并显式指定最终的类型
        train_dataset = TensorDataset(torch.from_numpy(trainX).float(), torch.from_numpy(trainY).long())
        test_dataset = TensorDataset(torch.from_numpy(testX).float(), torch.from_numpy(testY).long())

        train_loader_s = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        train_loader_t = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        model = MindGapModel(cfg.nBand, cfg.patch_size, class_num, guide_eps=getattr(cfg, 'ssc_guide_eps', 1e-2)).to(device)
        model.ssc.set_domain_statistics(data_s, data_t)
        teacher = build_ema_teacher(model).to(device)

        source_prior = _source_prior_from_label_map(label_s, class_num, device)
        prototype_bank = PrototypeBank(class_num, model.feature_dim, device=device, momentum=prototype_momentum)
        pseudo_engine = RobustPseudoLabelEngine(
            class_num,
            source_prior,
            device=device,
            base_threshold=base_threshold,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            prior_momentum=da_momentum,
            class_momentum=class_momentum,
            fallback_topk=pseudo_fallback_topk,
            fallback_min_weight=pseudo_min_weight,
            enable_distribution_alignment=enable_distribution_alignment,
        )

        optimizer = _build_optimizer(
            model,
            cfg.lr,
            cfg.momentum,
            cfg.l2_decay,

        )

        print("Training...")
        last_accuracy = 0.0
        best_episdoe = 0
        train_start = time.time()
        train_end = train_start
        test_end = train_start
        best_predict = np.array([], dtype=np.int64)
        best_labels = np.array([], dtype=np.int64)

        for epoch in range(1, epochs + 1):
            total_hit, size = 0.0, 0.0
            if inverse_lr_schedule:
                learning_rate = cfg.lr / math.pow((1 + 10 * (epoch - 1) / epochs), 0.75)
            else:
                learning_rate = cfg.lr

            ssc_trainable = epoch >= ssc_unfreeze_epoch
            use_ssc = epoch >= ssc_unfreeze_epoch
            target_loss_scale = _target_loss_scale(epoch, target_warmup_epochs, target_ramp_epochs)
            prototype_blend = prototype_attention_max_scale * _target_loss_scale(
                epoch,
                max(prototype_attention_start_epoch - 1, 0),
                prototype_attention_ramp_epochs,
            )
            _set_ssc_trainable(model, ssc_trainable)
            # -----------------------------------------------------------------
            # 新增：动态更新优化器内部的学习率，彻底删除这里重新实例化 optimizer 的代码
            optimizer.param_groups[0]['lr'] = learning_rate
            if len(optimizer.param_groups) > 1:
                optimizer.param_groups[1]['lr'] = learning_rate * ssc_lr_scale if ssc_trainable else 0.0
            # -----------------------------------------------------------------
            current_spectral_weight = _spectral_weight(epoch, epochs, ssc_unfreeze_epoch, spectral_reg_weight)

            model.train()
            teacher.eval()

            if epoch == target_warmup_epochs + 1:
                _sync_teacher(model, teacher)

            iter_source = iter(train_loader_s)
            iter_target = iter(train_loader_t)
            num_iter = len(train_loader_s)

            for _ in range(1, num_iter):
                source_data, source_label = next(iter_source)
                try:
                    target_data, _ = next(iter_target)
                except StopIteration:
                    iter_target = iter(train_loader_t)
                    target_data, _ = next(iter_target)

                source_data = source_data.float().to(device)
                source_label = source_label.long().to(device)
                target_data = target_data.float().to(device)

                target_weak = utils.weak_augmentation(target_data).float().to(device)
                target_strong_1 = utils.target_consistency_augmentation(target_data).float().to(device)
                target_strong_2 = utils.target_consistency_augmentation(target_data).float().to(device)
                source_view_1 = utils.strong_augmentation(source_data).float().to(device)
                source_view_2 = utils.strong_augmentation(source_data).float().to(device)

                source_outputs = model.forward_source(
                    source_data,
                    target_weak,
                    use_ssc=use_ssc,
                    use_prototype_attention=False,
                )
                prototype_bank.update_source(source_outputs['backbone_features'], source_label)

                active_source_prototypes = None
                if (
                    target_loss_scale > 0.0
                    and prototype_blend > 0.0
                    and prototype_bank.num_valid_source() == class_num
                ):
                    active_source_prototypes = prototype_bank.get_source()

                target_contrastive_scale = _target_loss_scale(
                    epoch,
                    max(target_contrastive_start_epoch - 1, 0),
                    target_ramp_epochs,
                )

                with torch.no_grad():
                    teacher_probs, uncertainty, teacher_target_outputs = mc_dropout_predict(
                        teacher,
                        target_weak,
                        source_prototypes=active_source_prototypes,
                        mc_passes=mc_dropout_passes,
                        use_ssc=use_ssc,
                        prototype_blend=prototype_blend,
                    )
                    pseudo_info = pseudo_engine.generate(teacher_probs, uncertainty)
                    if target_loss_scale > 0.0:
                        prototype_bank.update_target(
                            teacher_target_outputs['backbone_features'],
                            pseudo_info['labels'],
                            pseudo_info['strict_weights'],
                            pseudo_info['strict_mask'],
                        )

                target_sample_weights = pseudo_info['strict_weights']

                if active_source_prototypes is not None:
                    source_outputs = model.forward_source(
                        source_data,
                        target_weak,
                        target_prototypes=active_source_prototypes,
                        use_ssc=use_ssc,
                        use_prototype_attention=True,
                        prototype_blend=prototype_blend,
                    )
                    source_view_1_outputs = model.forward_source(
                        source_view_1,
                        target_weak,
                        target_prototypes=active_source_prototypes,
                        use_ssc=use_ssc,
                        use_prototype_attention=True,
                        prototype_blend=prototype_blend,
                    )
                    source_view_2_outputs = model.forward_source(
                        source_view_2,
                        target_weak,
                        target_prototypes=active_source_prototypes,
                        use_ssc=use_ssc,
                        use_prototype_attention=True,
                        prototype_blend=prototype_blend,
                    )
                else:
                    source_view_1_outputs = model.forward_source(
                        source_view_1,
                        target_weak,
                        use_ssc=use_ssc,
                        use_prototype_attention=False,
                    )
                    source_view_2_outputs = model.forward_source(
                        source_view_2,
                        target_weak,
                        use_ssc=use_ssc,
                        use_prototype_attention=False,
                    )
                target_alignment_outputs = model.forward_target(
                    target_weak,
                    source_prototypes=active_source_prototypes,
                    use_ssc=use_ssc,
                    use_prototype_attention=active_source_prototypes is not None,
                    prototype_blend=prototype_blend,
                )
                if target_contrastive_scale > 0.0:
                    target_view_1_outputs = model.forward_target(
                        target_strong_1,
                        source_prototypes=active_source_prototypes,
                        use_ssc=use_ssc,
                        use_prototype_attention=active_source_prototypes is not None,
                        prototype_blend=prototype_blend,
                    )
                    target_view_2_outputs = model.forward_target(
                        target_strong_2,
                        source_prototypes=active_source_prototypes,
                        use_ssc=use_ssc,
                        use_prototype_attention=active_source_prototypes is not None,
                        prototype_blend=prototype_blend,
                    )
                    target_contrastive_features = torch.cat([
                        target_view_1_outputs['projection'].unsqueeze(1),
                        target_view_2_outputs['projection'].unsqueeze(1),
                    ], dim=1)
                else:
                    target_view_1_outputs = target_alignment_outputs
                    target_view_2_outputs = target_alignment_outputs
                    target_contrastive_features = None

                source_contrastive_features = torch.cat([
                    source_view_1_outputs['projection'].unsqueeze(1),
                    source_view_2_outputs['projection'].unsqueeze(1),
                ], dim=1)

                classification_loss = cross_entropy(source_outputs['logits'], source_label)
                source_contrastive_loss = source_contrastive(source_contrastive_features, source_label)
                if target_contrastive_features is None:
                    target_contrastive_loss = torch.zeros((), device=device, dtype=source_outputs['features'].dtype)
                else:
                    target_contrastive_loss = _safe_target_contrastive(
                        target_contrastive,
                        target_contrastive_features,
                        pseudo_info['labels'],
                        target_sample_weights,
                        min_weight=target_contrastive_min_weight,
                        min_count=target_contrastive_min_count,
                    )
                pseudo_classification_loss = weighted_pseudo_ce_loss(
                    target_alignment_outputs['logits'],
                    pseudo_info['labels'],
                    target_sample_weights,
                )
                # 提取平滑的 target_prob 供 LMMD 使用
                student_target_prob = torch.nn.functional.softmax(target_alignment_outputs['logits'], dim=1)

                lmmd_loss = mmd.weighted_lmmd(
                    source_outputs['features'],
                    target_alignment_outputs['features'],
                    source_label,
                    student_target_prob,
                    target_weights=None,
                    CLASS_NUM=class_num,
                )
                lambd = 2 / (1 + math.exp(-10 * epoch / epochs)) - 1
                scaled_target_contrastive_loss = (
                        target_contrastive_scale * target_contrastive_weight * target_contrastive_loss
                )

                # ================= 核心修复：切断伪标签的直接 CE 惩罚，防止多数类崩塌 =================
                # 直接将目标域交叉熵损失置 0，仅让伪标签通过上方的 target_contrastive_loss 生效
                scaled_pseudo_classification_loss = torch.zeros_like(classification_loss)
                # ==============================================================================

                scaled_lmmd_loss = lmmd_weight * lambd * lmmd_loss

                total_loss = (
                        classification_loss
                        + source_contrastive_loss
                        + scaled_target_contrastive_loss
                        + scaled_lmmd_loss
                        + current_spectral_weight * source_outputs['spectral_reg']
                )

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                update_ema(model, teacher, momentum=teacher_momentum)

                pred = source_outputs['logits'].data.max(1)[1]
                total_hit += pred.eq(source_label.data).sum().item()
                size += source_label.data.size(0)

            pseudo_dom = float(pseudo_info['selected_distribution'].max().item()) if pseudo_info['mask'].any() else 0.0
            strict_selected = int(pseudo_info['strict_mask'].sum().item())
            contrastive_selected = int((target_sample_weights >= target_contrastive_min_weight).sum().item())
            print(
                'epoch {:>3d}: cls {:6.4f}, s_con {:6.4f}, t_con {:6.4f}, pseudo {:6.4f}, lmmd {:6.4f}, spec {:6.4f}, scale {:5.2f}, p_attn {:4.2f}, c_sel {:>2d}, s_sel {:>2d}, sel {:>2d}, pmax {:4.2f}, acc {:6.4f}, total {:6.4f}'.format(
                    epoch,
                    classification_loss.item(),
                    source_contrastive_loss.item(),
                    scaled_target_contrastive_loss.item(),
                    scaled_pseudo_classification_loss.item(),
                    scaled_lmmd_loss.item(),
                    source_outputs['spectral_reg'].item(),
                    target_loss_scale,
                    prototype_blend,
                    contrastive_selected,
                    strict_selected,
                    int(pseudo_info['mask'].sum().item()),
                    pseudo_dom,
                    total_hit / max(size, 1.0),
                    total_loss.item(),
                )
            )

            train_end = time.time()
            if epoch % eval_interval == 0 or epoch == epochs:
                eval_use_prototype_attention = (
                    epoch >= prototype_attention_start_epoch and prototype_bank.num_valid_source() == class_num
                )
                test_accuracy, predict, labels = _evaluate_teacher(
                    teacher,
                    test_loader,
                    prototype_bank,
                    device,
                    use_ssc=use_ssc,
                    use_prototype_attention=eval_use_prototype_attention,
                    prototype_blend=prototype_blend,
                )
                test_end = time.time()

                print('\t\tEval epoch {}: {:.2f}%'.format(epoch, test_accuracy))

                if test_accuracy > last_accuracy:
                    print("save networks for epoch:", epoch)
                    last_accuracy = test_accuracy
                    best_episdoe = epoch
                    best_predict = predict
                    best_labels = labels
                    best_predict_all = predict
                    best_G, best_RandPerm, best_Row, best_Column = G, RandPerm, Row, Column
                    print('best epoch:[{}], best accuracy={}'.format(best_episdoe, last_accuracy))

                print('iter:{} best epoch:[{}], best accuracy={}'.format(iDataSet, best_episdoe, last_accuracy))
                print('***********************************************************************************')

        acc[iDataSet] = last_accuracy
        if best_labels.size == 0:
            best_labels = np.array([], dtype=np.int64)
            best_predict = np.array([], dtype=np.int64)
        if best_labels.size > 0:
            C = metrics.confusion_matrix(best_labels, best_predict, labels=np.arange(class_num))
            row_sum = np.sum(C, 1, dtype=np.float64)
            row_sum[row_sum == 0] = 1e-12
            A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float64)
            k[iDataSet] = metrics.cohen_kappa_score(best_labels, best_predict)

    AA = np.mean(A, 1)
    AAMean = np.mean(AA, 0)
    AAStd = np.std(AA)
    AMean = np.mean(A, 0)
    AStd = np.std(A, 0)
    OAMean = np.mean(acc)
    OAStd = np.std(acc)
    kMean = np.mean(k)
    kStd = np.std(k)

    print("train time per DataSet(s): " + "{:.5f}".format(train_end - train_start))
    print("test time per DataSet(s): " + "{:.5f}".format(test_end - train_end))
    print("average OA: " + "{:.2f}".format(OAMean) + " +- " + "{:.2f}".format(OAStd))
    print("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
    print("average kappa: " + "{:.4f}".format(100 * kMean) + " +- " + "{:.4f}".format(100 * kStd))
    print("accuracy for each class: ")
    for i in range(class_num):
        print("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))

    best_iDataset = 0
    for i in range(len(acc)):
        print('{}:{}'.format(i, acc[i]))
        if acc[i] > acc[best_iDataset]:
            best_iDataset = i
    print('best acc all={}'.format(acc[best_iDataset]))

    return {
        'acc': acc,
        'A': A,
        'k': k,
        'best_predict_all': best_predict_all,
        'best_G': best_G,
        'best_RandPerm': best_RandPerm,
        'best_Row': best_Row,
        'best_Column': best_Column,
    }
