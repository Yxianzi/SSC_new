# -*- coding:utf-8 -*-
"""Prototype memory and robust pseudo-label utilities."""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeBank:
    def __init__(self, num_classes, feature_dim, device, momentum=0.9):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.device = device
        self.momentum = momentum

        self.source_prototypes = torch.zeros(num_classes, feature_dim, device=device)
        self.target_prototypes = torch.zeros(num_classes, feature_dim, device=device)
        self.source_valid = torch.zeros(num_classes, dtype=torch.bool, device=device)
        self.target_valid = torch.zeros(num_classes, dtype=torch.bool, device=device)

    def _update(self, bank, valid_mask, features, labels, weights=None):
        if weights is None:
            weights = torch.ones(features.size(0), device=features.device, dtype=features.dtype)

        for class_idx in range(self.num_classes):
            class_mask = labels == class_idx
            if class_mask.sum() == 0:
                continue
            class_weights = weights[class_mask].unsqueeze(1)
            weight_sum = class_weights.sum()
            if weight_sum <= 0:
                continue
            class_feature = (features[class_mask] * class_weights).sum(dim=0) / weight_sum
            if valid_mask[class_idx]:
                bank[class_idx] = self.momentum * bank[class_idx] + (1.0 - self.momentum) * class_feature
            else:
                bank[class_idx] = class_feature
                valid_mask[class_idx] = True

    def update_source(self, features, labels):
        self._update(self.source_prototypes, self.source_valid, features.detach(), labels.detach())

    def update_target(self, features, pseudo_labels, sample_weights, selection_mask):
        if selection_mask.sum() == 0:
            return
        selected_features = features[selection_mask].detach()
        selected_labels = pseudo_labels[selection_mask].detach()
        selected_weights = sample_weights[selection_mask].detach()
        self._update(self.target_prototypes, self.target_valid, selected_features, selected_labels, selected_weights)

    def num_valid_source(self):
        return int(self.source_valid.sum().item())

    def num_valid_target(self):
        return int(self.target_valid.sum().item())

    def get_source(self):
        if not self.source_valid.any():
            return None
        return self.source_prototypes[self.source_valid]

    def get_target(self):
        if not self.target_valid.any():
            return None
        return self.target_prototypes[self.target_valid]


class RobustPseudoLabelEngine:
    def __init__(
        self,
        num_classes,
        source_prior,
        device,
        base_threshold=0.80,
        min_threshold=0.40,
        max_threshold=0.95,
        prior_momentum=0.9,
        class_momentum=0.9,
        fallback_topk=4,
        fallback_min_weight=0.05,
        enable_distribution_alignment=True,
    ):
        self.num_classes = num_classes
        self.device = device
        self.base_threshold = base_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.prior_momentum = prior_momentum
        self.class_momentum = class_momentum
        self.fallback_topk = fallback_topk
        self.fallback_min_weight = fallback_min_weight
        self.enable_distribution_alignment = enable_distribution_alignment

        self.source_prior = source_prior.to(device)
        self.running_target_prior = torch.ones(num_classes, device=device) / num_classes
        self.classwise_progress = torch.ones(num_classes, device=device)
        self.running_class_distribution = torch.ones(num_classes, device=device) / num_classes

    def _distribution_align(self, probabilities):
        if not self.enable_distribution_alignment:
            return probabilities

        batch_prior = probabilities.mean(dim=0)
        self.running_target_prior = (
            self.prior_momentum * self.running_target_prior
            + (1.0 - self.prior_momentum) * batch_prior.detach()
        )

        adjusted = probabilities * (self.source_prior / self.running_target_prior.clamp(min=1e-6))
        adjusted = adjusted / adjusted.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return adjusted

    def _classwise_thresholds(self):
        normalized_progress = self.classwise_progress / self.classwise_progress.max().clamp(min=1e-6)
        class_frequency = self.running_class_distribution / self.running_class_distribution.mean().clamp(min=1e-6)
        balance_factor = torch.sqrt(class_frequency.clamp(min=1e-6))
        thresholds = self.base_threshold * normalized_progress * balance_factor
        return thresholds.clamp(min=self.min_threshold, max=self.max_threshold)

    def generate(self, teacher_probabilities, uncertainty=None):
        if uncertainty is None:
            uncertainty = torch.zeros(teacher_probabilities.size(0), device=self.device)

        aligned_probabilities = self._distribution_align(teacher_probabilities)
        batch_distribution = aligned_probabilities.mean(dim=0).detach()
        self.running_class_distribution = (
            self.prior_momentum * self.running_class_distribution
            + (1.0 - self.prior_momentum) * batch_distribution
        )
        confidence, pseudo_labels = aligned_probabilities.max(dim=1)
        class_thresholds = self._classwise_thresholds()
        thresholds = class_thresholds[pseudo_labels]

        raw_reliability = confidence * torch.exp(-uncertainty)
        strict_mask = confidence >= thresholds
        selection_mask = strict_mask.clone()
        fallback_k = min(self.fallback_topk, confidence.numel())
        if fallback_k > 0 and selection_mask.sum() < fallback_k:
            _, topk_indices = torch.topk(confidence, k=fallback_k)
            selection_mask[topk_indices] = True

        reliability = raw_reliability * selection_mask.float()
        if selection_mask.any():
            reliability = torch.where(
                selection_mask,
                reliability.clamp(min=self.fallback_min_weight),
                reliability,
            )
        strict_reliability = raw_reliability * strict_mask.float()

        for class_idx in range(self.num_classes):
            class_mask = strict_mask & (pseudo_labels == class_idx)
            if class_mask.any():
                class_confidence = confidence[class_mask].mean().detach()
                self.classwise_progress[class_idx] = (
                    self.class_momentum * self.classwise_progress[class_idx]
                    + (1.0 - self.class_momentum) * class_confidence
                )

        selected_distribution = torch.zeros(self.num_classes, device=self.device)
        if strict_mask.any():
            selected_counts = torch.bincount(
                pseudo_labels[strict_mask],
                minlength=self.num_classes,
            ).float()
            selected_distribution = selected_counts / selected_counts.sum().clamp(min=1.0)

        return {
            'probabilities': aligned_probabilities,
            'labels': pseudo_labels,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'weights': reliability,
            'strict_weights': strict_reliability,
            'mask': selection_mask,
            'strict_mask': strict_mask,
            'thresholds': thresholds,
            'selected_distribution': selected_distribution,
        }


def build_ema_teacher(student_model):
    teacher_model = copy.deepcopy(student_model)
    for parameter in teacher_model.parameters():
        parameter.requires_grad = False
    return teacher_model


def update_ema(student_model, teacher_model, momentum=0.999):
    with torch.no_grad():
        for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
            teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)
        for teacher_buffer, student_buffer in zip(teacher_model.buffers(), student_model.buffers()):
            teacher_buffer.copy_(student_buffer)


def _enable_dropout_only(module):
    if isinstance(module, nn.Dropout):
        module.train()


@torch.no_grad()
def mc_dropout_predict(
    teacher_model,
    target_x,
    source_prototypes=None,
    mc_passes=4,
    use_ssc=True,
    prototype_blend=1.0,
):
    teacher_model.eval()
    probability_samples = []
    for _ in range(mc_passes):
        teacher_model.apply(_enable_dropout_only)
        outputs = teacher_model.forward_target(
            target_x,
            source_prototypes=source_prototypes,
            use_ssc=use_ssc,
            prototype_blend=prototype_blend,
        )
        probability_samples.append(F.softmax(outputs['logits'], dim=1))

    stacked_probabilities = torch.stack(probability_samples, dim=0)
    mean_probability = stacked_probabilities.mean(dim=0)
    uncertainty = stacked_probabilities.var(dim=0).mean(dim=1)

    teacher_model.eval()
    deterministic_outputs = teacher_model.forward_target(
        target_x,
        source_prototypes=source_prototypes,
        use_ssc=use_ssc,
        prototype_blend=prototype_blend,
    )
    return mean_probability, uncertainty, deterministic_outputs


def weighted_pseudo_ce_loss(logits, pseudo_labels, sample_weights):
    if logits.size(0) == 0:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    loss = F.cross_entropy(logits, pseudo_labels, reduction='none')
    sample_weights = sample_weights.to(logits.device).float().clamp(min=0.0)
    return (loss * sample_weights).sum() / sample_weights.sum().clamp(min=1.0)
