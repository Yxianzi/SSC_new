# -*- coding:utf-8 -*-
"""Spectral Style Calibration with differentiable guided filtering."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralStyleCalibration(nn.Module):
    def __init__(self, num_bands, hidden_dim=256, guide_radius=1, guide_eps=1e-2):
        super(SpectralStyleCalibration, self).__init__()
        self.num_bands = num_bands
        self.guide_radius = guide_radius
        self.guide_eps = guide_eps
        self.register_buffer('source_mean', torch.zeros(1, num_bands, 1, 1))
        self.register_buffer('source_std', torch.ones(1, num_bands, 1, 1))
        self.register_buffer('target_mean', torch.zeros(1, num_bands, 1, 1))
        self.register_buffer('target_std', torch.ones(1, num_bands, 1, 1))
        self.register_buffer('source_stats_vector', torch.zeros(1, num_bands * 5))
        self.register_buffer('target_stats_vector', torch.zeros(1, num_bands * 5))
        self.register_buffer('has_global_stats', torch.tensor(False, dtype=torch.bool))

        self.condition_net = nn.Sequential(
            nn.Linear(num_bands * 5, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_bands * 2),
        )
        self._init_identity()

    def _init_identity(self):
        final_linear = self.condition_net[-1]
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

    def _flatten_domain_cube(self, cube):
        if isinstance(cube, torch.Tensor):
            array = cube.detach().float().cpu()
        else:
            array = torch.as_tensor(cube, dtype=torch.float32)

        if array.dim() != 3:
            raise ValueError('Expected a 3D HSI cube with shape [H, W, C].')

        if array.shape[-1] == self.num_bands:
            flattened = array.reshape(-1, self.num_bands)
        elif array.shape[0] == self.num_bands:
            flattened = array.permute(1, 2, 0).reshape(-1, self.num_bands)
        else:
            raise ValueError('Unable to infer spectral band dimension for SSC statistics.')

        return flattened

    def _build_stats_from_flattened(self, flattened):
        mean = flattened.mean(dim=0).view(1, self.num_bands, 1, 1)
        std = flattened.std(dim=0, unbiased=False).clamp(min=1e-5).view(1, self.num_bands, 1, 1)
        q25 = torch.quantile(flattened, 0.25, dim=0)
        q50 = torch.quantile(flattened, 0.50, dim=0)
        q75 = torch.quantile(flattened, 0.75, dim=0)
        stats_vector = torch.cat([
            mean.view(-1),
            std.view(-1),
            q25,
            q50,
            q75,
        ], dim=0).unsqueeze(0)
        return mean, std, stats_vector

    def set_domain_statistics(self, source_cube, target_cube):
        source_flattened = self._flatten_domain_cube(source_cube)
        target_flattened = self._flatten_domain_cube(target_cube)

        source_mean, source_std, source_stats_vector = self._build_stats_from_flattened(source_flattened)
        target_mean, target_std, target_stats_vector = self._build_stats_from_flattened(target_flattened)

        self.source_mean.copy_(source_mean)
        self.source_std.copy_(source_std)
        self.target_mean.copy_(target_mean)
        self.target_std.copy_(target_std)
        self.source_stats_vector.copy_(source_stats_vector)
        self.target_stats_vector.copy_(target_stats_vector)
        self.has_global_stats.fill_(True)

    def _domain_standardize(self, x, domain=None):
        if bool(self.has_global_stats.item()) and domain in ('source', 'target'):
            if domain == 'source':
                mean = self.source_mean
                std = self.source_std
            else:
                mean = self.target_mean
                std = self.target_std
        else:
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            std = x.std(dim=(0, 2, 3), keepdim=True, unbiased=False).clamp(min=1e-5)
        normalized = (x - mean) / std
        return normalized, mean, std

    def _domain_statistics(self, x, domain=None):
        if bool(self.has_global_stats.item()) and domain in ('source', 'target'):
            if domain == 'source':
                return self.source_stats_vector
            return self.target_stats_vector

        flattened = x.permute(1, 0, 2, 3).reshape(x.size(1), -1)
        mean = flattened.mean(dim=1)
        std = flattened.std(dim=1, unbiased=False).clamp(min=1e-5)
        q25 = torch.quantile(flattened, 0.25, dim=1)
        q50 = torch.quantile(flattened, 0.50, dim=1)
        q75 = torch.quantile(flattened, 0.75, dim=1)
        return torch.cat([mean, std, q25, q50, q75], dim=0).unsqueeze(0)

    def _box_filter(self, x, radius):
        kernel_size = radius * 2 + 1
        return F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=radius)

    def _guided_filter(self, guide, src):
        mean_guide = self._box_filter(guide, self.guide_radius)
        mean_src = self._box_filter(src, self.guide_radius)
        corr_guide = self._box_filter(guide * guide, self.guide_radius)
        corr_guide_src = self._box_filter(guide * src, self.guide_radius)

        var_guide = corr_guide - mean_guide * mean_guide
        cov_guide_src = corr_guide_src - mean_guide * mean_src

        a = cov_guide_src / (var_guide + self.guide_eps)
        b = mean_src - a * mean_guide

        mean_a = self._box_filter(a, self.guide_radius)
        mean_b = self._box_filter(b, self.guide_radius)
        return mean_a * guide + mean_b

    def spectral_shape_regularization(self, original, calibrated):
        original_flat = original.permute(0, 2, 3, 1).reshape(-1, original.size(1))
        calibrated_flat = calibrated.permute(0, 2, 3, 1).reshape(-1, calibrated.size(1))

        cosine = F.cosine_similarity(original_flat, calibrated_flat, dim=1).clamp(-1 + 1e-6, 1 - 1e-6)
        spectral_angle = torch.acos(cosine).mean()

        original_diff = original_flat[:, 1:] - original_flat[:, :-1]
        calibrated_diff = calibrated_flat[:, 1:] - calibrated_flat[:, :-1]
        spectral_gradient = F.l1_loss(calibrated_diff, original_diff)
        return spectral_angle + spectral_gradient

    def _predict_affine(self, target_norm, domain='target'):
        target_stats = self._domain_statistics(target_norm, domain=domain)
        affine_params = self.condition_net(target_stats).view(1, 2, self.num_bands)
        raw_alpha = affine_params[:, 0].view(1, self.num_bands, 1, 1)
        beta = affine_params[:, 1].view(1, self.num_bands, 1, 1)

        # Identity-preserving parameterization:
        # raw_alpha = 0 -> alpha = 1, keeping the frozen stage stable.
        alpha = torch.exp(0.1 * torch.tanh(raw_alpha))
        return alpha, beta

    def forward_source(self, source_x, target_x, use_calibration=True):
        source_norm, _, _ = self._domain_standardize(source_x, domain='source')
        target_norm, _, _ = self._domain_standardize(target_x, domain='target')

        if use_calibration:
            alpha, beta = self._predict_affine(target_norm, domain='target')
            calibrated_source = alpha * source_norm + beta
            filtered_source = self._guided_filter(calibrated_source, source_norm)
            spectral_reg = self.spectral_shape_regularization(source_norm, filtered_source)
        else:
            calibrated_source = source_norm
            filtered_source = source_norm
            spectral_reg = torch.zeros((), device=source_x.device, dtype=source_x.dtype)

        return filtered_source, spectral_reg, calibrated_source, target_norm

    def forward_target(self, target_x, use_calibration=True):
        target_norm, _, _ = self._domain_standardize(target_x, domain='target')
        if use_calibration:
            filtered_target = self._guided_filter(target_norm, target_norm)
        else:
            filtered_target = target_norm
        return filtered_target
