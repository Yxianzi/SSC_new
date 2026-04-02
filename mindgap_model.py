# -*- coding:utf-8 -*-
"""Original MLUDA backbone/classifier wrapped with the new add-on modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ssc_module import SpectralStyleCalibration


class PrototypeGuidedCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super(PrototypeGuidedCrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens, prototypes, blend=1.0):
        if prototypes is None or prototypes.numel() == 0 or blend <= 0.0:
            return tokens

        prototype_tokens = prototypes.unsqueeze(0).expand(tokens.size(0), -1, -1)
        aligned_tokens, _ = self.attention(tokens, prototype_tokens, prototype_tokens)
        fused_tokens = self.fusion(torch.cat([tokens, aligned_tokens], dim=-1))
        refined_tokens = self.norm(tokens + fused_tokens)
        if blend >= 1.0:
            return refined_tokens
        return tokens + blend * (refined_tokens - tokens)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        hidden_planes = max(in_planes // ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden_planes, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv1(attention)
        return self.sigmoid(attention)


class MLUDAFeatureExtractor(nn.Module):
    """Single-branch feature extractor restored from the original MLUDA backbone."""

    def __init__(self, input_channels, patch_size):
        super(MLUDAFeatureExtractor, self).__init__()
        self.feature_dim = input_channels
        self.patch_size = patch_size
        self.output_dim = 192 + 96

        self.conv1 = nn.Conv3d(1, 24, kernel_size=(7, 1, 1), stride=(2, 1, 1), bias=True)
        self.bn1 = nn.BatchNorm3d(24)
        self.activation1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), padding=(3, 0, 0), bias=True)
        self.bn2 = nn.BatchNorm3d(24)
        self.activation2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), padding=(3, 0, 0), bias=True)
        self.bn3 = nn.BatchNorm3d(24)
        self.activation3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv3d(
            24,
            192,
            kernel_size=(((self.feature_dim - 7) // 2 + 1), 1, 1),
            bias=True,
        )
        self.bn4 = nn.BatchNorm3d(192)
        self.activation4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv3d(1, 24, (self.feature_dim, 1, 1), bias=True)
        self.bn5 = nn.BatchNorm3d(24)
        self.activation5 = nn.ReLU(inplace=True)

        self.conv6 = nn.Conv3d(24, 24, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=True)
        self.bn6 = nn.BatchNorm3d(24)
        self.activation6 = nn.ReLU(inplace=True)

        self.conv7 = nn.Conv3d(24, 96, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=True)
        self.bn7 = nn.BatchNorm3d(96)
        self.activation7 = nn.ReLU(inplace=True)
        self.conv8 = nn.Conv3d(24, 96, kernel_size=1, bias=True)

        self.ca = ChannelAttention(self.output_dim)
        self.sa = SpatialAttention()
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self._reset_parameters()

    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _forward_branch(self, x):
        x = x.unsqueeze(1)

        spectral = self.conv1(x)
        spectral = self.activation1(self.bn1(spectral))
        spectral_residual = spectral
        spectral = self.conv2(spectral)
        spectral = self.activation2(self.bn2(spectral))
        spectral = self.conv3(spectral)
        spectral = spectral_residual + spectral
        spectral = self.activation3(self.bn3(spectral))
        spectral = self.conv4(spectral)
        spectral = self.activation4(self.bn4(spectral))
        spectral = spectral.reshape(spectral.size(0), spectral.size(1), spectral.size(3), spectral.size(4))

        spatial = self.conv5(x)
        spatial = self.activation5(self.bn5(spatial))
        spatial_residual = self.conv8(spatial)
        spatial = self.conv6(spatial)
        spatial = self.activation6(self.bn6(spatial))
        spatial = self.conv7(spatial)
        spatial = spatial_residual + spatial
        spatial = self.activation7(self.bn7(spatial))
        spatial = spatial.reshape(spatial.size(0), spatial.size(1), spatial.size(3), spatial.size(4))

        feature_map = torch.cat([spectral, spatial], dim=1)
        feature_map = self.ca(feature_map) * feature_map
        feature_map = self.sa(feature_map) * feature_map
        return feature_map

    def forward_features(self, x):
        feature_map = self._forward_branch(x)
        tokens = feature_map.flatten(2).transpose(1, 2).contiguous()
        pooled = self.avgpool(feature_map).flatten(1)
        return {
            'feature_map': feature_map,
            'tokens': tokens,
            'pooled': pooled,
        }


class MindGapModel(nn.Module):
    def __init__(self, n_band=198, patch_size=3, num_class=3, guide_eps=1e-2):
        super(MindGapModel, self).__init__()
        self.num_class = num_class

        self.ssc = SpectralStyleCalibration(n_band, hidden_dim=max(128, n_band), guide_eps=guide_eps)
        self.backbone = MLUDAFeatureExtractor(n_band, patch_size)
        self.feature_dim = self.backbone.output_dim
        self.prototype_attention = PrototypeGuidedCrossAttention(self.feature_dim, num_heads=4, dropout=0.1)

        # Restore the original MLUDA-style heads, while keeping a small dropout
        # wrapper for MC-dropout uncertainty estimation in the add-on pipeline.
        self.classifier = nn.Linear(self.feature_dim, num_class)
        self.domain_classifier = nn.Linear(self.feature_dim, 1)
        self.projection_head = nn.Linear(self.feature_dim, 128)
        self.projection_head_aux = nn.Linear(self.feature_dim, 128)
        self.dropout = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()

        self._reset_output_heads()

    def _reset_output_heads(self):
        for module in (
            self.classifier,
            self.domain_classifier,
            self.projection_head,
            self.projection_head_aux,
        ):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _pool_tokens(self, tokens):
        return tokens.mean(dim=1)

    def _build_outputs(self, tokens, backbone_tokens=None):
        if backbone_tokens is None:
            backbone_tokens = tokens

        pooled = self._pool_tokens(tokens)
        backbone_pooled = self._pool_tokens(backbone_tokens)
        dropped_features = self.dropout(pooled)
        logits = self.classifier(dropped_features)
        projection = F.normalize(self.projection_head(dropped_features), dim=1)
        projection_aux = F.normalize(self.projection_head_aux(dropped_features), dim=1)
        domain_score = self.sigmoid(self.domain_classifier(dropped_features))

        return {
            'tokens': tokens,
            'features': pooled,
            'backbone_tokens': backbone_tokens,
            'backbone_features': backbone_pooled,
            'projection': projection,
            'projection_aux': projection_aux,
            'logits': logits,
            'domain_score': domain_score,
        }

    def forward_source(
        self,
        source_x,
        target_x,
        target_prototypes=None,
        use_ssc=True,
        use_prototype_attention=False,
        prototype_blend=1.0,
    ):
        filtered_source, spectral_reg, calibrated_source, target_norm = self.ssc.forward_source(
            source_x,
            target_x,
            use_calibration=use_ssc,
        )
        backbone_outputs = self.backbone.forward_features(filtered_source)
        source_tokens = backbone_outputs['tokens']
        if use_prototype_attention:
            source_tokens = self.prototype_attention(
                source_tokens,
                target_prototypes,
                blend=prototype_blend,
            )

        outputs = self._build_outputs(source_tokens, backbone_tokens=backbone_outputs['tokens'])
        outputs['feature_map'] = backbone_outputs['feature_map']
        outputs['spectral_reg'] = spectral_reg
        outputs['calibrated_source'] = calibrated_source
        outputs['filtered_source'] = filtered_source
        outputs['target_norm'] = target_norm
        return outputs

    def forward_target(
        self,
        target_x,
        source_prototypes=None,
        use_ssc=True,
        use_prototype_attention=True,
        prototype_blend=1.0,
    ):
        filtered_target = self.ssc.forward_target(target_x, use_calibration=use_ssc)
        backbone_outputs = self.backbone.forward_features(filtered_target)
        target_tokens = backbone_outputs['tokens']
        if use_prototype_attention:
            target_tokens = self.prototype_attention(
                target_tokens,
                source_prototypes,
                blend=prototype_blend,
            )

        outputs = self._build_outputs(target_tokens, backbone_tokens=backbone_outputs['tokens'])
        outputs['feature_map'] = backbone_outputs['feature_map']
        outputs['filtered_target'] = filtered_target
        return outputs

    def predict(self, x):
        outputs = self.forward_target(x, source_prototypes=None, use_ssc=True)
        return outputs['features'], outputs['logits']

    def get_embedding(self, x):
        outputs = self.forward_target(x, source_prototypes=None, use_ssc=True)
        return outputs['features']
