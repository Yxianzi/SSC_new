# -*- coding:utf-8 -*-
"""Lightweight ILDA preprocessing without OpenCV dependencies."""

import numpy as np
from scipy.ndimage import uniform_filter
from sklearn.decomposition import PCA
from skimage import exposure


def _minmax_normalize(image):
    image = image.astype(np.float32, copy=False)
    min_value = np.min(image, axis=(0, 1), keepdims=True)
    max_value = np.max(image, axis=(0, 1), keepdims=True)
    scale = np.maximum(max_value - min_value, 1e-6)
    return (image - min_value) / scale


def pca(data, n_components):
    height, width, channels = data.shape
    data_2d = data.reshape(-1, channels)
    reduced = PCA(n_components=n_components).fit_transform(data_2d).reshape(height, width, n_components)
    return _minmax_normalize(reduced)


def gamma_correct(source_image, target_image):
    source_mean = float(np.mean(source_image))
    target_mean = float(np.mean(target_image))
    gamma_value = abs(source_mean / max(target_mean, 1e-6))

    corrected_source = np.power(np.clip(source_image, 0.0, None), gamma_value).astype(np.float32)
    corrected_target = np.clip(target_image, 0.0, 1.0).astype(np.float32)
    return corrected_source, corrected_target


def color_adaption(source_image, target_image):
    matched = exposure.match_histograms(source_image, target_image, channel_axis=-1)
    return matched.astype(np.float32), target_image.astype(np.float32)


def _guided_filter_single_channel(guide, src, radius=1, eps=1e-2):
    window = radius * 2 + 1

    mean_guide = uniform_filter(guide, size=window, mode='nearest')
    mean_src = uniform_filter(src, size=window, mode='nearest')
    corr_guide = uniform_filter(guide * guide, size=window, mode='nearest')
    corr_guide_src = uniform_filter(guide * src, size=window, mode='nearest')

    var_guide = corr_guide - mean_guide * mean_guide
    cov_guide_src = corr_guide_src - mean_guide * mean_src

    a = cov_guide_src / (var_guide + eps)
    b = mean_src - a * mean_guide

    mean_a = uniform_filter(a, size=window, mode='nearest')
    mean_b = uniform_filter(b, size=window, mode='nearest')
    return mean_a * guide + mean_b


def guided_filter_cube(cube, guide_rgb, eps=1e-2, radius=1):
    guide = np.mean(guide_rgb.astype(np.float32), axis=2)
    filtered_bands = [
        _guided_filter_single_channel(guide, cube[:, :, band].astype(np.float32), radius=radius, eps=eps)
        for band in range(cube.shape[2])
    ]
    filtered_cube = np.stack(filtered_bands, axis=2)
    return filtered_cube.astype(np.float32)


def total_adaption(source_cube, target_cube, pca_n, eps):
    source_pca = pca(source_cube, pca_n)
    target_pca = pca(target_cube, pca_n)

    source_gamma, target_gamma = gamma_correct(source_pca, target_pca)
    source_color, target_color = color_adaption(source_gamma, target_gamma)

    final_source = guided_filter_cube(source_cube, source_color, eps=eps, radius=1)
    final_target = guided_filter_cube(target_cube, target_color, eps=eps, radius=1)
    return final_source, final_target


def ILDA(data_s, data_t, pca_n, r):
    source, target = total_adaption(data_s, data_t, pca_n, r)
    return source.astype(np.float32), target.astype(np.float32)
