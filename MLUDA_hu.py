# -*- coding:utf-8 -*-
import config_Houston as cfg
from mindgap_runner import run_mindgap_experiment
import utils as mindgap_utils
from UtilsCMS import ILDA


def main():
    data_path_s = './datasets/Houston/Houston13.mat'
    label_path_s = './datasets/Houston/Houston13_7gt.mat'
    data_path_t = './datasets/Houston/Houston18.mat'
    label_path_t = './datasets/Houston/Houston18_7gt.mat'

    data_s, label_s = mindgap_utils.load_data_houston(data_path_s, label_path_s)
    data_t, label_t = mindgap_utils.load_data_houston(data_path_t, label_path_t)
    data_s, data_t = ILDA(data_s, data_t, cfg.pca_n, cfg.radius)

    cfg.inverse_lr_schedule = True
    cfg.source_samples_per_class = 180
    cfg.ssc_unfreeze_epoch = max(cfg.epochs // 3, 1)
    cfg.ssc_lr_scale = 0.1
    cfg.spectral_reg_weight = 1.0
    cfg.teacher_momentum = 0.99
    cfg.prototype_momentum = 0.9
    cfg.base_threshold = 0.92
    cfg.min_threshold = 0.75
    cfg.max_threshold = 0.98
    cfg.distribution_align_momentum = 0.9
    cfg.class_threshold_momentum = 0.9
    cfg.mc_dropout_passes = 4
    cfg.pseudo_loss_weight = 0.20
    cfg.target_contrastive_weight = 0.02
    cfg.lmmd_weight = 0.01
    cfg.ssc_guide_eps = 1e-2
    cfg.target_warmup_epochs = max(cfg.epochs // 3, 1)
    cfg.target_ramp_epochs = 10
    cfg.prototype_attention_start_epoch = max((cfg.epochs * 8) // 10, 1)
    cfg.target_contrastive_start_epoch = max((cfg.epochs * 9) // 10, 1)
    cfg.prototype_attention_ramp_epochs = 20
    cfg.prototype_attention_max_scale = 0.20
    cfg.target_contrastive_min_weight = 0.90
    cfg.target_contrastive_min_count = max(cfg.CLASS_NUM, 8)
    cfg.pseudo_fallback_topk = 0
    cfg.pseudo_min_weight = 0.0
    cfg.enable_distribution_alignment = False
    cfg.eval_interval = 5

    run_mindgap_experiment(data_s, label_s, data_t, label_t, cfg)


if __name__ == '__main__':
    main()
