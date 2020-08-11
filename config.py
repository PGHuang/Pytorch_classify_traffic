import os
import sys
import logging
from easydict import EasyDict as edict
from torchcv.utils_project.config_base import ConfigBase


# setup network cfg
def get_model_config(arch):
    conf = edict()
    if arch == "resnet50":
        conf.num_classes = 3
    else:
        logging.critical(f"Model type {arch} not defined!")
        sys.exit()
    return conf


# setup optim cfg
def get_optimizer_conf(optimizer_type):
    conf = edict()
    if optimizer_type == "Adam":
        conf.betas = (0.9, 0.999)
        conf.eps = 1e-8
        conf.weight_decay = 1e-4
        conf.amsgrad = False
    elif optimizer_type == "SGD":
        conf.momentum = 0.9
        conf.weight_decay = 1e-4
        conf.nesterov = True
    elif optimizer_type == "Adadelta":
        conf.rho = 0.95
        conf.eps = 1e-07
        conf.weight_decay = 2e-5
    else:
        logging.critical(f"Optimizer type {optimizer_type} not defined!")
        sys.exit()
    return conf


# setup lr_scheduler cfg
def get_lr_scheduler_conf(scheduler, epochs):
    conf = edict()
    if scheduler == "WarmupCosineLR":
        conf.T_max = epochs
        conf.warmup_size = 10
        conf.start_factor = 0.25
    elif scheduler == "ReduceLROnPlateau":
        conf.mode = "min"
        conf.factor = 0.5
        conf.patience = 10  # 阈值
        conf.min_lr = 1e-6
    elif scheduler == "MultiStepLR":
        conf.milestones = [100, 200, 300]
        conf.gamma = 0.5
    return conf


class Config(ConfigBase):
    ##########################################################################################
    #                                   experiment setting                                   #
    ##########################################################################################
    exp_name = "traffic_run01_baseline"
    exp_description = "resnet50, official_data"

    ##########################################################################################
    #                                         data                                           #
    ##########################################################################################
    data_description = "official data"
    path_train_folder = "/mnt/data1/huangpg/TianChi/traffic/data/amap_traffic_train_0712/"
    path_test_folder = "/mnt/data1/huangpg/TianChi/traffic/data/amap_traffic_test_0712/"
    path_train_anno = "/mnt/data1/huangpg/TianChi/traffic/data/amap_traffic_annotations_train.json"
    path_test_anno = "/mnt/data1/huangpg/TianChi/traffic/data/amap_traffic_annotations_test.json"

    ##########################################################################################
    #                                    hyper-parameters                                    #
    ##########################################################################################
    path_project = "/mnt/data1/huangpg/TianChi/traffic/Pytorch_classify_traffic/"

    seed = 1
    gpus = [0]
    epochs = 500
    batch_size = 16

    # 显示参数设置
    vis_nrows = 4
    freq_print = 10  # every batch

    freq_val = 10  # every epoch
    freq_save_pth = 10  # every epoch

    ##########################################################################################
    #                                      transforms                                        #
    ##########################################################################################
    trans = edict()
    trans.inp_size = (360, 640)  # (h, w)
    trans.mean_rgb = [0, 0, 0]
    trans.std_rgb = [1, 1, 1]

    # 训练集数据增强
    trans.hflip_p = 0.5
    trans.color_jitter_p = 0.01
    trans.noise_p = 0.01
    trans.blur_p = 0.01
    trans.point_light_p = 0.01

    num_workers = 4
    pin_memory = True

    ##########################################################################################
    #                                           model                                        #
    ##########################################################################################
    model = edict()
    model.arch = "resnet50"
    model.conf = get_model_config(model.arch)
    model.finetune_weight = None

    ##########################################################################################
    #                                  optimizer and lr_scheduler                            #
    ##########################################################################################
    optimizer = edict()
    optimizer.type = "Adam"  # [Adam, SGD, Adadelta]
    optimizer.init_lr = 0.002
    optimizer.conf = get_optimizer_conf(optimizer.type)

    lr_scheduler = edict()
    lr_scheduler.type = "WarmupCosineLR"  # [WarmupCosineLR, ReduceLROnPlateau, MultiStepLR]
    lr_scheduler.conf = get_lr_scheduler_conf(lr_scheduler.type, epochs)

    ##########################################################################################
    #                                        checkpoint                                      #
    ##########################################################################################
    path_save = path_project + "checkpoints/" + exp_name + "/"
    path_save_IMG = path_save + "IMGs/"
    path_save_ckpt = path_save + "checkpoints/"
    path_log_save = f"{path_save}train_{exp_name}.log"

    if not os.path.exists(path_save):
        os.makedirs(path_save_IMG)
        os.makedirs(path_save_ckpt)


cfg = Config()

if __name__ == '__main__':
    cfg.to_show()
