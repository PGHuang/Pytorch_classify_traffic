import sys

sys.path.append("../")

import os
import random
import torch
import torchvision
import numpy as np
from PIL import Image
import torch.nn as nn
from torchvision import transforms

from datasets import traffic_dataset
from models.load_models import load_net

from torchcv.utils_project import timer, logger, util_visdom
from torchcv.optim_lrscheduler.get_optim_lr_scheduler import get_optimizer, get_lr_scheduler
from torchcv.utils_vis.gen_vis_res import gen_res_compare


class Trainer(object):
    def __init__(self, cfg):
        self.cfg = cfg

        """ make reproducible """
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        random.seed(self.cfg.seed)
        torch.backends.cudnn.deterministic = True  # 保证实验的可重复性
        # select fast cudnn kernels
        torch.backends.cudnn.benchmark = True

        """ device """
        cuda_setting = "cuda" if len(self.cfg.gpus) == 0 else "cuda:" + str(self.cfg.gpus[0])
        self.device = torch.device(cuda_setting if torch.cuda.is_available() else "cpu")

        """ timer """
        self.timer = timer.Timer()

        """ visdom """
        self.vis = util_visdom.Visualizer(env=self.cfg.exp_name)

    def build_model(self):
        ##############################################################################################
        #                                       Config Information                                   #
        ##############################################################################################
        self.vis.text(win_name='configurations', content=self.cfg.to_dict())
        sys.stdout = logger.Logger(self.cfg.path_log_save)  # print同时输出到log文件
        self.cfg.to_json(self.cfg.path_save + "HyperParameters.json")
        self.cfg.to_show()

        ##############################################################################################
        #                                       DataLoader                                           #
        ##############################################################################################
        self.loader_train = traffic_dataset.get_loader_train(self.cfg.batch_size,
                                                             path_folder=self.cfg.path_train_folder,
                                                             path_anno_txt=self.cfg.path_train_anno)
        self.loader_val = traffic_dataset.get_loader_val(self.cfg.batch_size,
                                                         path_folder=self.cfg.path_test_folder,
                                                         path_anno_txt=self.cfg.path_test_anno)

        ##############################################################################################
        #                                          Model                                             #
        ##############################################################################################
        self.net = load_net(self.cfg)

        """ 预加载权重 """
        if not self.cfg.model.finetune_weight == None:
            self.net.load_state_dict(torch.load(self.cfg.model.finetune_weight, map_location="cpu"))

        """ setup GPU cfg """
        if len(self.cfg.gpus) > 1:
            self.net = nn.DataParallel(self.net, device_ids=self.cfg.gpus)
        self.net = self.net.to(self.device)

        ##############################################################################################
        #                                           Loss                                             #
        ##############################################################################################
        # self.criterion_bce = nn.BCELoss(reduction='mean').to(self.device)
        self.criterion_cross_entropy = nn.CrossEntropyLoss().to(self.device)

        ##############################################################################################
        #                                 optimizer and lr_scheduler                                 #
        ##############################################################################################
        self.optimizer = get_optimizer(self.net.parameters(), self.cfg.optimizer.init_lr, self.cfg)
        self.scheduler = get_lr_scheduler(self.optimizer, self.cfg)

    def training(self, epoch):
        # display learning_rate
        print("Epoch:%-4i" % (epoch), "learning_rate =", self.optimizer.param_groups[0]['lr'])
        self.vis.line(win_name='lr', x=epoch, value=self.optimizer.param_groups[0]['lr'])

        # 训练一个 epoch
        train_epoch_loss = 0.0
        self.net.train()
        confusion_matrix = torch.zeros((3, 3)).to(self.device)  # 混淆矩阵
        for batch_id, (images, targets) in enumerate(self.loader_train):
            images, targets = images.to(self.device), targets.to(self.device)
            predicts = self.net(images)

            """ Loss """
            loss = self._compute_loss(predict=predicts, target=targets)
            train_epoch_loss += loss.item()

            """ Backward """
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            """ metric """
            predicts_softmax = predicts.softmax(dim=-1)
            pre_cls = predicts_softmax.argmax(dim=-1)
            for p, t in zip(pre_cls.view(-1), targets.view(-1)):
                confusion_matrix[p.long(), t.long()] += 1

            """ 可视化训练过程 """
            if batch_id % self.cfg.freq_print == 0:
                print("Epoch:%-7i" % epoch, "Batch_ID: %-5i" % batch_id, " loss = %-5.4f" % loss.item())
                res_vis = gen_res_compare(images, targets, predicts)
                self.vis.image(win_name='train_sample', imgs=res_vis, nrow=self.cfg.vis_nrows)

        """ Epoch_Loss """
        print("Epoch:%-5i" % epoch, "Epoch_Loss = %-5.5f" % train_epoch_loss)
        self.vis.line(win_name='train_epoch_loss', x=epoch, value=train_epoch_loss)

        """ Metric """
        eps = 1e-8
        precision_each = confusion_matrix.diag() / (confusion_matrix.sum(1) + eps)
        recall_each = confusion_matrix.diag() / (confusion_matrix.sum(0) + eps)
        f1_each = 2 * precision_each * recall_each / (precision_each + recall_each + eps)

        precision_mean = torch.mean(precision_each)
        recall_mean = torch.mean(recall_each)
        f1_score = 0.2 * f1_each[0] + 0.2 * f1_each[1] + 0.6 * f1_each[2]

        print("Epoch:%-5i" % epoch, "precision_mean = %-3.3f" % precision_mean.item(),
              "   recall_mean = %-3.3f" % recall_mean.item(), "   f1_score = %-3.3f" % f1_score.item())
        self.vis.line(win_name='train_metric', x=epoch, value={'precision_mean': precision_mean.item(),
                                                               'recall_mean': recall_mean.item(),
                                                               'f1_score': f1_score.item()})

    def _compute_loss(self, predict, target):
        loss_cross_entropy = self.criterion_cross_entropy(predict, target)
        return loss_cross_entropy

    def validation(self, epoch):
        if_first_batch = True  # 仅可视化第一个batch分割效果
        self.net.eval()
        with torch.no_grad():
            for images, targets in self.loader_val:
                images, targets = images.to(self.device), targets.to(self.device)
                predicts = self.net(images)

                """ visdom """
                if if_first_batch:
                    res_vis = gen_res_compare(images, targets, predicts)
                    self.vis.image(win_name='val_sample', imgs=res_vis, nrow=self.cfg.vis_nrows)
                    # save res_img
                    path_img_save = self.cfg.path_save_IMG + "val_epoch-%s.jpg" % (str(epoch).zfill(4))
                    torchvision.utils.save_image(res_vis, path_img_save, nrow=1)
                    if_first_batch = False

                """ Metric """
                pass

                break

    def save_model(self, epoch):
        if_parallel = len(self.cfg.gpus) > 1
        path_save = self.cfg.path_save_ckpt + "epoch_" + str(epoch).zfill(4) + ".pth"
        if if_parallel:
            torch.save(self.net.module.state_dict(), path_save)
        else:
            torch.save(self.net.state_dict(), path_save)


def main():
    from config import cfg
    trainer = Trainer(cfg)
    trainer.build_model()
    for epoch in range(1, cfg.epochs + 1):
        # 训练集
        trainer.training(epoch)
        # 更新学习率
        trainer.scheduler.step()
        # 验证集
        if epoch % cfg.freq_val == 0:
            trainer.validation(epoch)
        # 保存模型参数
        if epoch % cfg.freq_save_pth == 0:
            trainer.save_model(epoch)
        # 输出运行时间
        trainer.timer.print_time_info()


if __name__ == '__main__':
    main()
