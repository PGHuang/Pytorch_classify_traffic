import sys

sys.path.append("../")

import os
import torch
from torchvision import transforms
from PIL import Image

import logging
import datetime
import json
from tqdm import tqdm
from easydict import EasyDict as edict


class Testor(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ################################################################################
        #                                  Model                                       #
        ################################################################################
        self.net = load_net(self.cfg)
        state_dict = torch.load(self.cfg.path_pth, map_location="cpu")
        self.net.load_state_dict(state_dict)
        self.net = self.net.to(self.device)
        self.net.eval()

        ################################################################################
        #                            anno_                                  #
        ################################################################################
        anno_test_dict = json.load(open(self.cfg.path_test_annofile_labeled, "r"))
        self.anno_list = anno_test_dict["annotations"]

        ################################################################################
        #                            confusion_matrix                                  #
        ################################################################################
        self.confusion_matrix = torch.zeros((3, 3)).to(self.device)

    def predict(self, path_img):
        ################################################################################
        #                                  load IMG                                    #
        ################################################################################
        img_pil = Image.open(path_img).convert("RGB").resize((self.cfg.inp_size[1], self.cfg.inp_size[0]))
        # img_pil.show()
        img_tensor = transforms.ToTensor()(img_pil)
        img_tensor = transforms.Normalize(mean=self.cfg.mean_rgb, std=self.cfg.std_rgb)(img_tensor)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        # print(img_tensor.shape)

        ################################################################################
        #                                   forward                                    #
        ################################################################################
        predicts = self.net(img_tensor)
        predicts_softmax = predicts.softmax(dim=-1)
        pre_cls = predicts_softmax.argmax(dim=-1)
        return pre_cls.item()

    def compute_confusion_matrix(self):
        for item in tqdm(self.anno_list):
            # for item in self.anno_list:
            sample_id = item['id']
            sample_label = item['status']
            path_img = self.cfg.path_folder_testdata + str(sample_id) + "/" + item['key_frame']
            res_cls = self.predict(path_img)
            self.confusion_matrix[res_cls, sample_label] += 1


def load_net(cfg):
    from models.resnet50 import resnet50
    from models.resnet18_tv import Resnet18_tv

    if cfg.model_arch == "resnet50":
        model = resnet50(**cfg.model_conf)
    elif cfg.model_arch == "resnet18_tv_pretrain":
        model = Resnet18_tv(**cfg.model_conf)
    elif cfg.model_arch == "resnet18_tv_woPretrain":
        model = Resnet18_tv(**cfg.model_conf)
    else:
        logging.critical(f"Model type {cfg.model_arch} not defined!")
        sys.exit()

    return model


def get_model_config(arch):
    conf = edict()
    if arch == "resnet50":
        conf.num_classes = 3
    elif arch == "resnet18_tv_pretrain":
        conf.num_classes = 3
        conf.if_pretrained = True
    elif arch == "resnet18_tv_woPretrain":
        conf.num_classes = 3
        conf.if_pretrained = False
    else:
        logging.critical(f"Model type {arch} not defined!")
        sys.exit()
    return conf


def main():
    ##############################################################
    #                             cfg                            #
    ##############################################################
    cfg = edict()
    # model
    cfg.model_arch = "resnet18_tv_pretrain"
    cfg.model_conf = get_model_config(cfg.model_arch)

    # transform
    cfg.inp_size = (360, 640)  # (h, w)
    cfg.mean_rgb = [0.485, 0.456, 0.406]  # [0.485, 0.456, 0.406]
    cfg.std_rgb = [0.229, 0.224, 0.225]  # [0.229, 0.224, 0.225]

    # path IMG and json
    cfg.path_test_annofile_labeled = "/mnt/data1/huangpg/TianChi/traffic/commit_json/" \
                                     "amap_traffic_annotations_test_labeled_v1.json"
    cfg.path_folder_testdata = "/mnt/data1/huangpg/TianChi/traffic/data/amap_traffic_test_0712/"

    ##############################################################
    #                        compute F1_score                    #
    ##############################################################
    path_folder_pth = "/mnt/data1/huangpg/TianChi/traffic/Pytorch_classify_traffic/checkpoints/" \
                      "traffic_run02_resnet18_tv_pretrain/checkpoints/"
    pth_files = os.listdir(path_folder_pth)
    pth_files.sort(reverse=True)
    for file in pth_files:
        cfg.path_pth = path_folder_pth + file

        ##############################################################
        #                            testor                          #
        ##############################################################
        testor = Testor(cfg)
        testor.compute_confusion_matrix()
        confusion_matrix = testor.confusion_matrix

        """ metric """
        eps = 1e-8
        precision_each = confusion_matrix.diag() / (confusion_matrix.sum(1) + eps)
        recall_each = confusion_matrix.diag() / (confusion_matrix.sum(0) + eps)
        f1_each = 2 * precision_each * recall_each / (precision_each + recall_each + eps)

        precision_mean = torch.mean(precision_each)
        recall_mean = torch.mean(recall_each)
        f1_score = 0.2 * f1_each[0] + 0.2 * f1_each[1] + 0.6 * f1_each[2]

        print(file + ":  F1-Score =", f1_score.item())


if __name__ == '__main__':
    main()
