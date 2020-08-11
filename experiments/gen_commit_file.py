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


class Predictor(object):
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

    def gen_commit_json_file(self):
        anno_test_dict = json.load(open(self.cfg.path_test_annofile, "r"))
        anno_list = anno_test_dict["annotations"]
        result_list = []
        for item in tqdm(anno_list):
            sample_id = item['id']
            path_img = self.cfg.path_folder_testdata + str(sample_id) + "/" + item['key_frame']
            res_cls = self.predict(path_img)
            item['status'] = res_cls
            result_list.append(item)

        with open(self.cfg.path_save_json, "w", encoding="utf-8") as w:
            anno_test_dict["annotations"] = result_list
            json.dump(anno_test_dict, w, indent="    ")

        print("* Generate", self.cfg.path_save_json.split("/")[-1], "Done!")


def load_net(cfg):
    from models.resnet50 import resnet50

    if cfg.model_arch == "resnet50":
        model = resnet50(**cfg.model_conf)
    else:
        logging.critical(f"Model type {cfg.model_arch} not defined!")
        sys.exit()

    return model


def get_model_config(arch):
    conf = edict()
    if arch == "resnet50":
        conf.num_classes = 3
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
    cfg.path_pth = "/mnt/data1/huangpg/TianChi/traffic/Pytorch_classify_traffic/checkpoints/" \
                   "traffic_run01_baseline/checkpoints/epoch_0500.pth"
    cfg.model_arch = "resnet50"
    cfg.model_conf = get_model_config(cfg.model_arch)

    # transform
    cfg.inp_size = (360, 640)  # (h, w)
    cfg.mean_rgb = [0, 0, 0]
    cfg.std_rgb = [1, 1, 1]

    # path IMG and json
    cfg.path_test_annofile = "/mnt/data1/huangpg/TianChi/traffic/data/amap_traffic_annotations_test.json"
    cfg.path_folder_testdata = "/mnt/data1/huangpg/TianChi/traffic/data/amap_traffic_test_0712/"
    cfg.path_save_json = "result_" + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + ".json"

    ##############################################################
    #                          predictor                         #
    ##############################################################
    predictor = Predictor(cfg)

    ##############################################################
    #                    gen json file to commit                 #
    ##############################################################
    predictor.gen_commit_json_file()


if __name__ == '__main__':
    main()
