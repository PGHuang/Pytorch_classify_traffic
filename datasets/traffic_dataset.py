import sys

sys.path.append("../")
import os
import torch
from torch.utils.data import Dataset, DataLoader
from config import cfg
from torchvision.transforms import transforms
from PIL import Image
import numpy as np
import albumentations as albu
from torchcv.transforms.random_point_lighting import random_lighting
import json


class TrafficDataset(Dataset):
    def __init__(self, path_data_folder, path_anno_txt, if_train=True):
        self.if_train = if_train

        ############################################################
        #                       parse path_imgs                    #
        ############################################################
        sub_folders = None
        for root, dirs, files in os.walk(path_data_folder):
            sub_folders = dirs
            break
        # print(sub_folders)
        self.path_imgs = []  # [[],[],...,[]]
        for sub_folder in sub_folders:
            temp = []
            files = os.listdir(path_data_folder + sub_folder + "/")
            for file in files:
                temp.append(path_data_folder + sub_folder + "/" + file)
            self.path_imgs.append(temp)
        # print(len(self.path_imgs))
        # print(self.path_imgs[0])

        ############################################################
        #                       parse anno_txt                     #
        ############################################################
        anno_origin = json.load(open(path_anno_txt, "r"))["annotations"]
        # print(anno_origin)
        self.anno_dict = {}
        for item in anno_origin:
            self.anno_dict[item['id']] = item
        # print(self.anno_dict)

    def __len__(self):
        return len(self.path_imgs)

    def __getitem__(self, index):
        path_imgs = self.path_imgs[index]
        sample_id = path_imgs[0].split("/")[-2]
        anno = self.anno_dict[sample_id]
        keyframe = anno['key_frame']
        sample_status = anno['status']

        if self.if_train:
            path_img_key = path_imgs[0][:-5] + keyframe
            img_np = np.array(Image.open(path_img_key).convert("RGB"))
            img_np_aug = self._train_aug(img_np)
            # Image.fromarray(img_np_aug).show()
            img_tensor = transforms.ToTensor()(Image.fromarray(img_np_aug))
            img_tensor = transforms.Normalize(mean=cfg.trans.mean_rgb, std=cfg.trans.std_rgb)(img_tensor)
        else:
            path_img_key = path_imgs[0][:-5] + keyframe
            img_pil = Image.open(path_img_key).convert("RGB").resize((cfg.trans.inp_size[1], cfg.trans.inp_size[0]))
            # img_pil.show()
            img_tensor = transforms.ToTensor()(img_pil)
            img_tensor = transforms.Normalize(mean=cfg.trans.mean_rgb, std=cfg.trans.std_rgb)(img_tensor)

        return img_tensor, sample_status

    def _train_aug(self, img_np):
        aug = albu.Compose([
            # resize
            albu.Resize(height=cfg.trans.inp_size[0], width=cfg.trans.inp_size[1], p=1),

            # h-flip
            albu.HorizontalFlip(p=cfg.trans.hflip_p),

            # color jitter
            albu.OneOf(
                [
                    albu.RandomGamma(gamma_limit=(50, 150), p=0.3),
                    albu.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.3, p=0.3),
                    albu.HueSaturationValue(p=0.3),
                ], p=cfg.trans.color_jitter_p
            ),

            # random noise
            albu.OneOf(
                [
                    albu.GaussNoise((5, 50), p=0.5),
                    albu.ISONoise(color_shift=(0.01, 0.3), intensity=(0.1, 0.5), p=0.5)
                ], p=cfg.trans.noise_p
            ),

            # random blur
            albu.OneOf(
                [
                    albu.GaussianBlur(blur_limit=5, p=0.5),
                    albu.MedianBlur(p=0.5),
                ], p=cfg.trans.blur_p
            ),
        ])
        augmented = aug(image=img_np)
        return augmented["image"]


def get_loader_train(batch_size, path_folder=cfg.path_train_folder, path_anno_txt=cfg.path_train_anno):
    datasets_train = TrafficDataset(path_folder, path_anno_txt, if_train=True)
    train_loader = DataLoader(
        datasets_train, batch_size=batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=True
    )
    return train_loader


def get_loader_val(batch_size, path_folder=cfg.path_test_folder, path_anno_txt=cfg.path_test_anno):
    datasets_val = TrafficDataset(path_folder, path_anno_txt, if_train=False)
    val_loader = DataLoader(
        datasets_val, batch_size=batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=False
    )
    return val_loader


def try_dataset():
    # dset_train
    dset_train = TrafficDataset(cfg.path_train_folder, cfg.path_train_anno, if_train=True)
    img, status = dset_train[0]
    transforms.ToPILImage()(img).show()
    print("train_status =", status)

    # dset_val
    dset_val = TrafficDataset(cfg.path_test_folder, cfg.path_test_anno, if_train=False)
    img, status = dset_val[0]
    transforms.ToPILImage()(img).show()
    print("val_status =", status)


def try_loader():
    import torchvision

    # check loader_train
    loader_train = get_loader_train(cfg.batch_size, path_folder=cfg.path_train_folder,
                                    path_anno_txt=cfg.path_train_anno)
    for batch_id, (imgs, status) in enumerate(loader_train):
        print("* loader_train batch :", batch_id, imgs.shape, status.shape)
        # transforms.ToPILImage()(torchvision.utils.make_grid(imgs, nrow=4)).show()
        # print(status)
        # break

    # check loader_val
    loader_val = get_loader_val(cfg.batch_size, path_folder=cfg.path_test_folder, path_anno_txt=cfg.path_test_anno)
    for batch_id, (imgs, status) in enumerate(loader_val):
        print("* loader_val batch :", batch_id, imgs.shape, status.shape)
        # transforms.ToPILImage()(torchvision.utils.make_grid(imgs, nrow=4)).show()
        # print(status)
        # break


if __name__ == '__main__':
    # try_dataset()

    try_loader()
