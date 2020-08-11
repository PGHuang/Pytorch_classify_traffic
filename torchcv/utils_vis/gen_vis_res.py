import torch
import numpy as np
import cv2
from PIL import Image


def gen_vis_result_binary(img_tensor, mask_tensor, oup_tensor):
    batch_merge = torch.cat([
        img_tensor,
        mask_tensor.repeat(1, 3, 1, 1),
        oup_tensor.repeat(1, 3, 1, 1),
    ], dim=2)
    return batch_merge


def gen_res_compare(imgs_tensor, targets_tensor, predicts_tensor):
    predicts_tensor = predicts_tensor.softmax(dim=-1)
    pre_cls = predicts_tensor.argmax(dim=-1)
    imgs_np = np.array(imgs_tensor.detach().cpu())
    targets_np = np.array(targets_tensor.detach().cpu())
    predicts_np = np.array(pre_cls.detach().cpu())
    predicts_list_np = np.array(predicts_tensor.detach().cpu())

    reuslt = []

    for i, img in enumerate(imgs_np):
        img = img.transpose(1, 2, 0) * 255
        img = np.ascontiguousarray(img)  # Attention!
        img = img.astype(np.uint8)

        cv2.putText(img, "target : " + str(targets_np[i]),
                    (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(img, "predict: " + str(predicts_np[i]) + "  " + str(predicts_list_np[i]),
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if targets_np[i] == -1:
            pass
        elif targets_np[i] == predicts_np[i]:
            cv2.putText(img, "result : True", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(img, "result : False", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        # Image.fromarray(img).show()

        img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1)
        reuslt.append(img_tensor)

    res_tensor = torch.stack(reuslt, dim=0)
    # print(res_tensor.shape)
    return res_tensor
