import torch


def fast_hist(predict_tensor, target_tensor, n_classes):
    k = (predict_tensor >= 0) & (predict_tensor < n_classes)
    hist = torch.bincount(
        n_classes * predict_tensor[k].to(torch.int) + target_tensor[k],
        minlength=n_classes ** 2
    ).reshape(n_classes, n_classes)
    return hist


def per_class_iu(hist):
    epsilon = 1e-5
    return (torch.diag(hist) + epsilon) / (hist.sum(1) + hist.sum(0) - torch.diag(hist) + epsilon)


def compute_global_accuracy(pred, label):
    val_bool = (pred == label)
    val = val_bool.int()
    acc = (val.sum().float()) / val.numel()
    return acc


if __name__ == '__main__':
    predict = torch.tensor([0, 0, 0, 0, 1], dtype=torch.int)
    target = torch.tensor([0, 0, 1, 1, 1], dtype=torch.int)

    print(compute_global_accuracy(predict, target))

    hist = fast_hist(predict, target, 2)
    print(hist)

    miou_list = per_class_iu(hist)
    print(miou_list)
