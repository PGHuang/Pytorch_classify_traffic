import json
import torch


def test_between_json(path_result_json, path_anno_label):
    anno_result_dict = json.load(open(path_result_json, "r"))
    anno_result_list = anno_result_dict["annotations"]

    anno_label_dict = json.load(open(path_anno_label, "r"))
    anno_label_list = anno_label_dict["annotations"]

    confusion_matrix = torch.zeros((3, 3))

    for res, label in zip(anno_result_list, anno_label_list):
        if res['id'] == label['id']:
            res_status = res['status']
            label_status = label['status']
            confusion_matrix[res_status, label_status] += 1
        else:
            print("res['id'] != label['id']", res['id'], label['id'])
            return
    # print(confusion_matrix)

    """ metric """
    eps = 1e-8
    precision_each = confusion_matrix.diag() / (confusion_matrix.sum(1) + eps)
    recall_each = confusion_matrix.diag() / (confusion_matrix.sum(0) + eps)
    f1_each = 2 * precision_each * recall_each / (precision_each + recall_each + eps)

    precision_mean = torch.mean(precision_each)
    recall_mean = torch.mean(recall_each)
    f1_score = 0.2 * f1_each[0] + 0.2 * f1_each[1] + 0.6 * f1_each[2]
    return precision_mean, recall_mean, f1_score, confusion_matrix


def batch_test(list_res_json, path_anno_label):
    for item in list_res_json:
        precision_mean, recall_mean, f1_score, confusion_matrix = test_between_json(item, path_anno_label)
        print(item.split("/")[-1] + ":  F1-Score =", f1_score)


if __name__ == '__main__':
    list_res_json = [
        "/mnt/data1/huangpg/TianChi/traffic/commit_json/result_20200813-002956.json",
        "/mnt/data1/huangpg/TianChi/traffic/commit_json/result_20200813-002150.json",
        "/mnt/data1/huangpg/TianChi/traffic/commit_json/result_20200812-013911.json",
        "/mnt/data1/huangpg/TianChi/traffic/commit_json/result_20200812-012808.json"
    ]
    path_anno_label = "/mnt/data1/huangpg/TianChi/traffic/commit_json/amap_traffic_annotations_test_labeled_v1.json"
    batch_test(list_res_json, path_anno_label)
