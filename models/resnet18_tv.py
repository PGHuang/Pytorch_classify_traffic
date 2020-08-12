import torch
import torch.nn as nn
import torchvision


class Resnet18_tv(nn.Module):
    def __init__(self, num_classes=3, if_pretrained=True):
        super(Resnet18_tv, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=if_pretrained, progress=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        oup = self.model(x)
        return oup


if __name__ == '__main__':
    inp = torch.randn(4, 3, 360, 640)
    net = Resnet18_tv(num_classes=3, if_pretrained=True)
    oup = net(inp)
    print(oup.shape)
    print(net)
