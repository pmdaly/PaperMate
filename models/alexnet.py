import torch
import torch.nn.functional as F
from torch import nn, optim


class AlexNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=11, stride=4),
                nn.ReLU(),
                nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(96, 256, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(256, 384, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(384, 384, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2)
                )
        self.classifier = nn.Sequential(
                nn.Linear(9216, 4096),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 4096),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 1000),
                nn.LogSoftmax(dim=1)
                )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    # ImageNet
    x = torch.randn((1, 3, 227, 227))
    model = AlexNet()
    print(model(x).shape)
