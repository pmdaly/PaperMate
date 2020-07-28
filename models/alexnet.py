import torch
import torch.nn.functional as F
from torch import nn, optim


class AlexNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(9216, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout = nn.Dropout(p=0.5)
        self.local_response_norm = nn.LocalResponseNorm(size=5, alpha=0.0001,
                                                        beta=0.75, k=2)

    def forward(self, x):
        # network
        x = F.relu(self.conv1(x))
        x = self.pool(self.local_response_norm(x))
        x = F.relu(self.conv2(x))
        x = self.pool(self.local_response_norm(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(F.relu(self.conv5(x)))
        # classifier
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x



if __name__ == '__main__':
    # ImageNet
    x = torch.randn((1, 3, 227, 227))
    model = AlexNet()
    print(model(x).shape)
