import torch
from torch import nn


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
                nn.Linear(4096, 1000)
                )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(16, 64, 3),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, 3),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
                )
        self.classifier = nn.Sequential(
                nn.Linear(512,128),
                nn.Linear(128, 64),
                nn.Linear(64, 10)
                )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    import torch.optim as optim
    import torchvision.transforms as transforms
    from train import Trainer

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')

    #model = AlexNet()
    model = Net()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    trainer = Trainer(device=device)
    trainer.load_data(transform=transform, dataset='cifar10')
    trainer.train(model, criterion, optimizer, n_epochs=30)
    trainer.test(model, criterion)
