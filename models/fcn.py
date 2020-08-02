import torch
from torch import nn
from torchvision.models import AlexNet

class AlexNetFCN(nn.Module):

    def __init__(self):
        super().__init__()
        alexnet = AlexNet()
        self.features = alexnet.features
        self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Conv2d(256, 4096, kernel_size=6),
                nn.ReLU(),
                nn.Dropout(),
                nn.Conv2d(4096, 4096, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(4096, 1000, kernel_size=1)
                )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x



if __name__ == '__main__':
    import torch.optim as optim
    from torchvision import transforms
    from train import Trainer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AlexNetFCN()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(227),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        ])

    trainer = Trainer(device=device)
    trainer.load_data(transform=transform, dataset='cifar10')
    trainer.train(model, criterion, optimizer, n_epochs=1)
    trainer.test(model, criterion)
