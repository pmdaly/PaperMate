import torch
from torch import nn
from torchvision.models import AlexNet

class AlexNetFCN(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = AlexNet().features
        self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Conv2d(256, 4096, kernel_size=6),
                nn.ReLU(),
                nn.Dropout(),
                nn.Conv2d(4096, 4096, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(4096, 21, kernel_size=1)
                )

    def forward(self, x):
        orig_size = x.shape[-2:]
        x = self.features(x)
        x = self.classifier(x)
        x = nn.functional.interpolate(x, orig_size)
        return x


class SqueezeToLong:
    '''Converts the target class in VOCSegmentation from FloatTensor to
    LongTensor and squeezes the channel dimension out.'''

    def __call__(self, target):
        return target.squeeze().type(torch.LongTensor)


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
        transforms.Resize((256, 256)),
        transforms.ToTensor()
        ])

    target_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        SqueezeToLong()
        ])

    trainer = Trainer(device=device)
    trainer.load_data(transform=transform, target_transform=target_transform, dataset='vocseg')
    trainer.train(model, criterion, optimizer, n_epochs=1)
    #trainer.test(model, criterion)
