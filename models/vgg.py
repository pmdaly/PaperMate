import torch
import torch.nn as nn


# A-LRN left out as it doesn't improve ILRVRC peformance and leads to increased
# memory consumption and computiation
convnet_configs = {
        'A': ['conv3-64x1', 'conv3-128x1', 'conv3-256x2', 'conv3-512x2', 'conv3-512x2'],
        'B': ['conv3-64x2', 'conv3-128x2', 'conv3-256x2', 'conv3-512x2', 'conv3-512x2'],
        'C': ['conv3-64x2', 'conv3-128x2', 'conv3-256x3', 'conv3-512x3', 'conv3-512x3'],
        'D': ['conv3-64x2', 'conv3-128x2', 'conv3-256x3', 'conv3-512x3', 'conv3-512x3'],
        'E': ['conv3-64x2', 'conv3-128x2', 'conv3-256x4', 'conv3-512x4', 'conv3-512x4'],
        }

def block_to_params(block):
    '''Easier to have more desrciptive convnet config above and add parsing for
    readibility'''
    kernel, chanreps = block.split('-')
    kernel_size = int(kernel[-1])
    out_channels, reps = map(int, chanreps.split('x'))
    return kernel_size, out_channels, reps

def feature_builder(config, conv1=False):
    '''Returs the layer sequence for a given model'''
    layers = []
    in_channels = 3
    for kernel_size, out_channels, reps in map(block_to_params, config):
        for rep in range(reps):
            if conv1 and rep == reps-1: kernel_size = 1
            layers += [nn.Conv2d(in_channels, out_channels, kernel_size), nn.ReLU(), nn.MaxPool2d(2, 2)]
            in_channels = out_channels
    return nn.Sequential(*layers)


# adaptive avg pool ensures the same output size for all models
class VGG(nn.Module):

    def __init__(self, config=convnet_configs['D'], conv1=False):
        super().__init__()
        self.features = feature_builder(config, conv1)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
                nn.Linear(512*7*7, 4096),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(4096, 1000)
                )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class VGG11(VGG):

    def __init__(self):
        super().__init__(convnet_configs['A'])


class VGG13(VGG):

    def __init__(self):
        super().__init__(convnet_configs['B'])


class VGG16A(VGG):

    def __init__(self):
        super().__init__(convnet_configs['C'], conv1=True)


class VGG16(VGG):

    def __init__(self):
        super().__init__(convnet_configs['C'])


class VGG19(VGG):

    def __init__(self):
        super().__init__(convnet_configs['D'])
