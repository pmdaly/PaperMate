import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
from tqdm import tqdm


class Trainer:

    def __init__(self, data_dir='./data/', checkpoint_dir='./ckpts/', device='cpu',
                       batch_size=32, valid_size=0.2, num_workers=0):
        self.data_dir = data_dir
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.batch_size = batch_size
        self.valid_size = valid_size
        self.num_workers = num_workers

    def load_data(self, transform=None, dataset='cifar10'):

        if dataset == 'cifar10':
            train_data = datasets.CIFAR10(self.data_dir, train=True,
                    download=True, transform=transform)
            test_data = datasets.CIFAR10(self.data_dir, train=False,
                    download=True, transform=transform)
        elif dataset == 'vocseg':
            train_data = datasets.VOCSegmentation(self.data_dir, image_set='train',
                    download=True, transform=transform, target_transform=transform)
            test_data = datasets.VOCSegmentation(self.data_dir, image_set='val',
                    download=True, transform=transform, target_transform=transform)
        else:
            # raise an error?
            pass

        train_sampler, valid_sampler = self._train_valid_samplers(train_data)

        self.train_loader = DataLoader(train_data, batch_size=self.batch_size,
                                       sampler=train_sampler, num_workers=self.num_workers)
        self.valid_loader = DataLoader(train_data, batch_size=self.batch_size,
                                       sampler=valid_sampler, num_workers=self.num_workers)
        self.test_loader = DataLoader(test_data, batch_size=self.batch_size,
                                       num_workers=self.num_workers)

    def _train_valid_samplers(self, train_data):
        num_train = len(train_data)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]
        return SubsetRandomSampler(train_idx), SubsetRandomSampler(valid_idx)

    def train(self, model, criterion, optimizer, n_epochs=30):
        valid_loss_min = np.Inf

        for epoch in range(1, n_epochs+1):

            train_loss = 0.0
            valid_loss = 0.0

            import ipdb; ipdb.set_trace()
            model.train()
            for data, target in tqdm(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()*data.size(0)

            model.eval()
            for data, target in self.valid_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                valid_loss += loss.item()*data.size(0)



            train_loss = train_loss/len(self.train_loader.dataset)
            valid_loss = valid_loss/len(self.valid_loader.dataset)

            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch, train_loss, valid_loss))

            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    valid_loss_min, valid_loss))
                torch.save(model.state_dict(), self.checkpoint_dir + 'model_cifar.pt')
                valid_loss_min = valid_loss


    def test(self, model, criterion):
        n_classes = model.classifier[-1].out_features
        test_loss = 0.0
        class_correct = list(0. for i in range(n_classes))
        class_total = list(0. for i in range(n_classes))

        model.eval()
        for data, target in self.test_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()*data.size(0)
            _, pred = torch.max(output, 1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            correct = np.squeeze(correct_tensor.cpu().numpy())
            for i in range(n_classes):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

        test_loss = test_loss/len(self.test_loader.dataset)
        print('Test Loss: {:.6f}, '.format(test_loss), end='')
        print('Test Accuracy (Overall): %2d%% (%2d/%2d)' % (
            100. * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct), np.sum(class_total)))
