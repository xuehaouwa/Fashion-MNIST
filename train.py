import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import logging
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import torch.nn.init as init
from models.cnn_fashion_mnist_v1 import FashionCNNV1
from models.cnn_fashion_mnist_v2 import FashionCNNV2
from models.cnn_fashion_mnist_v3 import FashionCNNV3


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--train_model', default='v1', type=str, help="trained model type")
    parser.add_argument('-da', '--data_aug', default=False, type=str2bool, help="whether perform data augmentation for training data")
    parser.add_argument('-lr', '--learning_rate', default=0.1, type=float, help="learning rate")
    parser.add_argument('-mo', '--momentum', default=0.9, type=float, help="momentum")
    parser.add_argument('-e', '--epochs', default=30, type=int, help="epochs")
    parser.add_argument('-bs', '--batch_size', default=128, type=int, help="batch size")
    return parser.parse_args()


class Train:
    use_gpu = torch.cuda.is_available()
    loss_func = nn.CrossEntropyLoss()

    def __init__(self, args):
        self.args = args
        self.net = None
        self.train_dataloader = None
        self.test_dataloader = None



    def create_model(self):

        if 'v1' in self.args.train_model:
            self.net = FashionCNNV1()
        elif 'v2' in self.args.train_model:
            self.net = FashionCNNV2()
        else:
            self.net = FashionCNNV3()

        if self.use_gpu:
            self.net.cuda()

        logging.info("Init model weights")
        self.net.apply(self.init_weights)

        # optimizer
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.args.learning_rate, momentum=self.args.momentum)

    def build_dataloader(self):

        if self.args.data_aug:
            train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        else:
            train_transform = transforms.Compose([transforms.ToTensor()])

        test_transform = transforms.Compose([transforms.ToTensor()])

        self.train_dataloader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('./fashionmnist_data/', train=True, download=True,
                                  transform=train_transform), batch_size=self.args.batch_size, shuffle=True)
        self.test_dataloader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('./fashionmnist_data/', train=False, transform=test_transform),
            batch_size=self.args.batch_size, shuffle=False)

    def training(self):

        self.net.train()
        epoch = 0

        while epoch < self.args.epochs:
            epoch += 1
            losses = []
            for i, (batch_x, batch_y) in enumerate(self.train_dataloader):
                if self.use_gpu:
                    data, target = Variable(batch_x).cuda(), Variable(batch_y).cuda()
                else:
                    data, target = Variable(batch_x), Variable(batch_y)

                self.optimizer.zero_grad()
                out = self.net(data)
                loss = self.loss_func(out, target)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.data.cpu()[0])

            logging.info(f"Epoch: {epoch} Loss: {np.mean(losses)}")

    def testing(self):
        pass

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.BatchNorm2d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)


if __name__ == "__main__":
    args = get_args()

    trainer = Train(args=args)
    trainer.build_dataloader()
    trainer.create_model()
    trainer.training()

