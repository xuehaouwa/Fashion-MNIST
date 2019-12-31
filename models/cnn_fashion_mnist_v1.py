import torch
import torch.nn as nn
import torch.nn.functional as F


class FashionCNNV1(nn.Module):
    def __init__(self):
        super(FashionCNNV1, self).__init__()

        self.cnn_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5)
        self.cnn_2 = nn.Conv2d()

        self.pooling_1 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.fc_1 = nn.Linear(20, 512)
        self.fc_2 = nn.Linear(512, 10)

        self.relu = nn.ReLU()


    def forward(self, x):

        out = self.cnn_1(x)
        out = self.relu(out)

        out = self.relu(self.fc_1(out))

        return self.fc_2(out)

