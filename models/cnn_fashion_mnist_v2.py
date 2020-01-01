import torch.nn as nn


class FashionCNNV2(nn.Module):
    def __init__(self):
        super(FashionCNNV2, self).__init__()

        self.cnn_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.cnn_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)

        self.pooling_1 = nn.MaxPool2d(kernel_size=2)
        self.pooling_2 = nn.MaxPool2d(kernel_size=2)
        self.fc_1 = nn.Linear(1600, 512)
        self.fc_2 = nn.Linear(512, 10)

        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()
        self.relu = nn.ReLU()

        self.bn_1 = nn.BatchNorm2d(32)
        self.bn_2 = nn.BatchNorm2d(64)


    def forward(self, x):
        batch_size = x.size(0)
        out = self.cnn_1(x)
        out = self.relu_1(out)
        out = self.bn_1(out)
        out = self.pooling_1(out)
        out = self.cnn_2(out)
        out = self.relu_2(out)
        out = self.bn_2(out)
        out = self.pooling_2(out)
        # print(out.size())
        out = self.relu(self.fc_1(out.view(batch_size, -1)))

        return self.fc_2(out)


