import torch
import torch.nn as nn
import torch.nn.functional as F


class FashionCNNV2(nn.Module):
    def __init__(self):
        super(FashionCNNV2, self).__init__()

        self.cnn_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5)


    def forward(self, *input):
        pass

