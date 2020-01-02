import torch.nn as nn


class FashionCNNV3(nn.Module):
    def __init__(self):
        super(FashionCNNV3, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2)
        )

        self.mlp = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 7 * 7, 2048),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 10)
        )


    def forward(self, x):
        batch_size = x.size(0)

        out = self.cnn(x)
        out = self.mlp(out.view(batch_size, -1))

        return out




