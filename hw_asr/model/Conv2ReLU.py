from torch import nn


class Conv2dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout=0.1, padding='same'):
        super(Conv2dReLU, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        return res + x
