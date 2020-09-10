import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    '''conv3x3 + relu -> conv3x3 + relu '''
    def __init__(self, input_channel, output_channel):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(Down, self).__init__()
        self.conv_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(input_channel, output_channel)
        )

    def forward(self, x):
        return self.conv_pool(x)


class Up(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(Up, self).__init__()
        self.upconv = nn.ConvTranspose2d(input_channel, output_channel, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(input_channel,output_channel)

    def forward(self,x1, x2):
        x1 = self.upconv(x1)

        # padding
        diffY = x2.shape[2] - x1.shape[2]
        diffX = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x1,x2], dim=1)
        x = self.double_conv(x)
        return x


