# Import libraries
import torch
from torch import nn
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3,
                      stride=1, padding=1, bias=False),
        )
        self.shortcut = nn.Sequential()

        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1,
                          stride=stride, bias=False),
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()

        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(),
        )
        self.conv_0_0 = nn.Sequential(
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
        )
        self.down_0_0 = nn.Sequential(
            nn.MaxPool2d(2, 2)
        )
        self.up_1_0 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, 2)
        )
        self.conv_0_1 = nn.Sequential(
            ResidualBlock(64, 32),
        )
        self.conv_1_0 = nn.Sequential(
            ResidualBlock(32, 64),
            ResidualBlock(64, 64),
        )
        self.down_1_0 = nn.Sequential(
            nn.MaxPool2d(2, 2)
        )
        self.up_2_0 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, 2)
        )
        self.conv_1_1 = nn.Sequential(
            ResidualBlock(128, 64),
        )
        self.up_1_1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, 2)
        )
        self.conv_0_2 = nn.Sequential(
            ResidualBlock(96, 32),
        )
        self.conv_2_0 = nn.Sequential(
            ResidualBlock(64, 128),
            ResidualBlock(128, 128),
        )
        self.down_2_0 = nn.Sequential(
            nn.MaxPool2d(2, 2)
        )
        self.up_3_0 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, 2)
        )
        self.conv_2_1 = nn.Sequential(
            ResidualBlock(256, 128),
        )
        self.up_2_1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, 2)
        )
        self.conv_1_2 = nn.Sequential(
            ResidualBlock(192, 64),
        )
        self.up_1_2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, 2)
        )
        self.conv_0_3 = nn.Sequential(
            ResidualBlock(128, 32),
        )
        self.conv_3_0 = nn.Sequential(
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
        )
        self.down_3_0 = nn.Sequential(
            nn.MaxPool2d(2, 2)
        )
        self.up_4_0 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, 2)
        )
        self.conv_3_1 = nn.Sequential(
            ResidualBlock(512, 256),
        )
        self.up_3_1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, 2)
        )
        self.conv_2_2 = nn.Sequential(
            ResidualBlock(384, 128),
        )
        self.up_2_2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, 2)
        )
        self.conv_1_3 = nn.Sequential(
            ResidualBlock(256, 64),
        )
        self.up_1_3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, 2)
        )
        self.conv_0_4 = nn.Sequential(
            ResidualBlock(160, 32),
        )
        self.conv_4_0 = nn.Sequential(
            ResidualBlock(256, 512),
            ResidualBlock(512, 512),
        )
        self.segment_head = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3,
                      stride=1, padding=1, bias=False),
        )

    def forward(self, x):

        x = self.pre_layer(x)

        x_0_0_output = self.conv_0_0(x)
        x_1_0_input = self.down_0_0(x_0_0_output)
        x_1_0_output = self.conv_1_0(x_1_0_input)
        x_2_0_input = self.down_2_0(x_1_0_output)
        x_1_0_up = self.up_1_0(x_1_0_output)
        x_0_1_output = self.conv_0_1(torch.cat((x_1_0_up, x_0_0_output)))

        x_2_0_output = self.conv_2_0(x_2_0_input)
        x_3_0_input = self.down_2_0(x_2_0_output)
        x_2_0_up = self.up_2_0(x_2_0_output)
        x_1_1_output = self.conv_1_1(torch.cat((x_2_0_up, x_1_0_output)))
        x_1_1_up = self.up_1_1(x_1_1_output)
        x_0_2_output = self.conv_0_2(torch.cat((x_1_1_up, x_0_0_output, x_0_1_output)))

        x_3_0_output = self.conv_3_0(x_3_0_input)
        x_4_0_input = self.down_3_0(x_3_0_output)
        x_3_0_up = self.up_3_0(x_3_0_output)
        x_2_1_output = self.conv_2_1(torch.cat((x_3_0_up, x_2_0_output)))
        x_2_1_up = self.up_2_1(x_2_1_output)
        x_1_2_output = self.conv_1_2(torch.cat((x_2_1_up, x_1_0_output, x_1_1_output)))
        x_1_2_up = self.up_1_2(x_1_2_output)
        x_0_3_output = self.conv_0_3(torch.cat((x_1_2_up, x_0_0_output, x_0_1_output, x_0_2_output)))

        x_4_0_output = self.conv_4_0(x_4_0_input)
        x_4_0_up = self.up_4_0(x_4_0_output)
        x_3_1_output = self.conv_3_1(torch.cat((x_4_0_up, x_3_0_output)))
        x_3_1_up = self.up_3_1(x_3_1_output)
        x_2_2_output = self.conv_2_2(torch.cat((x_3_1_up, x_2_0_output, x_2_1_output)))
        x_2_2_up = self.up_2_2(x_2_2_output)
        x_1_3_output = self.conv_1_3(torch.cat((x_2_2_up, x_1_0_output, x_1_1_output, x_1_2_output)))
        x_1_3_up = self.up_1_3(x_1_3_output)
        x_0_4_output = self.conv_0_4(torch.cat((x_1_3_up, x_0_0_output, x_0_1_output, x_0_2_output, x_0_3_output)))

        output = self.segment_head(x_0_4_output)

        return output

