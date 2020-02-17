import sys, os 
sys.path.append(os.path.join(os.path.dirname(__file__),'module'))
import torch
from torch import nn, optim

class Convolution_block(nn.Module):
    def __init__(self,in_ch, out_ch):
        super().__init__()
        self.convolution = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=4,
                                     stride=2, padding=1)
        self.activation = nn.ReLU()
        self.normarization = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        x = self.convolution(x)
        x = self.activation(x)
        x = self.normarization(x)
        return x 

class Deconvolution_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.deconvolution = nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=4,
                                                stride=2, padding=1)
        self.activation = nn.ReLU()
        self.normarization = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.deconvolution(x)
        x = self.activation(x)
        x = self.normarization(x)
        return x

class Srcnn_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            Convolution_block(3, 256),
            Deconvolution_block(256, 64),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1)
        ])

    def forward(self, x):
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        return x


