import sys
sys.path.append("../modules")
import torch.nn as nn
import torch.nn.functional as F

class Espcn(nn.Module):
    def __init__(self, upscale=2, in_ch=3):
        super().__init__()
        self.upscale = upscale
        self.module_list = nn.ModuleList([nn.Conv2d(in_ch, 32, kernel_size=5, stride=1, padding=2),
                                          nn.LeakyReLU(),
                                          nn.BatchNorm2d(64),
                                          nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                                          nn.LeakyReLU(),
                                          nn.BatchNorm2d(64),
                                          nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
                                          nn.LeakyReLU(),
                                          nn.BatchNorm2d(32),
                                          nn.Conv2d(32, 3*(upscale**2), kernel_size=3, stride=1, padding=1),
                                          nn.PixelShuffle(upscale),
                                          ])
        

    def forward(self, x):
        for model in self.module_list:
            x = model(x)
        return x
