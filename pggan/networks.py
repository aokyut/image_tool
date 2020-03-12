import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2

# ----- Generator -----

class G_first_block(nn.Module):
    def __init__(self, latent_size = 512, out_ch):
        self.init_conv1 = nn.Conv2d(latent_size, out_ch,
                                    kernel_size=4,
                                    padding=3,
                                    stride=1)
        self.init_conv2 = nn.Conv2d(out_ch, out_ch,
                                    kernel_size=3,
                                    padding=1,
                                    stride=1)
        self.activation1 = nn.LeakyReLU()
        self.activation2 = nn.LeakyReLU()

    def forward(self, x):
        x = self.init_conv1(x)
        x = self.activation1(x)
        x = self.init_conv2(x)
        x = self.activation(x)
        norm = torch.sqrt(torch.mean(torch.pow(x, 2)) + 1e-10)
        return x / norm

class G_block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel=3):
        super().__init__()
        assert kernel % 2 == 1, "kernel size must be odd number"
        padding_pix = kernel // 2
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel, padding=padding_pix)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=kernel, padding=padding_pix)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x)
        # pixel wise normalization
        norm = torch.sqrt(torch.mean(torch.pow(x, 2)) + 1e-10)
        x = x / norm
        x = self.conv2(x)
        x = F.leaky_relu(x)
        norm = torch.sqrt(torch.mean(torch.pow(x, 2)) + 1e-10)
        return x / norm


class Torgb(nn.Module):
    def __init__(self, in_channel, out_channel=3):
        super().__init__()
        self.net = nn.Conv2d(in_channel, out_channel)
    def forward(self, x):
        return self.net(x)

class Pg_Generator(nn.Module):
    def __init__(self, in_channel, out_channel, resolution,
                 latent_size=512, transition_iter = 8000):
        super().__init__()
        
        # ----- resolution validation -----
        assert log2(resolution) == int(log2(resolution)), "resolution must be power of 2"
        assert log2(resolution) > 2 "resolution must be more than 2"
        
        self.latent_size = latent_size
        self.channels = []
        for stage in range(log2(resolution) - 1):
            if stage >= 9:
                real_stage = 9
            else:
                real_stage = stage
            channels.append(2 ** (real_stage + 4))
        
        # input latent: dim[batch, latent_size, 1, 1]
        blocks = []
        torgbs = []
        in_chs = self.channels[-2::-1]
        out_chs = self.channels[:0:-1]
        block.append(G_first_block(latent_size=latent_size, in_chs[0]))
        block.append(nn.Upsample(scale_factor=2, mode='bilinear'))
        torgbs.append(Torgb(in_chs[0]))
        for i in len(in_chs):
            blocks.append(G_block(in_chs[i], out_chs[i]))
            blocks.append(nn.Upsample(scale_factor=2, mode='bilinear'))
            torgbs.append(Torgb(in_chs[i]))

        self.blocks = nn.ModuleList(blocks[:-1])
        self.torgbs = nn.ModuleList(torgbs)

        self.stage = 1
        self.transition = False
        self.transition_iter_current = 0
        self.transition_iter = transition_iter

    def forward(self, x):
        if self.transition is True:
            for i in range(2 * self.stage):
                x = self.blocks[i](x)

            # ----- transition process -----
            if self.transition_iter_current < self.transition_iter:
                self.transition_iter_current += 1
                alpha = self.transition_iter_current / self.transition_iter
                x_u = self.blocks[self.stage](x)
                output = x_u * alpha + x * (1 - alpha)
                return self.torgbs[self.stage](x)
            else:
                self.transition_iter_current = 0
                alpha = 1.0
                self.transition = False
                self.stage += 1

                x_u = self.blocks[self.stage](x)




        else:
            for i in range(2 * self.stage - 1):
                x = self.blocks[i]

            return self.torgbs[self.stage - 1](x)


# ----- Discriminator -----

class D_final_block(nn.Module):
    def __init__(self, in_ch=, out_ch)
class D_block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=kernel, paddint=1, stride=1)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=1, stride=1)
        self.net = nn.Sequential([
            nn.Conv2d(in_ch, in_ch, kernel_size=kernel, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=1),
            nn.LeakyReLU()
        ])


class Fromrgb(nn.Module):
    def __init__(self, in_ch=3, out_ch=16):
        super().__init__()
        self.net = nn.Conv2d(in_ch, out_ch)
    def forward(self, x):
        return self.net(x)

class Pg_Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        pass