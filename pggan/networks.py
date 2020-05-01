import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2

# ----- Generator -----

class G_first_block(nn.Module):
    def __init__(self, out_ch, latent_size = 512, upsample=False):
        super().__init__()
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
        self.upsample = upsample

    def forward(self, x):
        x = self.init_conv1(x)
        x = self.activation1(x)
        x = self.init_conv2(x)
        x = self.activation2(x)
        norm = torch.sqrt(torch.mean(torch.pow(x, 2)) + 1e-10)
        x = x / norm
        if self.upsample is True:
            x = F.interpolate(x, scale_factor=2.0)
            return x
        else:
            return x


class G_block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, upsample=False):
        super().__init__()
        assert kernel % 2 == 1, "kernel size must be odd number"
        padding_pix = kernel // 2
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=padding_pix)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel, padding=padding_pix)
        self.activation1 = nn.LeakyReLU()
        self.activation2 = nn.LeakyReLU()

        self.upsample = upsample

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation1(x)
        # pixel wise normalization
        x = x / torch.sqrt(torch.mean(torch.pow(x, 2)) + 1e-10)
        x = self.conv2(x)
        x = self.activation2(x)
        x = x / torch.sqrt(torch.mean(torch.pow(x, 2)) + 1e-10) 

        if self.upsample is True:
            return F.interpolate(x, scale_factor=2.0)
        else:
            return x


class Torgb(nn.Module):
    def __init__(self, in_ch, out_ch=3):
        super().__init__()
        self.net = nn.Conv2d(in_ch, out_ch, 1)
    def forward(self, x):
        return self.net(x)

class Pg_Generator(nn.Module):
    def __init__(self, resolution,
                 latent_size=512, transition_iter=8000,
                 start_stage=1,
                 ):
        super().__init__()
        # 入力層も含めた層の数
        self.max_stage = log2(resolution) - 1
        # validation
        assert int(self.max_stage) == self.max_stage, "resolution should be power of 2"
        assert isinstance(start_stage, int), "start_stage should be Integer"

        stages = list(range(1, int(self.max_stage + 1)))
        assert start_stage in stages, "start_stage must be in {}".format(",".join(list(map(str,stages))))

        # sub function
        def get_ch(stage):
            return int(min(16 * (2 ** (self.max_stage - stage)), 512))

        self.net = nn.ModuleList()
        self.torgb = nn.ModuleList()
    
        for i in stages:
            if i is 1:
                upsample = i < start_stage
                self.net.append(G_first_block(latent_size=latent_size, out_ch=get_ch(i), upsample=upsample))
                self.torgb.append(Torgb(in_ch=get_ch(i)))
            else:
                upsample = i < start_stage
                self.net.append(G_block(in_ch=get_ch(i - 1), out_ch=get_ch(i), upsample=upsample))
                self.torgb.append(Torgb(in_ch=get_ch(i)))

        self.stage = start_stage
        self.growing_flag = False
        self.growing_iter = 0
        self.transition_iter = transition_iter


    def forward(self, x):
        if self.growing_flag is True:
            return self.growing_forward(x)
        else:
            return self.simple_forward(x)

    def growing_forward(self, x):
        self.growing_iter += x.shape[0]
        self.net[self.stage - 1].upsample = True
        for i in range(self.stage):
            x = self.net[i](x)
        alpha = self.growing_iter / self.transition_iter

        x_downer = self.torgb[self.stage - 1](x)
        x_upper = self.net[self.stage](x)
        x_upper = self.torgb[self.stage](x_upper)

        if alpha >= 1.0:
            self.growing_flag = False
            self.growing_iter = 0
            alpha = 1.0
            self.stage += 1

        x = x_upper * (alpha) + x_downer * (1 - alpha)
        return x

    def simple_forward(self, x):
        for i in range(self.stage):
            x = self.net[i](x)
        return self.torgb[self.stage - 1](x)

    def stand_growing_flag(self):
        self.growing_flag = True


# ----- Discriminator -----

class D_final_block(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch + 1, in_ch, kernel_size=3, padding=1, stride=1),

            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_ch, in_ch, kernel_size=4, padding=0, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_ch, 1, kernel_size=1)
        )
    
    def forward(self, x):
        stddev
        batch = x.shape[0]
        stddev = torch.std(x, dim=0, keepdim=True)
        std_mean = torch.mean(stddev, dim=(0, 1, 2, 3), keepdim=True)
        std_channel = std_mean.repeat(batch, 1, 4, 4)
        
        x = torch.cat([x, std_channel], dim=1)
        return self.net(x)

class D_block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=kernel, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=1),
            nn.LeakyReLU()
        )
    
    def forward(self, x):
        x = self.net(x)
        return F.interpolate(x, scale_factor=0.5)


class Fromrgb(nn.Module):
    def __init__(self, in_ch=3, out_ch=16):
        super().__init__()
        self.net = nn.Conv2d(in_ch, out_ch, 1)
    def forward(self, x):
        return self.net(x)

class Pg_Discriminator(nn.Module):
    def __init__(self, resolution,
                 transition_iter=8000,
                 start_stage=1):
        super(Pg_Discriminator, self).__init__()
        # 層の数
        max_stage = log2(resolution) - 1
        # validation
        assert int(max_stage) == max_stage, "resolution must be power of 2"
        assert isinstance(start_stage, int), "start_stage must be Integer"
        
        stages = list(range(1, int(max_stage + 1)))
        assert start_stage in stages, "start_stage must be in {}".format(",".join(list(map(str,stages))))

        def get_ch(stage):
            return int(min(512, 16 * (2 ** (max_stage - stage))))
        
        def get_model(stage):
            if stage == 1:
                return D_final_block(in_ch=get_ch(stage))
            else:
                return D_block(in_ch=get_ch(stage), out_ch=get_ch(stage-1))

        self.net = nn.ModuleList([
            D_final_block(in_ch=get_ch(i)) if i==1 else D_block(in_ch=get_ch(i), out_ch=get_ch(i-1)) for i in stages
        ])
        self.fromrgb = nn.ModuleList([
            Fromrgb(out_ch=get_ch(i)) for i in stages
        ])

        self.stage = start_stage
        self.growing_flag = False
        self.growing_iter = 0
        self.transition_iter = transition_iter

    
    def forward(self, x):
        if self.growing_flag is True:
            return self.growing_forward(x)
        else:
            return self.simple_forward(x)

    def growing_forward(self, x):
        self.growing_iter += x.shape[0]

        alpha = self.growing_iter / self.transition_iter

        if alpha >= 1.0:
            self.growing_iter = 0
            self.growing_flag = False
            alpha = 1.0
    
        x_upper = self.fromrgb[self.stage](x)
        x_upper = self.net[self.stage](x_upper)
        x_downer = nn.functional.interpolate(x, scale_factor=0.5)
        x_downer = self.fromrgb[self.stage - 1](x_downer)

        x = x_upper * (alpha) + x_downer * (1 - alpha)
        for i in range(self.stage - 1, -1, -1):
            x = self.net[i](x)
        
        if alpha >= 1.0:
            self.stage += 1
        
        return x


    def simple_forward(self, x):
        x = self.fromrgb[self.stage - 1](x) 
        for i in range(self.stage - 1, -1, -1):
            x = self.net[i](x)
        
        return x

    def stand_growing_flag(self):
        self.growing_flag = True
        