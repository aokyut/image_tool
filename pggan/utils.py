from torchvision import transforms
from torch.utils.data import Dataset
import torch
import os
from PIL import Image

from math import log2


class Scalable_Dataset(Dataset):
    def __init__(self, root, datamode = "train", transform = transforms.ToTensor(), latent_size=512):
        super().__init__()
        if datamode == "val":
            self.image_dir = root
        else:
            self.image_dir = os.path.join(root, datamode)

        self.image_names = os.listdir(self.image_dir)
        self.image_paths = [os.path.join(self.image_dir, name) for name in self.image_names]
        self.data_length = len(self.image_names)
        self.resolution = 4
        self.transform = transform
        self.latent_size = latent_size

    def set_resolution(self, resolution):
        self.resolution = resolution

    def __len__(self):
        return self.data_length

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img = Image.open(img_path)

        if img.size != (64,64):
            print(img_path)

        resized_img = transforms.functional.resize(img, self.resolution)

        
        latent = torch.randn(size=(self.latent_size, 1, 1))

        if not self.transform is None:
            resized_img = self.transform(resized_img)
        return latent, resized_img
        

class HingeLoss(torch.nn.Module):
    def __init__(self, mode="g", device="cpu"):
        super().__init__()
        assert mode in ["g", "d"], "mode shoud be g or d"
        self.mode = mode
        self.device = device

    def forward(self, output_d, isreal=True):
        if self.mode == "g":
            return -torch.mean(output_d)

        zero_tensor = torch.zeros(output_d.shape, device=self.device)
        
        if isreal is True:
            return -torch.mean(torch.min(output_d - 1, zero_tensor))
        else:
            return -torch.mean(torch.min(-output_d - 1, zero_tensor))

class BLoss(torch.nn.Module):
    def __init__(self, mode, device="cpu"):
        super().__init__()
        assert mode in ["g", "d"]
        self.mode = mode
        self.func = torch.nn.BCEWithLogitsLoss()
        self.device = device
    
    def forward(self, output_d, isreal=True):
        if self.mode == "g":
            return self.func(output_d, torch.ones(output_d.shape,device=self.device))
        
        elif self.mode == "d":
            if isreal is True:
                return self.func(output_d, torch.ones(output_d.shape, device=self.device))
            else:
                return self.func(output_d, torch.zeros(output_d.shape, device=self.device))

class LSLoss(torch.nn.Module):
    def __init__(self, mode, device="cpu"):
        super().__init__()
        assert mode in ["g", "d"]
        self.mode = mode
        self.func = torch.nn.MSELoss()
        self.device = device
    
    def forward(self, output_d, isreal=True):
        if self.mode == "g":
            return self.func(output_d, torch.ones(output_d.shape, device=self.device))
        
        elif self.mode == "d":
            if isreal is True:
                return self.func(output_d, torch.ones(output_d.shape, device=self.device))
            else:
                return self.func(output_d, torch.zeros(output_d.shape, device=self.device))
                