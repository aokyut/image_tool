import os

from torch.utils.data import Dataset
from torchvision import transforms
import torch
from PIL import Image

class Dcgan_Dataset(Dataset):
    def __init__(self, root, datamode = "train", transform = transforms.ToTensor(), latent_size=100):

        self.image_dir = os.path.join(root, datamode)
        self.image_paths = [os.path.join(self.image_dir, name) for name in os.listdir(self.image_dir)]
        self.data_length = len(self.image_paths)

        self.transform = transform
        self.latent_size = latent_size

    def __len__(self):
        return self.data_length
    
    def __getitem__(self, index):
        latent = torch.randn(size=(self.latent_size, 1, 1))
        img_path = self.image_paths[index]
        img = Image.open(img_path)

        if not self.transform is None:
            img = self.transform(img)
        
        return latent, img

class WLoss(torch.nn.Module):
    def __init__(self, mode, device="cpu"):
        super().__init__()
        assert mode in ["g", "d"]
        self.mode = mode
        self.device = device

    def forward(self, output_d, isreal=True):
        if self.mode == "g":
            return -torch.mean(output_d)
        elif self.mode == "d":
            if isreal is True:
                return -torch.mean(output_d)
            else:
                return torch.mean(output_d)


