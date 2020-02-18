import sys
sys.path.append("../modules")
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import random, string

def GetRandomStr(num):
    # 英数字をすべて取得
    dat = string.digits + string.ascii_lowercase + string.ascii_uppercase
    # 英数字からランダムに取得
    return ''.join([random.choice(dat) for i in range(num)])

class SetImageDataset(Dataset):
    def __init__(self, root, transform=None, smaller_pix=64, upscale=2, datamode="train"):
        super().__init__()
        self.transform = transform
        self.small_resizer = transforms.Resize(smaller_pix)
        self.large_resizer = transforms.Resize(smaller_pix * upscale)
        if datamode == "val":
            self.image_dir = root
        else:
            self.image_dir = os.path.join(root, datamode)
        self.image_names = os.listdir(self.image_dir)
        self.image_names = [image_name for image_name in self.image_names if "jpg" in image_name or "png" in image_name]

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self,index):
        image_name = os.path.join(self.image_dir, self.image_names[index])
        large_image = self.large_resizer(Image.open(image_name))
        small_image = self.small_resizer(Image.open(image_name))

        if self.transform is not None:
            large_image = self.transform(large_image)
            small_image = self.transform(small_image)

        return small_image, large_image