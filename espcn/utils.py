import sys
sys.path.append("../modules")
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class SetImageDataset(Dataset):
    def __init__(self, root, transform=None, smaller_pix=64, upscale=2, datamode="train"):
        super().__init__(root, transform=transform)
        self.transform = transform
        self.small_resizer = transforms.Resize(smaller_pix)
        self.large_resizer = transforms.Resize(smaller_pix * upscale)
        self.image_dir = os.path.join(root, datamode)
        self.image_names = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.image_names)
    
    def __getitem(self,index):
        image_name = os.path.join(self.image_dir, self.image_names[index])
        large_image = self.large_resizer(transforms.ToTensor(Image.open(image_name)))
        small_image = self.small_resizer(transforms.ToTensor(Image.open(image_name)))

        if self.transform is not None:
            large_image = self.transform(large_image)
            small_image = self.transform(small_image)

        return small_image, large_image