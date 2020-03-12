from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image

class Scalable_Dataset(Dataset):
    def __init__(self, root, datamode = "train", transform = transforms.ToTensor()):
        super().__init__()
        self.image_dir = os.path.join(root, datamode)
        self.image_names = os.listdir(self.image_dir)
        self.image_paths = [os.path.join(self.image_dir, name) for name in self.image_names]
        self.data_length = len(self.image_names)
        self.resolution = 2
        self.transform = transform

    def set_resolution(self, resolution):
        self.resolution = resolution

    def __len__(self):
        return self.data_length

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img = Image.open(img_path)
        
        resized_img = transforms.functional.resize(img, self.resolution)

        if not self.transform is None:
            resized_img = self.transform(resized_img)

        return resized_img
        
