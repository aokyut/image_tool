import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__),'module'))
from torchvision.datasets import ImageFolder
from torchvision import transforms
import random, string

def GetRandomStr(num):
    # 英数字をすべて取得
    dat = string.digits + string.ascii_lowercase + string.ascii_uppercase
    # 英数字からランダムに取得
    return ''.join([random.choice(dat) for i in range(num)])

class Pairimagefolder(ImageFolder):
    def __init__(self,root, transform=None, large_size = 128, small_size=64, **kwds):
        super().__init__(root, transform=transform)
        self.large_resizer = transforms.Resize(large_size)
        self.small_resizer = transforms.Resize(small_size)

    def __getitem__(self, index):
        path, _ = self.imgs[index]
        img = self.loader(path)

        large_img = self.large_resizer(img)
        small_img = self.small_resizer(img)

        if self.transform is not None:
            large_img = self.transform(large_img)
            small_img = self.transform(small_img)

        return small_img, large_img
