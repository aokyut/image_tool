import sys
sys.path.append("../modules")
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from utils import SetImageDataset, GetRandomStr
from torchvision import transforms
import torchvision.transforms.functional as F
from networks import Espcn
import os
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("input",type=str)
parser.add_argument("output_dir", type=str)
parser.add_argument("--model_path", type=str, default="../checkpoints/espcn_20k_image_2x/model_12500.pth")
parser.add_argument("--small_pix", type=int, default=64)
parser.add_argument("--upscale", type=int, default=2)

parser.add_argument("--val", action="store_true", default=False)

parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--gpu", action="store_true", default=False)

opt = parser.parse_args()


if __name__ == "__main__":
    print("input :",opt.input)
    print("output directory :", opt.output_dir)
    if opt.val is True:
        print("execute mode : val")
    else:
        print("execute mode : expand")
    # -----Device setting-----
    if opt.gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("gpu not found")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    print("Device :",device)

    # -----Net Work setting-----
    net = Espcn(upscale=opt.upscale)
    net.load_state_dict(torch.load(opt.model_path, map_location="cpu"))
    net.to(device)
    net.eval()
    print("model upscale factor :", opt.upscale)

def make_grid(input_path, output_path, model=net, device=device, opt=opt):
    input_image = Image.open(input_path)
    width, height = input_image.size
    small_resizer = transforms.Compose([
        transforms.Resize((height // opt.upscale, width // opt.upscale)),
        transforms.ToTensor()
    ])
    large_resizer = transforms.Compose([
        transforms.Resize(((height // opt.upscale) * opt.upscale, (width // opt.upscale) * opt.upscale)),
        transforms.ToTensor()
    ])
    xx = small_resizer(input_image).unsqueeze(0)
    yy = large_resizer(input_image).unsqueeze(0)
    pred_image = model(xx)
    size = int(opt.small_pix * opt.upscale)
    bl_recon = torch.nn.functional.upsample(xx, scale_factor=opt.upscale, mode="bilinear", align_corners=True)
    save_image(torch.cat([yy, bl_recon, pred_image], dim=0), output_path)

def expand(input_path, output_path, model=net, device=device, opt=opt):
    print(input_path)
    input_image = Image.open(input_path)
    width, height = input_image.size
    transform = transforms.ToTensor()
    xx = transform(input_image).unsqueeze(0)
    pred_image = model(xx)
    bl_recon = torch.nn.functional.upsample(xx, scale_factor=opt.upscale)
    save_image(pred_image, output_path)

def isimage(path):
    return os.path.isfile(path)

if __name__ == "__main__":
    if os.path.isfile(opt.input):  # when input is file
        output_name = os.path.basename(opt.input)
        output_path = os.path.join(opt.output_dir, output_name)
        if opt.val is True:
            make_grid(opt.input, output_path)
        else:
            expand(opt.input, output_path)
    else:  # when input is directory
        input_names = os.listdir(opt.input)
        for input_name in tqdm(input_names):
            input_path = os.path.join(opt.input, input_name)
            if not isimage(input_path):
                continue
            output_name = os.path.basename(input_name)
            output_path = os.path.join(opt.output_dir, output_name)
            if opt.val is True:
                make_grid(input_path, output_path)
            else:
                expand(input_path, output_path)


    













                            
                            