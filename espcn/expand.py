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
    print("model scale {} -> {}".format(opt.small_pix, opt.small_pix * opt.upscale))

    # -----Target dir-----
    image_paths = os.listdir(opt.input)


def expand_exact_pix(input_path, output_path, model=net, device=device, opt=opt):
    input_image = Image(input_path)
    small_resizer = transforms.Compose([
        transforms.Resize(opt.small_pix),
        transforms.ToTensor()
    ])
    xx = small_resizer(input_image)
    pred_image = model(xx)
    save_image(pred_image, output_path)

def make_grid(input_path, output_path, model=net, device=device, opt=opt):
    input_image = Image(input_path)
    small_resizer = transforms.Compose([
        transforms.Resize(opt.small_pix),
        transforms.ToTensor()
    ])
    large_resizer = transforms.Compose([
        transforms.Resize(opt.small_pix * opt.upscale),
        transforms.ToTensor()
    ])
    xx = small_resizer(input_image)
    yy = large_resizer(input_image)
    pred_image = model(xx)
    size = int(opt.small_pix * opt.upscale)
    bl_recon = torch.nn.functional.upsample(xx, 128, mode="bilinear", align_corners=True)
    save_image(torch.cat([yy, bl_recon, pre_image], dim=0), output_path)

def expand_any_pix(input_path, output_path, model=net, opt=opt):  # resize any size image
    im = Image.open(input_path)
    width, height = im.size
    print(im.size)
    # -----Size check-----
    assert width >= opt.small_pix and height >= opt.small_pix, \
        "width or height of image should be larger than small_pix"

    # -----Split and Super resolution-----
    whole_im_list = []
    for i in range(height//opt.small_pix):
        line_im_list = []
        for j in range(width//opt.small_pix):
            # crop target part
            top = i * opt.small_pix
            left = j * opt.small_pix
            src_im = F.crop(im, top, left, opt.small_pix, opt.small_pix)
            src_im = F.to_tensor(src_im).unsqueeze(0)
            pred_im = model(src_im)

            line_im_list.append(pred_im)

        top = i * opt.small_pix
        left = width - opt.small_pix
        src_im = F.crop(im, top, left, opt.small_pix, opt.small_pix)
        src_im = F.to_tensor(src_im).unsqueeze(0)
        pred_im = model(src_im)
        pred_im = pred_im[: , :, :, 2 * (opt.small_pix - (width % opt.small_pix)):]

        line_im_list.append(pred_im)

        line_im = torch.cat(line_im_list, dim=3)
        whole_im_list.append(line_im)
    
    line_im_list = []
    top = height - opt.small_pix
    
    for j in range(width//opt.small_pix):
        left = j * opt.small_pix
        src_im = F.crop(im, top, left, opt.small_pix, opt.small_pix)
        src_im = F.to_tensor(src_im).unsqueeze(0)
        pred_im = model(src_im)
        pred_im = pred_im[:, :, 2 * (opt.small_pix - (height % opt.small_pix)): , :]

        line_im_list.append(pred_im)
    
    top = height - opt.small_pix
    left = width - opt.small_pix
    src_im = F.crop(im, top, left, opt.small_pix, opt.small_pix)
    src_im = F.to_tensor(src_im).unsqueeze(0)
    pred_im = model(src_im)
    pred_im = pred_im[:, :, 2 * (opt.small_pix - (height % opt.small_pix)): , 2 * (opt.small_pix - (width % opt.small_pix)): ]

    line_im_list.append(pred_im)
    
    line_im = torch.cat(line_im_list, dim=3)
    whole_im_list.append(line_im)

    whole_im = torch.cat(whole_im_list, dim=2)

    if opt.val is True:
        bl_transform = transforms.Compose([
            transforms.Resize((opt.upscale * height, opt.upscale * width)),
            transforms.ToTensor()
        ])
        bl_recon = bl_transform(im).unsqueeze(0)
        save_image(torch.cat((whole_im, bl_recon), dim=3), output_path)
    else:
        save_image(whole_im, output_path)

def isimage(path):
    return os.path.isfile(path)

if __name__ == "__main__":
    if os.path.isfile(opt.input):  # when input is file
        output_name = os.path.basename(opt.input)
        output_path = os.path.join(opt.output_dir, output_name)
        expand_any_pix(opt.input, output_path)
    else:  # when input is directory
        input_names = os.listdir(opt.input)
        for input_name in input_names:
            input_path = os.path.join(opt.input, input_name)
            if not isimage(input_path):
                continue
            output_name = os.path.basename(input_name)
            output_path = os.path.join(opt.output_dir, output_name)
            expand_any_pix(input_path, output_path)


    













                            
                            