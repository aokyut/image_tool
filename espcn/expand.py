import sys
sys.path.append("../modules")
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from utils import SetImageDataset, GetRandomStr
from torchvision import transforms
from networks import Espcn
import os

parser = argparse.ArgumentParser()
parser.add_argument("input_dir",type=str)
parser.add_argument("output_dir", type=str)
parser.add_argument("--model_path", type=str, default="../checkpoints/espcn_20k_image_2x/model_12500.pth")
parser.add_argument("--small_pix", type=int, default=64)
parser.add_argument("--upscale", type=int, default=2)

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
    # -----Dataset setting-----
    dataset = SetImageDataset(opt.input_dir, 
                             smaller_pix=opt.small_pix,
                             upscale=opt.upscale,
                             datamode="val",
                             transform=transforms.Compose([
                                                            transforms.ToTensor()
                                                            ]))
    dataloader = DataLoader(dataset, opt.batch_size, shuffle=True)
    print("Dataroot :",opt.input_dir)

    # -----Net Work setting-----
    net = Espcn(upscale=opt.upscale)
    net.load_state_dict(torch.load(opt.model_path, map_location="cpu"))
    net.to(device)
    net.eval()
    print("small_pix :", opt.small_pix)
    print("upscale :",opt.upscale)
    print("target_pix :", opt.upscale * opt.small_pix)

    # -----Evalating roop-----
    for xx,yy in tqdm(dataloader):
        xx = xx.to(device)
        yy = yy.to(device)

        bl_recon = torch.nn.functional.upsample(xx, 128, mode="bilinear", align_corners=True)
        pred_y = net(xx)
    
        output_filename = GetRandomStr(10)+".png"
        save_image(torch.cat([yy, bl_recon, pred_y], 0),os.path.join(opt.output_dir, output_filename))








                            
                            