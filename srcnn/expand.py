import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "module"))
import torch 
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import DatasetFolder
from torchvision.utils import save_image
from utils import Pairimagefolder, GetRandomStr
import argparse
from networks import Srcnn_network
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", default="datasets/train")
parser.add_argument("--output_dir", default="datasets/results")
parser.add_argument("--model_path", default="datasets/models/trained_model")
parser.add_argument("--gpu", action="store_true", default=False)
args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir
model_path = args.model_path
if args.gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

def image_super_resolution(data_loader, model, output_dir, device):
    for xx,yy in tqdm(data_loader):
        xx = xx.to(device)
        yy = yy.to(device)

        bl_recon = torch.nn.functional.upsample(xx, 128, mode="bilinear", align_corners=True)
        pred_y = net(xx)
    
        output_filename = GetRandomStr(10)+".jpg"
        save_image(torch.cat([yy, bl_recon, pred_y], 0),os.path.join(output_dir, output_filename))

if __name__ == "__main__":
    data_folder = Pairimagefolder(input_dir, transform=transforms.ToTensor())
    data_loader = DataLoader(data_folder, batch_size=1, shuffle=True)

    net = Srcnn_network()
    net.load_state_dict(torch.load(model_path))

    image_super_resolution(data_loader, net, output_dir, device)
