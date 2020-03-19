import sys, os, argparse, math
sys.path.append("../modules")
import torch
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm

# 自作モジュール
from networks import Pg_Generator

def get_latent(latent_size):
    latent = torch.randn(size=(1, latent_size, 1, 1))
    return latent

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=".", help="output directory")
    parser.add_argument("--model", type=str, default="./model", help="generator model path")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--n_image", type=int, default=1, help="number of generated image")
    parser.add_argument("--device", type=str, choices=["gpu", "cpu"], default="cpu", help="device to use")
    parser.add_argument("--save_vec", action="store_true", default=False)
    parser.add_argument("--vec_dir", default=None)
    parser.add_argument("--latent_size", type=int, default=512)
    
    opt = parser.parse_args()
    # ----- Device Setting -----
    if opt.device == "gpu":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("gpu not found")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    print("Device :",device)

    # ----- Output Setting -----
    img_output_dir = opt.output
    if not os.path.exists(os.path.join(img_output_dir, "images")):
        os.mkdir(os.path.join(img_output_dir, "images"))
    
    if not os.path.exists(os.path.join(img_output_dir, "latents")):
        os.mkdir(os.path.join(img_output_dir, "latents"))

    print("Output :", img_output_dir)

    # ----- Model Loading -----
    print("Use model :", opt.model)
    print("Resolution :", opt.resolution)
    stage = int(math.log2(opt.resolution) - 1.0)
    model_g = Pg_Generator(resolution=opt.resolution, start_stage=stage)
    model_g.load_state_dict(torch.load(opt.model, map_location="cpu"))
    model_g.to(device)

    model_g.eval()
    # ----- Load Tensor -----
    if not opt.vec_dir is None:
        names = os.listdir(opt.vec_dir)
        paths = [os.path.join(opt.vec_dir, name) for name in names]
        nd_arrays = [np.load(path) for path in paths]
        latents = [torch.from_numpy(np_array) for np_array in nd_arrays]

    else:
        latents = [get_latent(opt.latent_size) for i in range(opt.n_image)]
    
    print("Generate image num :", len(latents))

    # ----- Generate Step -----
    print("Start Generate Process")
    for index,latent in tqdm(enumerate(latents)):
        latent.to(device)
        
        img_g = model_g(latent)

        img_path = os.path.join(img_output_dir, "images", str(index + 1) + ".png")
        save_image(img_g, img_path)

        if opt.save_vec:
            latent_path = os.path.join(img_output_dir, "latents", str(index + 1) + ".npy")
            np.save(latent_path, latent.numpy())

    print("Finish Generate Process")






