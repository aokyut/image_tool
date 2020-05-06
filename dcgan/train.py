import os
import sys
sys.path.append("../modules")
import argparse

from networks import Generator, Discriminator
from utils import Dcgan_Dataset, WLoss

import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

from tqdm import tqdm

def lerp(a,b, alpha):
    return a + (b - a) * alpha

def main(opt):
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    # ----- Device Setting -----
    if opt.gpu is True:
        print("try to use gpu")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print("Device :", device)

    # ----- Dataset Setting -----
    train_dataset = Dcgan_Dataset(opt.dataset, datamode="train",
                                  transform=transforms.Compose([transforms.RandomHorizontalFlip(),
                                                                transforms.ToTensor()]))
    
    test_dataset = Dcgan_Dataset(opt.dataset, datamode="test")

    print("Training Dataset :", os.path.join(opt.dataset, "train"))
    print("Testing Dataset :", os.path.join(opt.dataset, "test"))

    # ----- DataLoader Setting -----
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=True)

    print("batch_size :",opt.batch_size)
    print("test_batch_size :",opt.test_batch_size)

    # ----- Summary Writer Setting -----
    train_writer = SummaryWriter(log_dir=os.path.join(opt.tensorboard, opt.exper))
    test_writer = SummaryWriter(log_dir=os.path.join(opt.tensorboard, opt.exper + "_test"))

    print("log directory :",os.path.join(opt.tensorboard, opt.exper))
    print("log step :", opt.n_log_step)

    # ----- Net Work Setting -----
    latent_size = opt.latent_size
    model_D = Discriminator()
    model_G = Generator()

    # resume
    if opt.resume_epoch != 0:
        model_D_path = os.path.join(opt.checkpoints_dir, opt.exper, "model_D_{}.pth".format(str(opt.resume_epoch)))
        model_G_path = os.path.join(opt.checkpoints_dir, opt.exper, "model_G_{}.pth".format(str(opt.resume_epoch)))

        model_G.load_state_dict(torch.load(model_G_path, map_location="cpu"))
        model_D.load_state_dict(torch.load(model_D_path, map_location="cpu"))

    model_D.to(device)
    model_G.to(device)
    model_D.train()
    model_G.train()

    # ロスを計算するときのラベル変数
    ones = torch.ones(opt.batch_size).to(device) # 正例 1
    zeros = torch.zeros(opt.batch_size).to(device) # 負例 0

    val_latents = torch.randn(25, opt.latent_size, 1, 1).to(device)
    loss_fn_g = WLoss("g", device)
    loss_fn_d = WLoss("d", device)
    loss_fn_GP = nn.MSELoss()

    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=opt.lr)
    optimizer_G = torch.optim.Adam(model_G.parameters(), lr=opt.lr)

    print("Latent size :",opt.latent_size)

    # ----- Training Loop -----
    step = opt.resume_epoch * opt.batch_size
    for epoch in tqdm(range(opt.resume_epoch, opt.resume_epoch + opt.epoch)):
        print("epoch :",epoch + 1,"/", opt.resume_epoch + opt.epoch)

        # for latent, real_img in tqdm(train_loader):
        for latent, real_img in tqdm(train_loader):
            step += 1
            latent = latent.to(device)
            real_img = real_img.to(device)
            batch_len = len(real_img)

            fake_img = model_G(latent)

            pred_fake = model_D(fake_img)
            loss_G = loss_fn_g(pred_fake)

            model_D.zero_grad()
            model_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            pred_real = model_D(real_img)
            loss_D_real = loss_fn_d(pred_real, isreal=True)
            with torch.no_grad() :
                fake_img = model_G(latent)
            pred_fake = model_D(fake_img)
            loss_D_fake = loss_fn_d(pred_fake, isreal=False)

            # Calculating gradient penalty
            mixing_rate = torch.randn(size=(len(pred_fake), 1, 1, 1), device=device)
            mixed_image = torch.tensor(lerp(fake_img.clone().detach(), real_img.clone().detach(), mixing_rate), requires_grad=True, device=device)
            mixed_d = model_D(mixed_image)
            mixed_d_mean = torch.mean(mixed_d)
            mixed_d_mean.backward()
            abs_gradient = torch.abs(mixed_image.grad)
            loss_D_gp = loss_fn_GP(abs_gradient, torch.ones(size=abs_gradient.shape, device=device))

            loss_D = loss_D_real + loss_D_fake + loss_D_gp * opt.l_gp

            model_D.zero_grad()
            model_G.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            if step % opt.n_image_log == 0:
                model_G.eval()

                save_dir = opt.image_dir

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = os.path.join(save_dir, str(step) + ".png")

                pred_img = model_G(val_latents)
                save_image(pred_img, save_path, nrow=5)

                model_G.train()
                


            if step % opt.n_log_step == 0:
                # test step 
                model_G.eval()
                model_D.eval()
                test_d_losses = []
                test_d_real_losses = []
                test_d_fake_losses = []
                test_d_gp_losses = []
                test_g_losses = []

                for test_latent, test_real_img in test_loader:
                    test_latent = test_latent.to(device)
                    test_real_img = test_real_img.to(device)
                    batch_len = len(test_latent)
                    test_pred_img = model_G(test_latent)
                    test_fake_g = model_D(test_pred_img)
                    test_g_loss = loss_fn_g(test_fake_g)

                    test_g_losses.append(test_g_loss.item())

                    test_fake_d = model_D(test_pred_img)
                    test_real_d = model_D(test_real_img)
                    test_d_real_loss = loss_fn_d(test_real_d, isreal=True)
                    test_d_fake_loss = loss_fn_d(test_fake_d, isreal=False)

                    # Caluculating gradient penalty

                    mixing_rate = torch.randn(size=(len(test_fake_d), 1, 1, 1), device=device)
                    mixed_image = torch.tensor(lerp(test_pred_img.clone().detach(), test_real_img.clone().detach(), mixing_rate), requires_grad=True, device=device)
                    mixed_d = model_D(mixed_image)
                    mixed_d_mean = torch.mean(mixed_d)
                    mixed_d_mean.backward()
                    abs_gradient = torch.abs(mixed_image.grad)
                    test_d_gp_loss = loss_fn_GP(abs_gradient, torch.ones(size=abs_gradient.shape, device=device))

                    test_d_loss = test_d_real_loss + test_d_fake_loss + opt.l_gp * test_d_gp_loss

                    test_d_real_losses.append(test_d_real_loss.item())
                    test_d_fake_losses.append(test_d_fake_loss.item())
                    test_d_gp_losses.append(test_d_gp_loss)
                    test_d_losses.append(test_d_loss.item())
                
                # record process
                test_g_loss = sum(test_g_losses)/len(test_g_losses)
                test_d_loss = sum(test_d_losses)/len(test_d_losses)
                test_d_real_loss = sum(test_d_real_losses)/len(test_d_real_losses)
                test_d_fake_loss = sum(test_d_fake_losses)/len(test_d_fake_losses)
                test_d_gp_loss = sum(test_d_gp_losses)/len(test_d_gp_losses)


                train_writer.add_scalar("loss/g_loss", loss_G.item(), step)
                train_writer.add_scalar("loss/d_loss", loss_D.item(), step)
                train_writer.add_scalar("loss/d_real_loss", loss_D_real.item(), step)
                train_writer.add_scalar("loss/d_fake_loss", loss_D_fake.item(), step)
                train_writer.add_scalar("loss/d_gp_loss", loss_D_gp.item(), step)
                train_writer.add_scalar("loss/epoch", epoch + 1, step)

                test_writer.add_scalar("loss/g_loss", test_g_loss, step)
                test_writer.add_scalar("loss/d_loss", test_d_loss, step)
                test_writer.add_scalar("loss/d_real_loss", test_d_real_loss, step)
                test_writer.add_scalar("loss/d_fake_loss", test_d_fake_loss, step)
                test_writer.add_scalar("loss/d_gp_loss", test_d_gp_loss, step)

                pred_img = model_G(val_latents)
                grid_img = make_grid(pred_img, nrow=5, padding=0)
                grid_img = grid_img.mul(0.5).add_(0.5)

                train_writer.add_image("train/{}".format(epoch), grid_img, step)

                model_D.train()
                model_G.train()
                
        if (epoch + 1) % opt.n_save_epoch == 0:
            save_dir = os.path.join(opt.checkpoints_dir, opt.exper)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            model_g_path = os.path.join(save_dir, "model_G_{}.pth".format(str(epoch + 1)))
            model_d_path = os.path.join(save_dir, "model_D_{}.pth".format(str(epoch + 1)))
            torch.save(model_D.state_dict(), model_d_path)
            torch.save(model_G.state_dict(), model_g_path)

            print("save_model")

    # save model
    save_dir = os.path.join(opt.checkpoints_dir, opt.exper)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_g_path = os.path.join(save_dir, "model_G_{}.pth".format(str(opt.epoch)))
    model_d_path = os.path.join(save_dir, "model_D_{}.pth".format(str(opt.epoch)))
    torch.save(model_D.state_dict(), model_d_path)
    torch.save(model_G.state_dict(), model_g_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="../dataset/face_crop_img")
    parser.add_argument("--checkpoints_dir", default="../checkpoints")
    parser.add_argument("--exper", default="dcgan")
    parser.add_argument("--tensorboard", default="../tensorboard")
    parser.add_argument("--gpu", action="store_true", default=False)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=4)
    parser.add_argument("--n_log_step", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_save_epoch", type=int, default=10)

    parser.add_argument("--latent_size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--l_gp", type=float, default=10.0)
    parser.add_argument("--image_dir",type=str, default=".")
    parser.add_argument("--n_image_log", type=int, default=100)

    # resume
    parser.add_argument("--resume_epoch", type=int, default=0)

    opt = parser.parse_args()
    main(opt)