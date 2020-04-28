import os
import argparse

from networks import Generator, Discriminator
from utils import Dcgan_Dataset

import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from tqdm import tqdm



def main(opt):
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    # ----- Device Setting -----
    if opt.gpu is True:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print("Device :", device)

    # ----- Dataset Setting -----
    train_dataset = Dcgan_Dataset(opt.dataset, datamode="train",
                                  transform=transforms.Compose([transforms.RandomVerticalFlip(),
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

    model_D.to(device)
    model_G.to(device)
    model_D.train()
    model_G.train()

    # ロスを計算するときのラベル変数
    ones = torch.ones(opt.batch_size).to(device) # 正例 1
    zeros = torch.zeros(opt.batch_size).to(device) # 負例 0

    val_latents = torch.randn(9, opt.latent_size, 1, 1).to(device)
    loss_f = nn.BCEWithLogitsLoss()

    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=0.0002)
    optimizer_G = torch.optim.Adam(model_G.parameters(), lr=0.0002)

    print("Latent size :",opt.latent_size)

    # ----- Training Loop -----
    step = 0
    for epoch in range(opt.epoch):
        print("epoch :",epoch + 1,"/", opt.epoch)

        for latent, real_img in tqdm(train_loader):
            step += 1
            latent = latent.to(device)
            real_img = real_img.to(device)
            batch_len = len(real_img)

            fake_img = model_G(latent)

            pred_fake = model_D(fake_img)
            loss_G = loss_f(pred_fake, ones[: batch_len])

            model_D.zero_grad()
            model_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            pred_real = model_D(real_img)
            loss_D_real = loss_f(pred_real, ones[: batch_len])
            fake_img = model_G(latent)
            pred_fake = model_D(fake_img)
            loss_D_fake = loss_f(pred_fake, zeros[: batch_len])
            loss_D = loss_D_real + loss_D_fake

            model_D.zero_grad()
            model_G.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            if step % opt.n_log_step == 0:
                # test step 
                model_G.eval()
                model_D.eval()
                test_d_losses = []
                test_d_real_losses = []
                test_d_fake_losses = []
                test_g_losses = []
                for test_latent, test_real_img in test_loader:
                    batch_len = len(test_latent)
                    test_pred_img = model_G(test_latent)
                    test_fake_g = model_D(test_pred_img)
                    test_g_loss = loss_f(test_fake_g, ones[: batch_len])

                    test_g_losses.append(test_g_loss.item())

                    test_fake_d = model_D(test_pred_img)
                    test_real_d = model_D(test_real_img)
                    test_d_real_loss = loss_f(test_real_d, ones[: batch_len])
                    test_d_fake_loss = loss_f(test_fake_d, zeros[: batch_len])
                    test_d_loss = test_d_real_loss + test_d_fake_loss

                    test_d_real_losses.append(test_d_real_loss.item())
                    test_d_fake_losses.append(test_d_fake_loss.item())
                    test_d_losses.append(test_d_loss.item())
                
                # record process
                test_g_loss = sum(test_g_losses)/len(test_g_losses)
                test_d_loss = sum(test_d_losses)/len(test_d_losses)
                test_d_real_loss = sum(test_d_real_losses)/len(test_d_real_losses)
                test_d_fake_loss = sum(test_d_fake_losses)/len(test_d_fake_losses)


                train_writer.add_scalar("loss/g_loss", loss_G.item(), step)
                train_writer.add_scalar("loss/d_loss", loss_D.item(), step)
                train_writer.add_scalar("loss/d_real_loss", loss_D_real.item(), step)
                train_writer.add_scalar("loss/d_fake_loss", loss_D_fake.item(), step)
                train_writer.add_scalar("loss/epoch", epoch + 1, step)

                test_writer.add_scalar("loss/g_loss", test_g_loss, step)
                test_writer.add_scalar("loss/d_loss", test_d_loss, step)
                test_writer.add_scalar("loss/d_real_loss", test_d_real_loss, step)
                test_writer.add_scalar("loss/d_fake_loss", test_d_fake_loss, step)

                # latent_dir = os.path.join(opt.dataset, "val")
                # latent_names = os.listdir(latent_dir)
                # latent_paths = [os.path.join(latent_dir, name) for name in latent_names]
                
                # latents = [torch.from_numpy(np.load(latent_path)) for latent_path in latent_paths]
                # latents = torch.cat(latents, dim=0)[:,:opt.latent_size,:,:]
                pred_img = model_G(val_latents)
                grid_img = make_grid(pred_img, nrow=3, padding=0)
                grid_img = grid_img.mul(0.5).add_(0.5)

                train_writer.add_image("train/{}".format(epoch), grid_img, step)

                model_D.train()
                model_G.train()

    # save model
    save_dir = os.path.join(opt.checkpoints, opt.exper)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_g_path = os.path.join(save_dir, "model_G.pth".format(str(stage + 1)))
    model_d_path = os.path.join(save_dir, "model_D.pth".format(str(stage + 1)))
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

    parser.add_argument("--latent_size", type=int, default=100)

    opt = parser.parse_args()
    main(opt)