import sys
sys.path.append("../modules")

from networks import Pg_Generator, Pg_Discriminator
from utils import Scalable_Dataset, HingeLoss, BLoss, LSLoss, WLoss

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F

import argparse
import os
from math import log2
from tqdm import tqdm
import numpy as np

def lerp(a,b, alpha):
    return a + (b - a) * alpha

def main(opt):
    # ----- Device Setting -----
    if opt.device == "gpu":
        print("try to use gpu")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print("Device :", device)

    # ----- Dataset Setting -----
    resolution_list = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    train_dataset = Scalable_Dataset(root=opt.dataset_dir, datamode="train",
                                     transform=transforms.Compose([transforms.RandomHorizontalFlip(0.5),
                                                                   transforms.ToTensor()]),
                                     latent_size=opt.latent_size)
    test_dataset = Scalable_Dataset(root=opt.dataset_dir, datamode="test", latent_size=opt.latent_size)

    dataset_length = len(train_dataset)

    print("Training Dataset :", os.path.join(opt.dataset_dir, "train"))
    print("Testing Dataset :", os.path.join(opt.dataset_dir, "test"))

    # ----- DataLoader Setting -----
    batch_size_list = [512, 512, 256, 128, 64, 32, 16, 8, 3]
    batch_size_list = [64, 64, 32, 32, 32]
    if opt.device == "cpu":
        batch_size_list = [16, 16, 8, 6, 4]

    train_loader = DataLoader(train_dataset, batch_size=batch_size_list[0], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    print("batch_size :",batch_size_list)
    print("test_batch_size :","4")

    # ----- Summary Writer Setting -----
    train_writer = SummaryWriter(log_dir=os.path.join(opt.record_dir, opt.exper))
    test_writer = SummaryWriter(log_dir=os.path.join(opt.record_dir, opt.exper + "_test"))

    print("log directory :",os.path.join(opt.record_dir, opt.exper))
    print("log step :", opt.n_log_step)

    # ----- Net Work Setting -----
    latents = torch.randn(size=(25, opt.latent_size, 1, 1)).to(device)

    if opt.start_transition is True:
        start_stage = opt.start_stage - 1
    else:
        start_stage = opt.start_stage
    latent_size = opt.latent_size
    model_D = Pg_Discriminator(resolution=opt.resolution,
                               transition_iter=opt.transition_iter,
                               start_stage=start_stage)
    model_G = Pg_Generator(resolution=opt.resolution,
                           transition_iter=opt.transition_iter,
                           latent_size=opt.latent_size,
                           start_stage=start_stage)
    
    # ----- Resume -----
    if opt.start_stage != 1:
        print("resume :on")
        model_names = os.listdir(os.path.join(opt.checkpoints, opt.exper))
        model_num = [model_name.split("_")[1] for model_name in model_names]

        if str(opt.start_stage - 1) in model_num:
            model_path = os.path.join(opt.checkpoints, opt.exper)

            if opt.start_transition is True:
                model_g_path = os.path.join(model_path, "model_{}_G.pth".format(str(opt.start_stage - 1.5)))
                model_d_path = os.path.join(model_path, "model_{}_D.pth".format(str(opt.start_stage - 1.5)))
            else:
                model_g_path = os.path.join(model_path, "model_{}_G.pth".format(str(opt.start_stage - 1)))
                model_d_path = os.path.join(model_path, "model_{}_D.pth".format(str(opt.start_stage - 1)))

            model_G.load_state_dict(torch.load(model_g_path, map_location="cpu"))
            model_D.load_state_dict(torch.load(model_d_path, map_location="cpu"))
            print("model_G path :", model_g_path)
            print("model_D path :", model_d_path)
        else:
            print("model_{} not found".format(str(opt.start_stage - 1)))
    # -----

    model_D.to(device)
    model_G.to(device)
    model_D.train()
    model_G.train()

    loss_fn_G = WLoss(mode="g", device=device)
    loss_fn_D = WLoss(mode="d", device=device)
    loss_fn_GP = torch.nn.MSELoss()

    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=opt.lr)
    optimizer_G = torch.optim.Adam(model_G.parameters(), lr=opt.lr, betas=(0.0002, 0.5))


    print("Model resolution :",opt.resolution)
    print("Latent size :",opt.latent_size)

    
    # ----- Training Loop -----

    step = 0
    stages = log2(opt.resolution) - 1
    assert stages == int(stages), "resolution must be power of 2"
    stages = int(stages)
    epoch_num = 0

    for stage in range(stages):
        train_dataset.resolution = resolution_list[stage]
        train_loader = DataLoader(train_dataset, batch_size=batch_size_list[stage], shuffle=True)
        test_dataset.resolution = resolution_list[stage]
        print("stage :",stage)
        print("resolution :", resolution_list[stage])

         # ----- Training Step -----
        if opt.epoch != [-1]:
            epochs = opt.epoch
        else:
            epochs = [40, 80, 160, 320, 640]

        print("epochs :", epochs)
        for epoch in range(epochs[stage]):
            if stage + 1 < opt.start_stage:
                break

            print("epoch :", epoch)
            epoch_num += 1
            
            for latent, real_img in tqdm(train_loader):
                step += 1
                latent = latent.to(device)
                real_img = real_img.to(device)

                # train G
                pred_img = model_G(latent)
                fake_g = model_D(pred_img)
                loss_G = loss_fn_G(fake_g)

                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()

                # train D
                with torch.no_grad():
                    pred_img = model_G(latent)
                fake_d = model_D(pred_img)
                real_d = model_D(real_img)

                loss_fake_d = loss_fn_D(fake_d, isreal=False)
                loss_real_d = loss_fn_D(real_d, isreal=True)
                
                # Calculating gradient penalty
                mixing_rate = torch.randn(size=(len(fake_d), 1, 1, 1), device=device)
                mixed_image = torch.tensor(lerp(pred_img.clone().detach(), real_img.clone().detach(), mixing_rate), requires_grad=True, device=device)
                mixed_d = model_D(mixed_image)
                mixed_d_mean = torch.mean(mixed_d)
                mixed_d_mean.backward()
                abs_gradient = torch.abs(mixed_image.grad)
                loss_d_gp = loss_fn_GP(abs_gradient, torch.ones(size=abs_gradient.shape, device=device))

                loss_D = loss_fake_d + loss_real_d + opt.l_gp * loss_d_gp

                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()

                # test step
                if step % opt.n_log_step == 0:
                    model_G.eval()
                    model_D.eval()
                    test_d_losses = []
                    test_d_real_losses = []
                    test_d_fake_losses = []
                    test_g_losses = []
                    for test_latent, test_real_img in test_loader:
                        test_latent = test_latent.to(device)
                        test_real_img = test_real_img.to(device)
                        test_pred_img = model_G(test_latent)
                        test_fake_g = model_D(test_pred_img)
                        test_g_loss = loss_fn_G(test_fake_g)

                        test_g_losses.append(test_g_loss.item())

                        test_fake_d = model_D(test_pred_img)
                        test_real_d = model_D(test_real_img)
                        test_d_real_loss = loss_fn_D(test_real_d, isreal=True)
                        test_d_fake_loss = loss_fn_D(test_fake_d, isreal=False)
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
                    train_writer.add_scalar("loss/d_real_loss", loss_real_d.item(), step)
                    train_writer.add_scalar("loss/d_fake_loss", loss_fake_d.item(), step)
                    train_writer.add_scalar("loss/d_gp_loss", loss_d_gp.item(), step)
                    train_writer.add_scalar("loss/epoch", epoch_num, step)

                    test_writer.add_scalar("loss/g_loss", test_g_loss, step)
                    test_writer.add_scalar("loss/d_loss", test_d_loss, step)
                    test_writer.add_scalar("loss/d_real_loss", test_d_real_loss, step)
                    test_writer.add_scalar("loss/d_fake_loss", test_d_fake_loss, step)


                    # ----- eval -----
                    pred_img = model_G(latents) 
                    grid_img = make_grid(pred_img, nrow=5, padding=0)
                    grid_img = grid_img.mul(0.5).add_(0.5)

                    train_writer.add_image("train/{}/{}".format(stage, epoch), grid_img, step)
                    
                    
                    model_G.train()
                    model_D.train()
                
                if step % opt.n_display_step == 0:
                    print("mode : training")
                    print("epoch :", epoch)
                    print("loss_G :", loss_G.item())
                    print("loss_D :", loss_D.item())
            
            # epoch
            pred_img = model_G(latents)
            pred_img_resize = F.interpolate(pred_img, size=(opt.resolution, opt.resolution), mode="nearest")
            save_dir_path = os.path.join(opt.result_dir, str(stage))
            if not os.path.exists(save_dir_path):
                os.makedirs(save_dir_path)
            save_image(pred_img_resize.to("cpu"), os.path.join(save_dir_path,"{}.png".format(str(epoch))), nrow=5)
            
        
        if stage == stages - 1:
            break

        if stage + 2 < opt.start_stage:
            continue


        if opt.save is True:
            save_dir = os.path.join(opt.checkpoints, opt.exper)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            model_g_path = os.path.join(save_dir, "model_{}.5_G.pth".format(str(stage)))
            model_d_path = os.path.join(save_dir, "model_{}.5_D.pth".format(str(stage)))
            torch.save(model_D.state_dict(), model_d_path)
            torch.save(model_G.state_dict(), model_g_path)

        if opt.start_transition is False and stage + 2 ==  opt.start_stage:
            continue


        # transition step
        model_D.stand_growing_flag()
        model_G.stand_growing_flag()

        train_dataset.resolution = resolution_list[stage + 1]
        test_dataset.resolution = resolution_list[stage + 1]
        batch = batch_size_list[stage + 1]

        iter_loader = iter(DataLoader(train_dataset, batch_size=batch_size_list[stage + 1]))
        transition_step = (opt.transition_iter[stage] // batch) + 1
        model_D.transition_iter = opt.transition_iter[stage]
        model_G.transition_iter = opt.transition_iter[stage]

        print("transition iter :", opt.transition_iter[stage])
        for i in tqdm(range(transition_step)):
            step += 1
            try:
                latent, real_img = next(iter_loader)
            except:
                iter_loader = iter(DataLoader(train_dataset, batch_size=batch_size_list[stage + 1]))
                latent, real_img = next(iter_loader)
            
            latent = latent.to(device)
            real_img = real_img.to(device)

            pred_img = model_G(latent)
            fake_g = model_D(pred_img)
            loss_g = loss_fn_G(fake_g)
            optimizer_G.zero_grad()
            loss_g.backward()
            optimizer_G.step()

            with torch.no_grad():
                pred_img = model_G(latent)

            fake_d = model_D(pred_img)
            real_d = model_D(real_img)
            loss_d_real = loss_fn_D(real_d, isreal=True)
            loss_d_fake = loss_fn_D(fake_d, isreal=False)

            # Calculating gradient penalty
            mixing_rate = torch.randn(size=(len(fake_d), 1, 1, 1), device=device)
            mixed_image = torch.tensor(lerp(pred_img.clone().detach(), real_img.clone().detach(), mixing_rate), requires_grad=True, device=device)
            mixed_d = model_D(mixed_image)
            mixed_d_mean = torch.mean(mixed_d)
            mixed_d_mean.backward()

            abs_gradient = torch.abs(mixed_image.grad)
            loss_d_gp = loss_fn_GP(abs_gradient, torch.ones(size=abs_gradient.shape, device=device))

            loss_d = loss_d_real + loss_d_fake + opt.l_gp * loss_d_gp

            optimizer_D.zero_grad()
            loss_d.backward()
            optimizer_D.step()

            if step % opt.n_log_step == 0:
                model_G.eval()
                model_D.eval()
                test_d_losses = []
                test_d_real_losses = []
                test_d_fake_losses = []
                test_g_losses = []
                for test_latent, test_real_img in test_loader:
                    test_latent = test_latent.to(device)
                    test_real_img = test_real_img.to(device)
                    test_pred_img = model_G(test_latent)
                    test_fake_g = model_D(test_pred_img)
                    test_g_loss = loss_fn_G(test_fake_g)

                    test_g_losses.append(test_g_loss.item())

                    test_fake_d = model_D(test_pred_img)
                    test_real_d = model_D(test_real_img)
                    test_d_real_loss = loss_fn_D(test_real_d, isreal=True)
                    test_d_fake_loss = loss_fn_D(test_fake_d, isreal=False)
                    test_d_loss = test_d_real_loss + test_d_fake_loss

                    test_d_real_losses.append(test_d_real_loss.item())
                    test_d_fake_losses.append(test_d_fake_loss.item())
                    test_d_losses.append(test_d_loss.item())
                
                # record process
                test_g_loss = sum(test_g_losses)/len(test_g_losses)
                test_d_loss = sum(test_d_losses)/len(test_d_losses)
                test_d_real_loss = sum(test_d_real_losses)/len(test_d_real_losses)
                test_d_fake_loss = sum(test_d_fake_losses)/len(test_d_fake_losses)


                train_writer.add_scalar("loss/g_loss", loss_g.item(), step)
                train_writer.add_scalar("loss/d_loss", loss_d.item(), step)
                train_writer.add_scalar("loss/d_real_loss", loss_d_real.item(), step)
                train_writer.add_scalar("loss/d_fake_loss", loss_d_fake.item(), step)
                train_writer.add_scalar("loss/epoch", epoch_num, step)

                test_writer.add_scalar("loss/g_loss", test_g_loss, step)
                test_writer.add_scalar("loss/d_loss", test_d_loss, step)
                test_writer.add_scalar("loss/d_real_loss", test_d_real_loss, step)
                test_writer.add_scalar("real/d_fake_loss", test_d_fake_loss, step)

                # ----- eval -----
                
                pred_img = model_G(latents)
                grid_img = make_grid(pred_img, nrow=3, padding=0)
                grid_img = grid_img.mul(0.5).add_(0.5)

                train_writer.add_image("transition/{}".format(stage), grid_img, step)
                
                model_G.train()
                model_D.train()

            if i % opt.n_display_step == 0:
                print("mode : transition")
                print("loss_g :", loss_g.item())
                print("loss_d :", loss_d.item())
        
        if opt.save is True:
            save_dir = os.path.join(opt.checkpoints, opt.exper)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            model_g_path = os.path.join(save_dir, "model_{}_G.pth".format(str(stage + 1)))
            model_d_path = os.path.join(save_dir, "model_{}_D.pth".format(str(stage + 1)))
            torch.save(model_D.state_dict(), model_d_path)
            torch.save(model_G.state_dict(), model_g_path)

    # save model 
    print("finish training")
    print("model save")
    save_dir = os.path.join(opt.checkpoints, opt.exper)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_d_path = os.path.join(save_dir, "model_D.pth")
    model_g_path = os.path.join(save_dir, "model_G.pth")
    torch.save(model_D.state_dict(), model_d_path)
    torch.save(model_G.state_dict(), model_g_path)
    print("model_D saved :", model_d_path)
    print("model_G saved :", model_g_path)

    train_writer.close()
    test_writer.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="../dataset/face_crop_img")
    parser.add_argument("--checkpoints", type=str, default="../checkpoints")
    parser.add_argument("--exper", type=str, default="test_pg_local", help="experiment name to save")
    parser.add_argument("--record_dir", type=str, default="tensorboard", help="tensorboard name to save")
    parser.add_argument("--resolution", type=int, default=64, help="final resolution of model")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--latent_size", type=int, default=512)
    parser.add_argument("--result_dir", default="results")

    parser.add_argument("--epoch", type=int, nargs="*" ,default=-1, help="epoch number in each stage")
    parser.add_argument("--transition_iter", type=int, nargs="*", default=[8000, 16000, 32000, 64000], help="image number of transition step")

    parser.add_argument("--n_log_step", type=int, default=10)
    parser.add_argument("--n_display_step", type=int, default=10)
    parser.add_argument("--save", action="store_true", default=False) 

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--l_gp", type=float, default=10, help="lambda of gradient penalty of discriminator")

    # resume
    parser.add_argument("--start_stage", type=int, default=1)
    parser.add_argument("--start_transition", action="store_true", default=False)
    
    opt = parser.parse_args()

    main(opt)