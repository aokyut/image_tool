import sys
sys.path.append("../modules")
import torch
from networks import Espcn
from utils import SetImageDataset, raplacian_loss, Verticalrotation
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import os
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, default="../dataset/2d_image")
parser.add_argument("--checkpoints_dir", type=str, default="../checkpoints")
parser.add_argument("--exper_name", type=str, default="espcn")
parser.add_argument("--record_dir", type=str, default="../tensorboard")

#resume
parser.add_argument("--resume_step", type=int, default=0)

parser.add_argument("--gpu", action="store_true", default=False)

parser.add_argument("--small_pix", type=int, default=64)
parser.add_argument("--upscale", type=int, default=2)

parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--batch_size_test", type=int, default=4)
parser.add_argument("--n_record_iter", type=int, default=100)
parser.add_argument("--n_save_model", type=int, default=100)

parser.add_argument("--l_raplacian", type=float, default=0)

opt = parser.parse_args()

if __name__ == "__main__":

    # ---------------Device setting---------------
    if opt.gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print("Device :", device)
    
    # ---------------Dataset setting-----------------
    train_data = SetImageDataset(opt.dataset_dir,
                                 datamode="train",
                                 smaller_pix=opt.small_pix,
                                 preprocess=transforms.Compose([
                                     transforms.RandomHorizontalFlip,
                                     Verticalrotation
                                 ])
                                 transform=transforms.ToTensor(),
                                 upscale=opt.upscale)
    test_data = SetImageDataset(opt.dataset_dir,
                                 datamode="test",
                                 smaller_pix=opt.small_pix,
                                 transform=transforms.ToTensor(),
                                 upscale=opt.upscale)
        
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size_test, shuffle=False)
    print("dataset directory :", opt.dataset_dir)

    # ---------------Summary Writer setting---------------
    if opt.resume_step != 0:
        train_writer = SummaryWriter(log_dir=os.path.join(opt.record_dir,opt.exper_name),
                                     purge_step=opt.resume_step)
        test_writer = SummaryWriter(log_dir=os.path.join(opt.record_dir, opt.exper_name + "_test"),
                                    purge_step=opt.resume_step)
        pass
    else:
        train_writer = SummaryWriter(log_dir=os.path.join(opt.record_dir, opt.exper_name))
        test_writer = SummaryWriter(log_dir=os.path.join(opt.record_dir, opt.exper_name + "_test"))
    print("log directory :",os.path.join(opt.record_dir, opt.exper_name))
    
    # ----------------Net Work define-----------------
    net = Espcn(upscale=opt.upscale)

    if opt.resume_step != 0:
        load_model_path = os.path.join(opt.checkpoints_dir, opt.exper_name, "model_{}.pth".format(opt.resume_step))
        if os.path.exists(load_model_path):
            net.load_state_dict(torch.load(load_model_path))
            print("resume from {}".format(opt.resume_step))
        else:
            assert LoadModelNotFoundError
    else:
        print("Training full scratch")

    net.to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters())

        
    # ----------------Main train Process------------------
    print("training process start")
    net.train()
    running_loss = np.zeros(10)
    step = 0

    if opt.resume_step != 0:
        step = opt.resume_step

    for i in tqdm(range(opt.epoch)):
        with tqdm(train_loader, leave=False) as t:
            for xx,yy in t:
                step += 1
                xx = xx.to(device)
                yy = yy.to(device)
                y_pred = net(xx)

                # learning process
                mse_loss = loss_fn(y_pred, yy)
                rap_loss = raplacian_loss(y_pred, yy, device)
                loss = mse_loss + opt.l_raplacian * rap_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss = np.hstack((running_loss,loss.item()))[-10:]
                t.set_postfix(loss=running_loss.mean(), iter=i+1)
                t.update()

                # -----test process-----
                if step % opt.n_record_iter == 0:
                    net.eval()
                    test_loss_list = []
                    test_mse_loss_list = []
                    test_rap_loss_list = []
                    for xx,yy in test_loader:
                        xx = xx.to(device)
                        yy = yy.to(device)

                        y_pred = net(xx)

                        test_mse_loss = loss_fn(y_pred, yy)
                        test_rap_loss = raplacian_loss(y_pred, yy, device)
                        test_loss = test_mse_loss + opt.l_raplacian * test_rap_loss

                        test_loss_list.append(test_loss.item())
                        test_mse_loss_list.append(test_mse_loss.item())
                        test_rap_loss_list.append(test_rap_loss.item())

                    # -----record process-----

                    test_loss = sum(test_loss_list)/len(test_loss_list)
                    test_mse_loss = sum(test_mse_loss_list)/len(test_mse_loss_list)
                    test_rap_loss = sum(test_rap_loss_list)/len(test_rap_loss_list)

                    train_writer.add_scalar("loss/loss", loss.item(), step)
                    train_writer.add_scalar("loss/mse_loss", mse_loss.item(), step)
                    train_writer.add_scalar("loss/raplacian_loss", rap_loss.item(), step)
                    test_writer.add_scalar("loss/loss", test_loss, step)
                    test_writer.add_scalar("loss/mse_loss", test_mse_loss, step)
                    test_writer.add_scalar("loss/raplacian_loss", test_rap_loss, step)

                    net.train()
                if step % opt.n_save_model == 0:
                    if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.exper_name)):
                        os.makedirs(os.path.join(opt.checkpoints_dir,opt.exper_name))
                    model_save_path = os.path.join(opt.checkpoints_dir, opt.exper_name, "model_" + str(step) + ".pth")
                    torch.save(net.state_dict(), model_save_path)

    print("training process finish")
    train_writer.close()
    test_writer.close()
            
    






