import sys,os 
sys.path.append(os.path.join(os.path.dirname(__file__),"module"))
import torch
from torch import optim,nn
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import Pairimagefolder
from networks import Srcnn_network
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--datasets",default="datasets/train")
parser.add_argument("--project_name", default="srcnn_image_10k")
parser.add_argument("--not_save", action="store_false", default=True)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--model_save_path", default="datasets/models/trained_64to128_10k")
parser.add_argument("--iteration", type=int, default=10)
parser.add_argument("--gpu", action="store_true", default=False)
args = parser.parse_args()

dataset_path = args.datasets
save_flag = args.not_save
batch_size = args.batch_size
model_path = args.model_save_path
iteration = args.iteration

if args.gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

train_data = Pairimagefolder(os.path.join(dataset_path, "train"), transform=transforms.ToTensor())
test_data = Pairimagefolder(os.path.join(dataset_path, "test"), transform=transforms.ToTensor())
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=2, shuffle=False)

net = Srcnn_network()
net.to(device)
train_writer = SummaryWriter(log_dir="tensorboard/{}".format(args.project_name))
test_writer = SummaryWriter(log_dir="tensorboard/{}".format(args.project_name + "_test"))

def train_net(net, 
              train_loader,
              optimizer_cls = optim.Adam,
              loss_fn = nn.MSELoss(),
              n_iter = iteration, 
              device=device, 
              train_writer=train_writer, 
              test_wirter=test_writer):
              
    train_losses = []
    train_acc = []
    val_acc = []
    step = 0

    optimizer = optimizer_cls(net.parameters())

    for epoch in tqdm(range(n_iter)):
        running_loss = 0.0
        net.train()

        n = 0
        score = 0
       
        with tqdm(train_loader, leave=False) as t:
            for xx,yy in t:
                step += 1
                xx = xx.to(device)
                yy = yy.to(device)
                y_pred = net(xx)
                
                loss = loss_fn(y_pred, yy)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                t.set_postfix(loss = loss.item())
                t.update()
                n += len(xx)

                if step % 100 == 0:
                    net.eval()
                    test_loss_list = []
                    for xx,yy in test_loader:
                        xx = xx.to(device)
                        yy = yy.to(device)

                        y_pred = net(xx)
                        test_loss_list.append(loss_fn(y_pred, yy).item())
                    test_loss = sum(test_loss_list)/len(test_loss_list)
                    print("test_loss",test_loss)
                    print("train_loss", loss.item())

                    train_writer.add_scalar("mseloss", loss.item(), step)
                    test_writer.add_scalar("mseloss", test_loss, step)

                    net.train()
        
        train_losses.append(running_loss/len(train_loader))
    plt.plot(train_losses)
    plt.show()
    print("model saving at:",model_path)
    torch.save(net.state_dict(), model_path)
    train_writer.close()
    test_writer.close()

if __name__ == "__main__":
    train_net(net, train_loader)

