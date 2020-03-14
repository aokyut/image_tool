import sys
sys.path.append("../modules")
from networks import Pg_Generator, Pg_Discriminator
from utils import Scalable_Dataset
from torch.utils.data import DataLoader
import argparse

def main(opt):
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="../dataset")
    parser.add_argument("--checkpoints_dir", type=str, default="../checkpoints")
    parser.add_argument("--exper", type=str, default="test", help="experiment name to save")
    parser.add_argument("--record_dir", type=str, default="../tensorboard", help="tensorboard name to save")
    parser.add_argument("--resolution", type=int, default=64, help="final resolution of model")

    parser.add_argumnet("--epoch", type=int, default=5, help="epoch number in each stage")
    parser.add_argument("--transition_iter", type=int, default=8000, help="image number of transition step")
    
    opt = parser.parse_args()

    main(opt)