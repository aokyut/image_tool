import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), "modules"))
import cv2 
import numpy as np
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("input_dir")
parser.add_argument("output_dir")
parser.add_argument("--resolution", type=int, default=32)
args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir
rezo = args.resolution
def resize_images_from_dir(input_dir=input_dir, output_dir=output_dir, rezo = rezo):
    pathes = os.listdir(input_dir)
    
    for path in tqdm(pathes):
        img_path = os.path.join(input_dir,path)
        img = cv2.imread(img_path)
        try:
            small_img = cv2.resize(img, (rezo, rezo))
        except:
            print("exception happend")
            print(img_path)
            continue
        output_filename = path
        output_path = os.path.join(output_dir, output_filename)

        cv2.imwrite(output_path, small_img)

if __name__ == "__main__":
    resize_images_from_dir()

