import os,sys,time
sys.path.append(os.path.join(os.path.dirname(__file__),'modules'))
import cv2
from random import randint
import argparse
from tqdm import tqdm
from numpy.random import permutation

def cut_out(img,x,y,xd,yd):
    return img[y : yd , x : xd]

def get_point(width, min_pix = 20, max_pix = None):
    if max_pix is None:
        max_pix = width

    assert min_pix <= max_pix, "max_pix must be larger than min_pix"
    if min_pix >= width:
        assert False, "width shorter than min pix error"
    
    point1 = randint(0, width - 1 - min_pix)
    point2 = point1 + randint(min_pix, min(max_pix, width - 1 - point1))

    return max(point1, point2), min(point1, point2)


def random_cutout(filename, cut_num, output_dir, min_pix = 20, max_pix = None):
    whole_img = cv2.imread(filename, cv2.IMREAD_COLOR)

    if whole_img is None:
        return False

    height, width = whole_img.shape[:2]

    for i in range(cut_num):
        try: 
            xd, x = get_point(width, min_pix, max_pix)
            yd, y = get_point(height, min_pix, max_pix)
        except:
            continue
        cut_out_img = cut_out(whole_img,x,y,xd,yd)
        
        output_filename = str(i) + os.path.basename(filename)
        output_path = os.path.join(output_dir,output_filename)
        
        cv2.imwrite(output_path, cut_out_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process croping multiple images randomly from one image of directory")
    parser.add_argument("input_dir", help="Directory include image file")
    parser.add_argument("output_dir", help="Output destination directory")
    parser.add_argument("--use_file_num", type=int, default = 1000, help="Integer how many images use to process")
    parser.add_argument("--cut_num", default = 5, help="Integer how many images to crop from one image")
    parser.add_argument("--min_pix", default = 20, help="Integer minimum pix of croped image")
    parser.add_argument("--max_pix", default = None, help="Interger maximum pix of croped image")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    cut_num = int(args.cut_num)
    min_pix = int(args.min_pix)
    max_pix = int(args.max_pix) if args.max_pix is not None else None

    pathes = permutation(os.listdir(input_dir))[:args.use_file_num]
    with tqdm(pathes) as t:
        for path in t:
            filename = os.path.join(input_dir, path)
            t.set_postfix(path=path)
            t.update()
            random_cutout(filename, cut_num, output_dir, min_pix=min_pix, max_pix=max_pix)
    
