import os,sys,time
sys.path.append(os.path.join(os.path.dirname(__file__),'modules'))
import cv2
import numpy as np
import argparse
import tqdm
from math import atan, pi
"""
cropの手順
最初に元画像を全体が二倍のサイズになるようにマージンを取る
元画像から顔画像をクロップ　crop
それぞれのface_imgに対して目の位置を特定 eye_crop
二つ特定されれば次の処理に移る 
まず顔画像をface_imgよりも二倍のサイズでinput_imgから切り取り
回転させてface_imgと同じサイズでトリミング
"""
parser = argparse.ArgumentParser(description="crop from image file or directory include images")
parser.add_argument("input", help="image file or directory include image files")
parser.add_argument("output_dir", help="destination directory of process result")
parser.add_argument("--mark_eye", action="store_true", default=False, help="mark eye crop result")
parser.add_argument("--show", action="store_true", default=False, help="display crop result")
parser.add_argument("--save", action="store_false", default=True, help="save result")
parser.add_argument("--face_cascade_file", default="datasets/data/cascade_anime.xml", help="cascade file to use for face crop: ~.xml")
parser.add_argument("--eye_cascade_file", default="datasets/data/cascade_anime_eye2.xml", help="cascade file to use for eye crop: ~.xml")
args = parser.parse_args()

def add_margin(img, margin_pix):
    height, width = img.shape[:2]
    margin_img = np.zeros(shape=(height + 2*margin_pix, width + 2*margin_pix, 3),dtype=np.uint8)
    margin_img[margin_pix : margin_pix+height, margin_pix : margin_pix+width, :] = img
    return margin_img

def count_up(filename = "crop.log"):
    with open(filename) as f:
        count = int(f.read())
    count += 1
    with open(filename, "w") as f:
        f.write(str(count))

def eye_crop(face_img,cascade_file = args.eye_cascade_file,config = args):
    cascade = cv2.CascadeClassifier(cascade_file)
    face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_gray = cv2.equalizeHist(face_gray)
    eyes = cascade.detectMultiScale(face_gray,
                                    scaleFactor = 1.1,
                                     minNeighbors = 4,
                                     minSize = (20, 20))
    eye_points = []
    for (ex, ey, ew, eh) in eyes:
        if config.mark_eye:
            cv2.rectangle(face_img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        eye_points.append((int(ex+(ew/2)), int(ey+(eh/2))))
    
    if config.show:
        cv2.imshow("AnimeFaceDetect",face_img)
        cv2.waitKey(0)

    if len(eyes) == 2:
        count_up()
        eye_points.sort()
        return eye_points

    return False
    

def rotate(img, theta, center):
    trans = cv2.getRotationMatrix2D(center, theta , 1)
    rotate_img = cv2.warpAffine(img, trans, img.shape[:2])
    return rotate_img

def detect(filename,output_dir, cascade_file = args.face_cascade_file, config = args):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    input_image = cv2.imread(filename, cv2.IMREAD_COLOR)

    #add margin
    margin_pix = int(max(input_image.shape)/2)
    input_image = add_margin(input_image, margin_pix)
    
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (200, 200))
    count = 1

    if len(faces) == 0:
        return False

    for (x, y, w, h) in faces:
        #cut out
        face_img = input_image[y : y + h, x : x + w]

        center = (int(x+w/2), int(y+h/2))

        eye_points = eye_crop(face_img)
        if eye_points:
            dx = eye_points[1][0] - eye_points[0][0]
            dy = -(eye_points[1][1] - eye_points[0][1])
            
            if dx == 0:
                if dy > 0:
                    theta = 90.0
                else:
                    theta = -90.0
            else:
                theta = 360*atan(dy/dx)/(2*pi)
            
            theta *= -1

            eye_center = (x + int((eye_points[0][0]+eye_points[1][0])/2), y + int((eye_points[0][1]+eye_points[1][1])/2))

            rotate_img = rotate(input_image, theta, center)

            d_center = (eye_center[0]-center[0],eye_center[1]-center[1])
            
            croped_face_img = rotate_img[y + d_center[1] : y + h + d_center[1], x + d_center[0]: x + w + d_center[0]]
        else:
            continue


        #set save file name
        size = min(croped_face_img.shape[:2])
        output_filename = str(count) + "_" + str(size) + "_" +os.path.basename(filename)
        output_path = os.path.join(output_dir,output_filename)

        #save
        if config.save:
            cv2.imwrite(output_path, croped_face_img)

        count += 1



def istarget(filename):
    extensions = [".jpg",".png"]
    if any([ ext in filename for ext in extensions]):
        return True
    else:
        return False


if __name__ == "__main__":

    input_file_or_dir = args.input
    output_dir = args.output_dir

    if os.path.isfile(input_file_or_dir):
        filename = input_file_or_dir
        detect(filename,output_dir)
    else:
        input_pathes = os.listdir(input_file_or_dir)
        with tqdm.tqdm(input_pathes,smoothing=0.96) as t:
            for path in t:
                t.set_postfix(path=path)
                t.update()
                if istarget(path):
                    filename = os.path.join(input_file_or_dir,path)
                    try:
                        detect(filename, output_dir)
                    except:
                        continue
                else:
                    continue
                


