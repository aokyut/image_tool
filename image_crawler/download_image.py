import sys,os,time
sys.path.append(os.path.join(os.path.dirname(__file__),'modules'))
import requests
from PIL import Image
from io import BytesIO
import argparse
from tqdm import tqdm
import pandas as pd
import imghdr

def change_jpg_to_png(filename):
    path_without_ext = os.path.splitext(filename)[0] + ".png"
    return path_without_ext


def download_from_url(url, output_dir="results"):
    res = requests.get(url)
    file_name = os.path.join(output_dir,os.path.basename(url))
    if res.status_code == 200:
        i = Image.open(BytesIO(res.content))
        if i.format == "JPEG":
            file_name = change_jpg_to_png(file_name)
        i.save(file_name)
    for j in tqdm(range(100), leave=False):
        time.sleep(0.01)

def url_iterator(urls):
    iterate_num = len(urls)
    for i in range(iterate_num):
        yield urls.pop()

def main_routain(url_csv, output_dir, download_num, start_num = 0):
    df = pd.read_csv(url_csv, header=None)
    urls = set(df[0])
    iterate_num = len(urls)
    count = 0
    for url in tqdm(url_iterator(urls), total=iterate_num):
        if count < start_num:
            count += 1
            continue
        if download_num:
            if download_num <= count:
                break 
        if not "http" in str(url) or (not ".png" in str(url) and not ".jpg" in str(url)):
            continue
        if os.path.exists(os.path.join(output_dir, os.path.splitext(os.path.basename(url))[0] + ".jpg")):
            print("skip")
            continue
        try:
            download_from_url(url, output_dir)
        except:
            import traceback
            traceback.print_exc()
        count += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("url_csv")
    parser.add_argument("--output_dir",default = "datasets/crawl_result")
    parser.add_argument("--start_num" ,type=int, default = 0)
    parser.add_argument("--end_num",type=int, default = 1000)
    args = parser.parse_args()

    url_csv = args.url_csv
    output_dir = args.output_dir
    start_num = args.start_num
    if args.end_num == -1:
        download_num = float("inf")
    else:
        download_num = args.start_num

    main_routain(url_csv, output_dir, download_num, start_num)
