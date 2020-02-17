import sys,os,time
sys.path.append(os.path.join(os.path.dirname(__file__),'modules'))
from tqdm import tqdm
from bs4 import BeautifulSoup
import csv
import requests 
from time import sleep
import argparse

def config():
    crawl1 = {
        "url_file": "2d_page_url_1.csv",
        "base_url": "https://www.niji-wired.info/"
    }
    crawl2 = {
        "url_file": "2d_page_url_2.csv",
        "base_url": "https://gennji.com/"
    }

def page_crawl1(csv_filename):
    base_url = "https://www.niji-wired.info/"

    for i in tqdm(range(1,72)):
        for j in tqdm(range(100), leave=False):
            sleep(0.05)
        status = get_page_url1(csv_filename, i)
        if status:
            continue
        else:
            break

def get_page_url1(csv_filename, page_num):
    base_url = "https://www.niji-wired.info/"
    if page_num == 1:
        url = base_url
    else:
        url = os.path.join(base_url, "page", str(page_num))

    res = requests.get(url)
    if res.status_code is not 200:
        return False

    soup = BeautifulSoup(res.text, "html.parser")
    a_tags = soup.find_all("a")
    if a_tags == []:
        return False
    for a_tag in a_tags:
        link_url = None
        if a_tag.get("class") is not None:
            if "listlink" in a_tag.get("class"):
                link_url = a_tag.get("href")
            else:
                continue
        else:
            continue
        with open(csv_filename, "a") as f:
            writer = csv.writer(f)
            writer.writerow([link_url])
    return True

def crawl_page1(csv_filename, url):
    pass

def tmp_page_crawl(csv_filename, max_page_num, tag_classname, base_url):
    for i in tqdm(range(1,max_page_num+1)):
        for j in tqdm(range(100), leave=False):
            sleep(0.05)
        status = tmp_get_page_url(csv_filename, tag_classname, i, base_url)
        if status:
            continue
        else:
            print("error happend page {}".format(i))
            continue

def tmp_get_page_url(csv_filename, tag_classname, page_num, base_url):
    if page_num == 1:
        url = base_url
    else:
        url = os.path.join(base_url, "page", str(page_num))
    
    res = requests.get(url)
    if res.status_code != 200:
        return False
    
    soup = BeautifulSoup(res.text, "html.parser")
    a_tags = soup.find_all("a")

    for a_tag in a_tags:
        if a_tag.get("rel") is not None:
            if tag_classname in a_tag.get("rel"):
                if "archives" in a_tag.get("href"):
                    with open(csv_filename, "a") as f:
                        writer = csv.writer(f)
                        writer.writerow([a_tag.get("href")])
    return True




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--max_page_num", default=1, type=int)
    parser.add_argument("--tag_classname", default=None)
    args = parser.parse_args()

    max_page_num = args.max_page_num
    base_url = args.base_url
    tag_classname = args.tag_classname

    tmp_page_crawl(args.csv,max_page_num, tag_classname, base_url)





