from crawl_config import Config
import sys,os,time
sys.path.append(os.path.join(os.path.dirname(__file__),'modules'))
from tqdm import tqdm
from bs4 import BeautifulSoup
import csv
import requests 
from time import sleep
import argparse
import pandas as pd

config_data = Config()

def get_img_url_from_page(csv_filename, page_url, tag_classname):
    url = page_url

    res = requests.get(page_url)
    if res.status_code != 200:
        print("URL not found {}".format(page_url))
        return True

    soup = BeautifulSoup(res.text, "html.parser")
    img_tags = soup.find_all("img")

    for img_tag in img_tags:
        if img_tag.get("class") is not None:
            if tag_classname in img_tag.get("class"):
                with open(csv_filename, "a") as f:
                    writer = csv.writer(f)
                    writer.writerow([img_tag.get("src")])
    return True

def get_url_from_csv(csv_filename):
    df = pd.read_csv(csv_filename, header = None)
    return list(df.iloc[:,0])

if __name__ == "__main__":
    img_url_csv = "img_urls.csv"

    task = config_data.crawl_info[0]
    page_url_csv = task["csv_file"]
    img_tag_classname = task["img_tag_class"]
    page_urls = get_url_from_csv(page_url_csv)

    for url in tqdm(page_urls):
        get_img_url_from_page(img_url_csv, url, img_tag_classname)

        
    
    

