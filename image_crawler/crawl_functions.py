import os, sys, time
sys.path.append(os.path.join(os.path.dirname(__file__),'modules'))
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from bs4 import BeautifulSoup
import base64
from PIL import Image
from io import BytesIO

class Moeimg_crawler:
    def __init__(self, url_file_name):
        self.urls_file_name = url_file_name
        with open(url_file_name) as f:
            self.request_urls_set = set(f.readlines())
        self.iter_num = len(self.request_urls_set)

    def make_request(self):
        if len(self.request_urls_set) == 0:
            return False
        req_url = self.request_urls_set.pop()
        return req_url

    def delete_url_from_database(self, delete_url):
        #ファイルを上書き
        with open(self.urls_file_name, mode="w") as f:
            f.write("".join(self.request_urls_set))

def base64_to_image(base64_str):
    """
    base64_str: data:image/jpeg;base64,/~
    """
    img_base64 = base64_str.split(",")[1]
    img_bin = base64.b64decode(img_base64)
    with BytesIO(img_bin) as b:
        img = Image.open(b).copy().convert("RGB")
    return img


class Image_crawler:
    def __init__(self):
        option = Options()        
        option.add_argument('--headless') 
        self.driver = webdriver.Chrome(executable_path="./chromedriver",options=option)
    
    def get_from_google(self,*q):
        print(q)
        base_url = "https://google.com/search?tbm=isch&q="
        search_quary = "+".join(q)

        search_url = base_url + search_quary
        print(search_url)

        self.driver.get(search_url)
        # print(self.driver.page_source)
        html = self.driver.page_source
        soup = BeautifulSoup(html, "html.parser")

        img_tags = soup.find_all("img")

        images = []

        for img_tag in img_tags:
            src = img_tag.get("src")
            if src is None:
                continue
            if "," not in src :
                continue
            if src is not None:
                image = base64_to_image(src)
                images.append(image)

        return images
    
    def get_from_bing(self, *q):
        base_url = "https://www.bing.com/?scope=images&q="
        search_quary = "+".join(q)

        search_url = base_url + search_quary
        print(search_url)

        self.driver.get(search_url)
        # print(self.driver.page_source)
        html = self.driver.page_source
        soup = BeautifulSoup(html, "html.parser")

        img_tags = soup.find_all("img")
        images = []
        count = 0

        for img_tag in img_tags[:10]:
            src = img_tag.get("src")
            if src is None:
                continue
            if "," not in src :
                continue
            if src is not None:
                try:
                    image = base64_to_image(src)
                    # image.show()
                    # images.append(image)
                    count += 1
                except:
                    print("next")
                    continue
        print(count)
    def get_from_duck(self, *q):
        base_url = "https://duckduckgo.com/?q=hoge&t=h_&ia=images&iax=images"


    

crawler = Image_crawler()
crawler.get_from_bing("城")