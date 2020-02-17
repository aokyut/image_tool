class Config():
    def __init__(self):
        self.crawl_info = [
            {    
                "base_url": "https://www.niji-wired.info/",
                "csv_file": "2d_page_url_1.csv",
                "img_tag_class": "imglinl01",
                "a_tag_class": "listlink",
                "max_page_num": 72,
                "lazy_load":False
            },{
                "base_url":  "https://gennji.com/",
                "csv_file": "2d_page_url_2.csv",
                "a_tag_class": "entry-link", 
                "max_page_num": 185,
                "lazy_load":True
            },{
                "base_url": "http://momoniji.com/",
                "csv_file": "2d_page_url_3.csv",
                "a_tag":{"class":"entry-card-wrap"},
                "max_page_num": 580,
                "lazy_load": True
            },{
                "base_url": "https://shikorina.net/category/%e4%ba%8c%e6%ac%a1%e5%85%83%e3%82%a8%e3%83%ad%e7%94%bb%e5%83%8f",
                "csv_file": "2d_page_url_4.csv",
                "a_tag": {"rel": "bookmark"},
                "lazy_laod":True
            }
        ]