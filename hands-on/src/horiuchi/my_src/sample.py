# import requests
# import re
# import uuid
# from bs4 import BeautifulSoup

# # # url = "https://search.nifty.com/imagesearch/search?select=1&chartype=&q=%s&xargs=2&img.fmt=all&img.imtype=color&img.filteradult=no&img.type=all&img.dimensions=large&start=%s&num=20"
# # url = "https://www.bing.com/search?q=%e3%83%aa%e3%83%b3%e3%82%b4%e3%80%80%e7%94%bb%e5%83%8f%e3%80%80%e3%83%87%e3%83%bc%e3%82%bf&FORM=HDRSC1"
# # keyword = "猫"
# # pages = [1,20,40,60,80,100]

# # for p in pages:
# #         r = requests.get(url%(keyword,p))
# #         soup = BeautifulSoup(r.text,'lxml')
# #         imgs = soup.find_all('img',src=re.compile('^https://msp.c.yimg.jp/yjimage'))
# #         for img in imgs:
# #                 r = requests.get(img['src'])
# #                 with open(str('./picture/')+str(uuid.uuid4())+str('.jpeg'),'wb') as file:
# #                         file.write(r.content)

# # import requests
# # from bs4 import BeautifulSoup
# # url = "https://www.bing.com/search?q=%e3%83%aa%e3%83%b3%e3%82%b4%e3%80%80%e7%94%bb%e5%83%8f%e3%80%80%e3%83%87%e3%83%bc%e3%82%bf&FORM=HDRSC1"
# # response = requests.get(url)
# # soup = BeautifulSoup(response.text, 'html.parser')

# # images = soup.find_all('img')
# # image_urls = [image.get('src') for image in images]

# # print(image_urls)
# # for i, image_url in enumerate(image_urls):
# #     response = requests.get(image_url)
# #     with open(f'image_{i}.jpg', 'wb') as file:
# #         file.write(response.content)




# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from webdriver_manager.chrome import ChromeDriverManager

# # 最新のchromeドライバーをインストールして、インストール先のローカルパスを取得
# driver_path = ChromeDriverManager().install()
# # chromeドライバーのあるパスを指定して、起動
# chrome_service = Service(executable_path=driver_path)
# driver = webdriver.Chrome(service=Service(executable_path=driver_path))

# # 「いらすとや」のWebページにアクセス
# url = "https://www.irasutoya.com/search/label/%E8%81%B7%E6%A5%AD"
# driver.get(url=url)

# # コピーしたXPathを使って画像のWeb要素を取得
# xpath = "/html/body/div[2]/div[2]/div[2]/div/div[2]/div[1]/div[5]/div/div/div/div[1]/a/img"
# element = driver.find_element(by=By.XPATH, value=xpath)

# # 取得した画像のHTML情報を表示
# print(element.get_attribute("outerHTML"))

# # Webドライバーの終了
# driver.quit()

# # # 最新のchromeドライバーをインストールして、インストール先のローカルパスを取得
# # driver_path = ChromeDriverManager().install()
# # # chromeドライバーのあるパスを指定して、起動
# # driver = webdriver.Chrome(service=Service(executable_path=driver_path))

# # # 最新のchromeドライバーをインストールして、インストール先のローカルパスを取得
# # driver_path = ChromeDriverManager().install()
# # # chromeドライバーのあるパスを指定して、起動
# # chrome_service = Service(executable_path=driver_path)

# # 「いらすとや」のWebページにアクセス
# # url = "https://www.irasutoya.com/search/label/%E8%81%B7%E6%A5%AD"
# url = "https://www.bing.com/images/search?q=%E3%83%AA%E3%83%B3%E3%82%B4%E3%80%80%E7%94%BB%E5%83%8F%E3%80%80%E3%83%87%E3%83%BC%E3%82%BF&form=IQFRML&first=1&cw=1177&ch=1311"

# driver.get(url=url)

# # コピーしたXPathを使って画像のWeb要素を取得
# # xpath = "/html/body/div[2]/div[2]/div[2]/div/div[2]/div[1]/div[5]/div/div/div/div[1]/a/img"
# xpath = '/html/body/div[4]/div[5]/div[2]/div[1]/ul[1]/li[1]'

# element = driver.find_element(by=By.XPATH, value=xpath)

# # Web上の画像URLを取得
# img_url = element.get_attribute("src")
# print(img_url)

# # Webドライバーの終了
# driver.quit()






# ライブラリやモジュールをimport
from bs4 import BeautifulSoup
import requests
import bs4
from PIL import Image
import io

#取得したい画像があるURLの情報を変数に格納｡
# url = "https://scraping-for-beginner.herokuapp.com/image"
url = "https://www.bing.com/images/search?q=%e3%83%aa%e3%83%b3%e3%82%b4%e3%80%80%e7%94%bb%e5%83%8f%e3%80%80%e3%83%87%e3%83%bc%e3%82%bf&form=HDRSC3&first=1"
res = requests.get(url)

#そのサイトのhtmlを表示
soup = BeautifulSoup(res.text, "html.parser")
#画像が欲しいのでimg要素を全摘出する。
img_tags = soup.find_all("img")

for i,img_tag in enumerate(img_tags):
    # root_url = "https://scraping-for-beginner.herokuapp.com"
    root_url = "https://www.bing.com/search?q=%e3%83%aa%e3%83%b3%e3%82%b4%e3%80%80%e7%94%bb%e5%83%8f%e3%80%80%e3%83%87%e3%83%bc%e3%82%bf&FORM=HDRSC1" 
    img_url = root_url +  img_tag["src"]

    img = Image.open(io.BytesIO(requests.get(img_url).content))  #f strings以外で出来ないものか…？
    img.save(f'img/{i}.jpeg')

