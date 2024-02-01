# coding: utf-8

from bs4 import BeautifulSoup
import json
import urllib.request
import requests
from pathlib import Path
import csv
import re

# 出力フォルダを作成
# output_folder = Path('いらすとや')
output_folder = Path('my_data')
output_folder.mkdir(exist_ok=True)
# 「いらすとや」の画像URLを取得
# url = "https://www.irasutoya.com/search/label/%E3%83%93%E3%82%B8%E3%83%8D%E3%82%B9"
# url = "https://www.bing.com/images/search?q=%E3%83%AA%E3%83%B3%E3%82%B4%E3%80%80%E7%94%BB%E5%83%8F%E3%80%80%E3%83%87%E3%83%BC%E3%82%BF&form=IQFRBA&id=D1F907B3D489D432A187EC664B1616A6D811E9F4&first=1&disoverlay=1"
url = "https://free-materials.com/tag/%E3%83%AA%E3%83%B3%E3%82%B4/"


# 画像ページのURLを格納するリストを用意
link_list = []
response = urllib.request.urlopen(url)
soup = BeautifulSoup(response, "html.parser")
# 画像リンクのタグをすべて取得
image_list = soup.select('div.boxmeta.clearfix > h2 > a')

print("url: {}".format(response))
print("soup: {}".format(soup))
print("image_list: {}".format(image_list))

i = 0
# 画像リンクを1つずつ取り出す
for image_link in image_list:
    link_url = image_link.attrs['href']
    link_list.append(link_url)
    # i += 1
    # print("現在の枚数: {} 枚".format(i))

for page_url in link_list:
    page_html = urllib.request.urlopen(page_url)
    page_soup = BeautifulSoup(page_html, "html.parser")
    # 画像ファイルのタグをすべて取得
    img_list = page_soup.select('div.separator > a > img')
    # imgタグを1つずつ取り出す
    for img in img_list:
        # img_name = (img.attrs['alt'])
        # 画像ファイルのURLを抽出
        img_url = (img.attrs['src'])
        file_name = re.search(".*\/(.*png|.*jpg)$", img_url)
        # file_name = xpath = '/html/body/div[4]/div[5]/div[2]/div[1]/ul[1]/li[1]'
        save_path = output_folder.joinpath(file_name.group(1))
        # 画像ファイルのURLからデータをダウンロード
        try:
            # 画像ファイルのURLからデータを取得
            image = requests.get(img_url)
            # 保存先のファイルパスにデータを保存
            open(save_path, 'wb').write(image.content)
            # 保存したファイル名を表示
            print(save_path)
        except ValueError:
            print("ValueError!")
        
        i += 1
        print("現在の枚数: {} 枚".format(i))

    if i >= 100:
        print("100枚保存したので終了します")
        break

print("合計保存枚数: {} 枚".format(i))
