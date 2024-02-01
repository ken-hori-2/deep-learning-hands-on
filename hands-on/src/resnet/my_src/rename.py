import glob
import os

# 拡張子.pngの画像ファイルを取得する
# path = './dir/*.png'
path = './my_data/pineapple_sample01.jpg'
print(path)

# 画像ファイルを取得する
before_file_list = glob.glob(path)
print('変更前')
print(before_file_list)

# ファイル名を一括で変更する
for i, filename in enumerate(before_file_list, 1):
    os.rename(filename, f'./my_data/pine_sample{i}.png')

after_file_list = glob.glob(path)
print('変更後')
print(after_file_list)
