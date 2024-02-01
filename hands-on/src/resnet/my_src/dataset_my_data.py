import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import os
import time
from PIL import Image




"Target Object Name"
obj1 = "dog" # "apple"
obj2 = "cat" # "orange"


def pre_set(data_path):
    # data_path = "./train"
    data_path = data_path # "./my_data"

    # file_list = os.listdir("./train")
    file_list = os.listdir(data_path)
    # cat_files = [file_name for file_name in file_list if "cat" in file_name]
    # dog_files = [file_name for file_name in file_list if "dog" in file_name]

    # "Target Object Name"
    # obj1 = "apple"
    # obj2 = "orange"
    cat_files = [file_name for file_name in file_list if obj1 in file_name]
    dog_files = [file_name for file_name in file_list if obj2 in file_name]

    transform = transforms.Compose([
        transforms.Resize((256, 256)), # 入力画像のサイズがバラバラなので、すべて256*256にリサイズする
        transforms.ToTensor()
    ])

    return cat_files, dog_files, transform


class CatDogDataset(Dataset):
    def __init__(self, file_list, dir, transform=None):
        # super().__init__()
        self.file_list = file_list
        self.dir = dir
        self.transform = transform
        # if "dog" in self.file_list[0]:
        if obj1 in self.file_list[0]:
            self.label = 1
        else:
            self.label = 0
    
    # 特殊メソッド
    def __len__(self):
        return len(self.file_list) # 画像の枚数を返す
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.dir, self.file_list[idx])
        img = Image.open(file_path)
        if self.transform is not None: # 前処理がある場合は前処理をする
            img = self.transform(img)
        
        print("チャネル(RGB)？？:{}".format(len(img)))
        return img, self.label

if __name__ == "__main__":
    
    dir_path = "../train/"
    # dir_path = "my_data/"

    cat_files, dog_files, transform = pre_set(dir_path)
    
    cat_dataset = CatDogDataset(cat_files, dir_path, transform=transform)
    dog_dataset = CatDogDataset(dog_files, dir_path, transform=transform)

    # 二つのデータセットを結合して一つのデータセットにする
    cat_dog_dataset = ConcatDataset([cat_dataset, dog_dataset])

    # DataLoader作成
    data_loader = DataLoader(cat_dog_dataset, batch_size=32, shuffle=True)
    data_iter = iter(data_loader)
    imgs, labels = data_iter.__next__() # 1バッチ分表示(size=32)
    print(labels)

    grid_imgs = torchvision.utils.make_grid(imgs[:32]) # 24]) # 32枚表示
    grid_imgs_arr = grid_imgs.numpy()

    plt.figure(figsize=(16, 24))
    plt.imshow(np.transpose(grid_imgs_arr, (1, 2, 0)))
    plt.show()

    # 表示
    print("{} 枚".format(cat_dataset.__len__()))
    print("index=0の写真のRGBごとの値:\n{}".format(cat_dataset.__getitem__(0)))