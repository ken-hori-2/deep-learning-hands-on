import torch
import torch.nn as nn
import torch.optim as optim
import numpy as numpy
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader




if __name__ == "__main__":
    # # testset = datasets.MNIST(root='./data_test', 
    # #                              train=False, 
    # #                              download=True, 
    # #                              transform=None)  # 検証用データセット

    # # print(type(testset))

    # testset = datasets.MNIST(root='./data', 
    #                             train=False, 
    #                             download=True, 
    #                             transform=transforms.ToTensor())  # 検証用データセット

    # test_dataloader = torch.utils.data.DataLoader(testset, 
    #                                         batch_size=32, 
    #                                         shuffle=False, 
    #                                         num_workers=2)  # ミニバッチごとにデータを纏める(学習時にはshuffle=False)

    # # print(type(testset))

    # for imgs, labels in test_dataloader: # train_dataloader:

    #         # 画像データを一次元に変換
    #         "There are two ways to change data"
    #         "How to (1)"# E資格講座
    #         print(imgs)
    #         imgs = imgs.reshape(-1, 28*28*1)  # 画像データを1次元に変換


    #         print(imgs)
    #         break
    
    "値が同じときは同じアドレスを参照する"
    import copy
    aa = 1
    bb = 2 # 1
    print(id(aa), id(bb))
    if id(aa) == id(bb):
        print("True")
    else:
        print("False")
    
    bb = copy.deepcopy(aa) # 1
    bb += 1
    aa += 1
    
    print(id(aa), id(bb))
    if id(aa) == id(bb):
        print("True")
    else:
        print("False")
    # print("aa:{}, bb:{}".format(id(aa), id(bb)))
    
    # print("{}".format(140720321800016))

    import ctypes
    bi = lambda x :ctypes.cast(x, ctypes.py_object).value

    print(id(bb))
    print(bi(140720768427856)) # "hello world"
    
    # a = 1
    # b = a # :True # +1 :False
    # print(id(a), id(b))
    # if id(a) == id(b):
    #     print("True")
    # else:
    #     print("False")
    # # print("a:{}, b:{}".format(id(a), id(b)))
    
    
    
    # "リスト型の時は中身が同じでも異なるアドレスを参照"
    # a_list = [1, 2, 3]
    # b_list = a_list # [1, 2, 3]
    # a_list.append(100)

    # print(id(a_list), id(b_list))
    # if id(a_list) == id(b_list):
    #     print("True")
    # else:
    #     print("False")
    
    # print(a_list, b_list)