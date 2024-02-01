import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib as plt

# # 問題1
# data = torch.tensor([-2, -1, 1, 2], dtype=torch.float32)
"dtype=torch.float32必須"
# print(data)
# model = nn.Linear(4, 2)
# data = model(data)
# print(data)
# # print(model(data))
# output = F.relu(data)
# print(output)
# # print(F.relu(data))

"deep learningは入力いくつから出力いくつにしてくださいとしか言っていない…"
"なのに学習を進めると意味のある計算になるのがすごいところ"

import torch.optim as optim
import torchvision.models.vgg

# # 問題2
# x = torch.tensor([1, 1, 1, 1], dtype=torch.float32)
# y = torch.tensor([0, 2, 4, 6], dtype=torch.float32)
# criterion = nn.MSELoss()
# loss_result = criterion(x, y)
# print(loss_result)

# 問題3

class Model(nn.Module):

    def __init__(self):
        super().__init__()
    
        # 全結合層2つ
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 2)
        # 損失関数と最適化関数
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters())
    
    def forward(self, data):
        
        x = self.fc1(data)
        # 活性化関数
        x = F.relu(x)
        x = self.fc2(x)

        return x


if __name__ == "__main__":

    model = Model()

    data = torch.rand(4)

    # output = model.forward(data)
    # print(model)
    output = model(data)
    print(output)

    # from torch import utils
    # from torchvision import datasets
    # import torchvision.transforms as transforms
    
    # ## こうやってダウンロードして使うことができるよ
    # trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())  # 学習用データセット
    # train_loader = utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)  # ミニバッチごとにデータを纏める(学習時にはshuffle=True)

    # testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())  # 検証用データセット
    # test_loader = utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)  # ミニバッチごとにデータを纏める(学習時にはshuffle=False)

    # "中間層は適当だが、入力>=中間層>=出力くらい。"
    # "あまりここのハイパーパラメータは結果に大きく影響しないのでとりあえず適当"

    # "二次元の画像データを一次元のtensorデータに変換する"