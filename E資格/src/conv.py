import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim

# conv = nn.Conv2d(3, 5, 3) # 入力=3, 出力=3, カーネルサイズ=3*3

# x = torch.Tensor(torch.rand(1, 3, 28, 28)) # 1枚の画像が3次元(=入力チャネル=カラー画像)で28*28のサイズ
# x = conv(x) # 畳み込みの実行

# print(x.shape)

# 問題
conv = nn.Conv2d(1, 2, 3) # 入力=1, 出力=2, カーネルサイズ=3*3

x = torch.Tensor(torch.rand(100, 1, 64, 64)) # 100枚の画像が1次元で64*64のサイズ
x = conv(x) # 畳み込みの実行

print(x.shape)



"とりあえずモデルを作るとなったら…"
"構築済みモデル(vgg/resnet18など)をコピペして、出力層に100→10層にする"
"train関数をコピペして使ってみる"