import torch
import torchvision
import numpy as np
import matplotlib as plt

# print(torch.__version__)
# print(torchvision.__version__)

# print(np.zeros((2, 3)))
# print(torch.zeros(2, 3)) # np.arrayとほぼ同じ

# # 問題1
# prog01_array = np.array(([0, 1, 2], [3, 4, 5]))
# prog01_torch = torch.tensor(([0, 1, 2], [3, 4, 5]), dtype=torch.int32)

# print(prog01_array)
# print(prog01_torch)

# # 問題2
# prog02_torch = torch.ones((2, 3), dtype=torch.int64) # float32) # int64)
# print(prog02_torch)

"しっかりdtypeを指定することで、numpy.arrayに変換するときにオーバーフローやデータ破壊されることを防ぐ"

# # 問題3
# # prog03 = torch.ones((2, 3), dtype=torch.float64)
# prog03 = np.ones((2, 3)) # , dtype=np.float64)
# after = torch.from_numpy(prog03)
# print(after)
"numpyのデータ型のデフォルトは64なので、tensor型に変換した後は64のまま…tensorのデフォルトではないため、dtypeも出力される"
"変換前と変換後は同じアドレスを参照しているのでどちらもデータが変換されることに注意"

# 問題4
# CPU GPU 転送

# # 問題5
# c = torch.ones(3, 2)
# d = torch.zeros(3, 2)
# print(c*3 + d*2)

"backward()は繰り返し実行すると加算されるので初期化する"
# # 問題6
# v = torch.tensor(1.0, requires_grad=True)
# w = torch.tensor(1.0, requires_grad=True)
# out = 4*v + 6*w + 1
# out.backward() # 偏微分
# print(v.grad, w.grad)

# 問題7
# x = torch.tensor(([1, 1, 1], [1, 2, 2], [2, 2, 3], [3, 3, 3]))
# x = x.reshape(3, -1)
# print(x)
"前側のデータから4つずつ(1行分)取り出されている"

"pytorchでは、_ がついている関数は破壊的メソッドだから本当に書き換えていいかというものとしてみる"
# # 問題8
# x = torch.tensor(([1, 1, 1], [1, 2, 2], [2, 2, 3], [3, 3, 3]))
# # x = 
# x.zero_() # これだけで元の値が書き換わっている
# print(x)