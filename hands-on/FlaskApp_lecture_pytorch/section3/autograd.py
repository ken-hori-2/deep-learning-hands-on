import torch
import torch.nn as nn

x = torch.ones(2, 3, requires_grad=True) # requires_grad属性をTrueに設定することで計算過程が記録されるようになります。
print(x)

y = x + 2
print(y)
print(y.grad_fn)

z = y * 3
print(z)

out = z.mean()
print(out)

a = torch.tensor([1.0], requires_grad=True) # requires_grad属性をTrueに設定することで計算過程が記録されるようになります。
b = a * 2  # bの変化量はaの2倍
# a, b = 1, 2 ... 傾き=2
b.backward()  # 逆伝播
print(a.grad)  # aの勾配（aの変化に対するbの変化の割合）



print("***勾配計算 ***")

def calc(a): # 手計算で勾配計算をやるとこうなる
    b = a*2 + 1
    c = b*b 
    d = c/(c + 2)
    e = d.mean()
    return e

x = [1.0, 2.0, 3.0]
x = torch.tensor(x, requires_grad=True)
y = calc(x)
y.backward()
print("backwardで計算: ", x.grad.tolist())  # xの勾配（xの各値の変化に対するyの変化の割合）　　


# 手計算と比較
delta = 0.001  #xの微小変化

x = [1.0, 2.0, 3.0]
x = torch.tensor(x, requires_grad=True)
y = calc(x).item()

x_1 = [1.0+delta, 2.0, 3.0]
x_1 = torch.tensor(x_1, requires_grad=True)
y_1 = calc(x_1).item()

x_2 = [1.0, 2.0+delta, 3.0]
x_2 = torch.tensor(x_2, requires_grad=True)
y_2 = calc(x_2).item()

x_3 = [1.0, 2.0, 3.0+delta]
x_3 = torch.tensor(x_3, requires_grad=True)
y_3 = calc(x_3).item()

# 勾配の計算
grad_1 = (y_1 - y) / delta
grad_2 = (y_2 - y) / delta
grad_3 = (y_3 - y) / delta

print("手計算:          ", grad_1, grad_2, grad_3)