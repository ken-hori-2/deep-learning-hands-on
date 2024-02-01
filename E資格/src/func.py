import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from torchvision.models import vgg11

# data = torch.tensor([-5.0, 5.0, -10.0, 10.0])
# print(F.relu(data))
# print(F.softmax(data, dim=0))
# print(torch.sigmoid(data))

# data = torch.tensor([[0.3, 0.3, 0.3, 0.1]])
# mse_label = torch.tensor([[0, 0, 0, 1]])
# cel_label = torch.tensor([3])
# print(nn.MSELoss(reduction='mean')(data, mse_label))
# print(nn.CrossEntropyLoss()(data, cel_label))

model = vgg11()

adam = optim.Adam(model.parameters())
sgd = optim.SGD(model.parameters(), lr=0.2) # 0.01)
momentum = optim.SGD(model.parameters(), lr=0.2, momentum=0.9) # 0.01, momentum=0.9)

# print("adam:", adam)
# print("sgd:", sgd)
# print("momentum:", momentum)

# x = torch.tensor([0, 0, 1, 0], dtype=torch.float32)
# # y = torch.tensor([0, 2, 4, 6], dtype=torch.float32)
# y = torch.tensor([0, 2, 4, 6], dtype=torch.float32)
# criterion = nn.MSELoss()
# loss_result = criterion(x, F.softmax(y, dim=0))
# print(loss_result)