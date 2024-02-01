import torch
import torch.nn as nn
import torch.nn.functional as F

class Model():

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 2)
    
    def forward(self, x):

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":

    model = Model()

    data = torch.tensor([1, 2, 3, 4], dtype=torch.float32)

    # print(model(data))
    print(model.forward(data))