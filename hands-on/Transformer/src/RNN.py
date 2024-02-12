import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import random
 
# 乱数のシードを固定
torch.manual_seed(0)

BATCH_SIZE = 2 # 250

class ProductDictionary():
 
  def __init__(self, products):
    self.product_sequence = {}
    self.sequence_product = {}
    for product_line in products:
      for product in product_line.split(','):
        if product not in self.product_sequence:
          sequence = len(self.product_sequence) + 1
          self.product_sequence[product] = sequence
          self.sequence_product[sequence] = product
 
  def get_sequences_by_products(self, products):
    sequences = []
    for product_line in products:
      temp = [self.product_sequence[product] for product in product_line.split(',')]
      sequences.append(temp)
    return sequences
 
  def get_sequence_count(self):
    return len(self.product_sequence) + 1





class ProductDataset(Dataset):
  WINDOW_SIZE = 1
 
  def __init__(self, sequences):
    super().__init__()
    self.x = []
    self.t = []
 
    for sequence_line in sequences:
      if len(sequence_line) <= self.WINDOW_SIZE: # WINDOW_SIZE以下ならサンプルにしない
        continue
      for i in range(len(sequence_line) - self.WINDOW_SIZE):
        tmp_x = sequence_line[i:i+self.WINDOW_SIZE]
        tmp_t = sequence_line[i+1:i+1+self.WINDOW_SIZE]
        self.x.append(tmp_x)
        self.t.append(tmp_t)
    self.length = len(self.x)
 
  def __len__(self):
    return self.length
 
  def __getitem__(self, sequence):
    return torch.tensor(self.x[sequence]), torch.tensor(self.t[sequence])




class Net(nn.Module):
  HIDDEN_SIZE = 300
  NUM_L = 1
  EMB_DIM = 300

  def __init__(self, sequence_count):
    super().__init__()
    self.sequence_count = sequence_count
    self.hidden = torch.zeros(self.NUM_L, BATCH_SIZE, self.HIDDEN_SIZE) # .cuda()
    self.emb = nn.Embedding(self.sequence_count, self.EMB_DIM, padding_idx=0)
    self.rnn = nn.RNN(self.EMB_DIM, self.HIDDEN_SIZE, batch_first=True, num_layers=self.NUM_L, nonlinearity='relu')
    self.lin = nn.Linear(self.HIDDEN_SIZE, self.sequence_count)
    # self = self.cuda()

  def forward(self, x):
    o = self.emb(x)
    o, self.hidden = self.rnn(o, self.hidden)
    y = self.lin(o)
    return y
  
  def init_hidden(self, batch_size=BATCH_SIZE):
    self.hidden = torch.zeros(self.NUM_L, batch_size, self.HIDDEN_SIZE) # .cuda()

class Training():
  EPOCHS = 100

  def __init__(self, net, dataloader):
    self.net = net
    self.dataloader = dataloader
    self.loss_func = nn.CrossEntropyLoss(ignore_index=0)
    self.optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
  
  def train(self):
    self.net.train()
    for epoch in range(self.EPOCHS):
      for cnt, (x, t) in enumerate(self.dataloader):
        self.optimizer.zero_grad()
        # x = x.cuda()
        # # print("x.shape:", x.shape)
        # t = t.cuda()
        # print("t.shape:", t.shape)
        self.net.init_hidden()
        y = self.net(x)
        # print("y.shape:", y.shape)
        y = y.reshape(-1, self.net.sequence_count)
        t = t.reshape(-1)
        # print("y.reshaped:", y.shape)
        # print("t.reshaped:", t.shape)
        loss = self.loss_func(y, t)
        loss.backward()
        self.optimizer.step()
      print("epoch:", epoch, "\t" , "loss:", loss.item())
    return

class Evaluation():
  
  PREDICT_COUNT = 5

  def __init__(self, net, pdic):
    self.net = net
    self.pdic = pdic
    self.net.eval()

  def calc_accuracy(self, data_loader):
    with torch.no_grad():
      total = 0
      correct = 0

      for batch in data_loader:
        x, t = batch
        # x = x.cuda()
        # t = t.cuda()
        self.net.init_hidden(batch_size=1)
        predicted = self.net(x)
        predicted = predicted.reshape(-1, self.pdic.get_sequence_count())
        probability = torch.softmax(predicted[0], dim=0).cpu().detach().numpy()
        next_product = np.random.choice(self.pdic.get_sequence_count(), p=probability)
        if next_product == 0:
          continue
        next_product = self.pdic.sequence_product[next_product]
        detached_t = t.to('cpu').detach().numpy().copy()
        t_product_sequence = detached_t[0][0]
        if t_product_sequence == 0:
          continue
        t_product = self.pdic.sequence_product[t_product_sequence]

        total += 1
        correct += 1 if next_product == t_product else 0
      accuracy = correct / total
      return accuracy
  
  def predict(self, products):
    sequence_count = self.pdic.get_sequence_count()

    with torch.no_grad():
      predicted = products
      sequences = self.pdic.get_sequences_by_products([products])

      i = 0
      while i < self.PREDICT_COUNT:
        x = torch.tensor(sequences) # .cuda()
        self.net.init_hidden(batch_size=1)
        y = self.net(x)
        y = y.reshape(-1, sequence_count)
        probability = torch.softmax(y[0], dim=0).cpu().detach().numpy()
        next_sequence = np.random.choice(sequence_count, p=probability)
        next_product = self.pdic.sequence_product[next_sequence]

        if next_product in predicted:
          continue
        predicted = predicted + ',' + next_product
        sequences = self.pdic.get_sequences_by_products([predicted])
        i+=1
    return predicted

with open('data.csv') as f:
    products = f.read().splitlines()

pdic = ProductDictionary(products)

sequences = pdic.get_sequences_by_products(products)
# sequences = 
random.shuffle(sequences)

dataset = ProductDataset(sequences)
print(len(dataset))

# 訓練用、テスト用にdatasetを分割
splitted_train, splitted_test = train_test_split(list(range(len(dataset))), test_size=0.4)
train = torch.utils.data.Subset(dataset, splitted_train)
test = torch.utils.data.Subset(dataset, splitted_test)
 
print(len(train))
print(len(test))

# BATCH_SIZE = 250
# 訓練用dataloader作成
dataloader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
# 正解率測定用dataloader作成
train_loader = DataLoader(train, batch_size=1, shuffle=True, drop_last=True)
test_loader = DataLoader(test, batch_size=1, shuffle=True, drop_last=True)

net = Net(pdic.get_sequence_count())
training = Training(net, dataloader)
training.train()

ev = Evaluation(net, pdic)
ev.calc_accuracy(train_loader)
ev.calc_accuracy(test_loader)

ordered_products = '1'
ev.predict(ordered_products)