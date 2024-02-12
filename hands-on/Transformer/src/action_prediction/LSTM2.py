import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("ggplot")

# # df = pd.read_csv("data_pred_3.csv",sep=",")
# df = pd.read_csv("data_pred.csv",sep=",")
# df.columns = ["date", "youbi", "actions"] # ["datetime","id","value"]
# from datetime import datetime as dt
# df.date = df.date.apply(lambda d: dt.strptime(str(d), "%Y%m%d%H%M%S"))

df = pd.read_csv("test_small.csv",sep=",")
# df = pd.read_csv("kabuka_small.csv",sep=",")
df.columns = ["date", "actions"]
from datetime import datetime as dt
df.date = df.date.apply(lambda d: dt.strptime(str(d), "%Y/%m/%d"))
print("df : ", df)

# # plt.scatter(df['youbi'], df['actions'])
# plt.plot(df['date'], df['actions'])
# plt.title('youbi vs actions')
# plt.ylabel('action')
# plt.xlabel('date')
# plt.grid(True)
# plt.show()
print(df["date"])

x = df["date"] # np.linspace(0, 499, 500)
y = df['actions'] # np.sin(x*2 * np.pi / 50)

plt.plot(x, y)
plt.show()

# 一定のシーケンスを持ったデータ列に変換する必要がある
# 例：seq_data=4 の場合、4つのまとまりを一つの入力にする

def make_sequence_data(y, num_sequence):
    num_data = len(y)
    seq_data = []
    target_data = [] # 次に予測したいデータ(seq=5)

    for i in range(num_data - num_sequence):

        seq_data.append(y[i : i+num_sequence]) # iからi+num_sequenceのデータを追加（シーケンス分のデータ）
        target_data.append(y[i+num_sequence : i+num_sequence+1]) # 予測したい次のデータを読み込む
    
    seq_arr = np.array(seq_data)
    target_arr = np.array(target_data)

    return seq_arr, target_arr

class LSTM(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size = 1, hidden_size=self.hidden_size) # in:40個の1次元データを入力([0, ..., 39])
        # lstmへのinputする順番 : seq, batch_size, input_data]

        self.linear = nn.Linear(in_features=self.hidden_size, out_features=1) # hidden_size -> 1つのデータに絞る
    
    def forward(self, x):
        x, _ = self.lstm(x) # LSTMの順伝播を行うと2つの値が返ってくる x:シーケンス長(40)を持ったoutput, _:隠れ層やセルの状態がタプルでまとまって返却
        # x:[40個]　こっちのみ使う
        # _:[[hidden], [cell]]

        x_last = x[-1] # xの最後の値を取得 = 40個の予測された値の一番最後
        # xはシーケンス長の次元(40個の要素を持つ1次元配列)

        x = self.linear(x_last)
        return x


if __name__ == "__main__":

    seq_length = 40 # 30 #  20 # 40 # 今回は40刻みでデータを扱う　短くするほど精度は下がる
    y_seq, y_target = make_sequence_data(y, seq_length)

    print(y_seq.shape)

    num_test = 20 # 30 # 20 # 10

    y_seq_train = y_seq[:-num_test] # -num_testまで=最初の490個を読み込む(0~499のデータ500個のうちの後ろから10個まで)
    y_seq_test = y_seq[-num_test:] # -num_testから=残り10個を読み込んでくれる
    y_target_train = y_target[:-num_test] # 上記同様に0~490
    y_target_test = y_target[-num_test:] # 上記同様に残り10個

    print(y_seq_train)
    
    print(y_seq_train.shape)

    # float Tensor に変換
    y_seq_t = torch.FloatTensor(y_seq_train)
    y_target_t = torch.FloatTensor(y_target_train)

    model = LSTM(100) # hidden_sizeは今回は100

    criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(y_seq_t.size()) # 450, 40だが、[シーケンス長のサイズ, バッチサイズ, inputサイズ] にする必要がある
    # つまり順番が逆
    
    # permuteで入れ替える
    y_seq_t = y_seq_t.permute(1, 0)
    y_target_t = y_target_t.permute(1, 0)

    print(y_seq_t.size()) # 40, 450 : seqが最初に来ている(2次元のTensor)
    # 実際には最後に入力次元数の1が必要
    
    y_seq_t = y_seq_t.unsqueeze(dim=-1) # squeezeは絞る その逆をする dimでどこに挿入するか指定(1番最後に挿入)
    y_target_t = y_target_t.unsqueeze(dim=-1) # squeezeは絞る その逆をする dimでどこに挿入するか指定

    print(y_seq_t.size())


    num_epochs = 100 # 0 # 80
    losses = []
    for epoch in range(num_epochs): # 今回はミニバッチ学習ではなくてバッチ学習
        
        optimizer.zero_grad()
        output = model(y_seq_t)
        
        y_target_t = y_target_t.squeeze(0) # やると警告が出なくなるが、やらなくても大丈夫 
        loss = criterion(output, y_target_t)
        
        # output = transformer(time_data, action_data) # .long() # src, tgt)
        # loss = criterion(output, torch.argmax(y_target_t, dim=1))

        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        if epoch % 10 == 0: # 10step刻みで出力
            print("epoch: {}, loss: {}".format(epoch, loss.item()))
            # print(" *** output: {}, target: {} ***".format(output, y_target_t))
    
    plt.plot(range(num_epochs), losses)
    plt.show()

    y_seq_test_t = torch.FloatTensor(y_seq_test)
    y_seq_test_t = y_seq_test_t.permute(1, 0)
    y_seq_test_t = y_seq_test_t.unsqueeze(dim=-1)
    print(y_seq_test_t.size()) # 40, 10, 1

    y_pred = model(y_seq_test_t)
    print(y_pred.size()) # 10, 1

    plt.plot(x, y)
    # plt.plot(np.arange(490, 500), y_pred.detach())


    # index = df["date"][-10-1:-1:1] # 後ろ10この予測
    index = df["date"][-num_test-1:-1:1] # 後ろ20この予測


    print(index)
    # plt.plot(index, y[-num_test-1:-1:1], label="true")
    plt.plot(index, y_pred.detach(), label="pred") # , color="orange")
    # plt.xlim([450, 500])
    plt.show()
    # plt.legend()
    # plt.savefig('LSTM.png')