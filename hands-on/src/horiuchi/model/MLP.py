import torch
import torch.nn as nn
import torch.optim as optim
import numpy as numpy
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
start = time.time()

# torch.manual_seed(123)



# ***** このコードがmlpのメイン *****
# 一番基礎的なMLP



def data_load():
    "========== 今回のやり方 =========="
    # # 前処理
    # transform = transforms.Compose([
    #     transforms.ToTensor()
    # ])
    # train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    # num_batches = 100
    # train_dataloader = DataLoader(train_dataset, batch_size=num_batches, shuffle=True)
    # train_iter = iter(train_dataloder)

    # imgs, labels = train_iter.__next__()
    # print(imgs.size())
    # # torch.Size([100, 1, 28, 28]) # [batch, channel, 28*28size]
    # print(labels)

    # img = imgs[0]
    # img_permute = img.permute(1, 2, 0) # 軸の順番を入れ替える(一番右を一番最初にする)
    # sns.heatmap(img_permute.numpy()[:, :, 0])
    # # plt.show()
    "========== 今回のやり方 =========="

    "========== E資格のやり方 =========="
    ## こうやってダウンロードして使うことができるよ
    trainset = datasets.MNIST(root='./data', 
                              train=True, 
                              download=True, 
                              transform=transforms.ToTensor())  # 学習用データセット
    train_dataloader = torch.utils.data.DataLoader(trainset, 
                                         batch_size=100, 
                                         shuffle=True, 
                                         num_workers=2)  # ミニバッチごとにデータを纏める(学習時にはshuffle=True)

    testset = datasets.MNIST(root='./data', 
                             train=False, 
                             download=True, 
                             transform=transforms.ToTensor())  # 検証用データセット
    test_dataloader = torch.utils.data.DataLoader(testset, 
                                        batch_size=100, 
                                        shuffle=False, 
                                        num_workers=2)  # ミニバッチごとにデータを纏める(学習時にはshuffle=False)
    "========== E資格のやり方 =========="

    return train_dataloader, test_dataloader


# modelの定義

class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.classifer = nn.Sequential(
            nn.Linear(28*28, 400),
            nn.ReLU(inplace=True),
            nn.Linear(400, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 10)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
    
    def forward(self, x):
        output = self.classifer(x)
        return output

def train(train_dataloader, model):
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    # train
    num_epochs = 15
    losses = [] # 損失
    accs = [] # 精度

    for epoch in range(num_epochs):
        "(1)"
        # runnin_loss = 0.0
        # running_acc = 0.0
        "-----"
        "(2)"
        "----- E資格 -----"
        total_correct = 0
        total_data_len = 0
        total_loss = 0
        "-----------------"

        for imgs, labels in train_dataloader:
            # 画像データを1次元に変換
            # imgs = imgs.view(num_batches, -1) # 画像の順番になっているので、バッチを先頭に持ってくる. 後ろは任意.
            imgs = imgs.reshape(-1, 28*28*1)  # 画像データを1次元に変換
            
            # imgs = imgs.to(device)
            # labels = labels.to(device)
            
            # optimizer.zero_grad()
            model.optimizer.zero_grad()

            output = model(imgs) # 順伝播
            # loss = criterion(output, labels)
            loss = model.criterion(output, labels)
            "(1)"
            # runnin_loss += loss.item()
            "-----"
            "(2)"
            total_loss += loss.item()
            "-----"

            "このやり方ならone-hotベクトルにしなくてもいいかも"
            pred = torch.argmax(output, dim=1) # dimが0だとバッチ方向(縦). 1だと分類方向(横:0~9の分類) # イメージ:[batch, 予測]
            
            "(1)" # 簡単な方法
            # running_acc += torch.mean(pred.eq(labels).float()) # 正解と一致したものの割合=正解率を計算
            "-----"
            "(2)" # より原始的な方法
            batch_size = len(labels)  # バッチサイズの確認
            for i in range(batch_size):  # データ一つずつループ,ミニバッチの中身出しきるまで
                total_data_len += 1  # 全データ数を集計
                if pred[i] == labels[i]:
                    total_correct += 1 # 正解のデータ数を集計
            "-----"

            loss.backward() # 逆伝播で勾配を計算
            # optimizer.step()
            model.optimizer.step() # 勾配をもとにパラメータを最適化
        
        "(1)"
        # runnin_loss /= len(train_dataloader)
        # running_acc /= len(train_dataloader)
        # losses.append(runnin_loss)
        # accs.append(running_acc)
        # print("epoch: {}, loss: {}, acc: {}".format(epoch, runnin_loss, running_acc))
        "-----"
        "(2)"
        accuracy = total_correct/total_data_len*100  # 予測精度の算出
        loss = total_loss/total_data_len  # 損失の平均の算出
        # print(f'正答率: {accuracy}, 損失: {loss}')
        print("epoch: {}, loss: {}, acc: {}".format(epoch, loss, accuracy))
        "-----"

    # plt.legend()
    # plt.plot(losses , label="loss")
    # plt.show()
    # plt.plot(accs, label="acc")
    # plt.show()
    # plt.legend()
    # plt.plot(loss, label="loss")
    # plt.show()
    # plt.plot(accuracy, label="acc")
    # plt.show()

    train_iter = iter(train_dataloader)
    imgs, labels = train_iter.__next__()
    print(labels)

    # パラメータの保存
    params = model.state_dict()
    torch.save(params, "model.param")

def eval(test_dataloader, model):
    # パラメータの読み込み
    param_load = torch.load("model.param")
    model.load_state_dict(param_load)

    total_correct = 0
    total_data_len = 0

    for imgs, labels in test_dataloader: # train_dataloader:

        # 画像データを一次元に変換
        "There are two ways to change data"
        "How to (1)"# E資格講座
        imgs = imgs.reshape(-1, 28*28*1)  # 画像データを1次元に変換
        "How to (2)"# 今回
        # imgs = imgs.view(num_batches, -1) # 画像の順番になっているので、バッチを先頭に持ってくる. 後ろは任意.
        "結局どちらも画像データ(28*28など)を後ろに持ってきている. -1は型に合うように揃えてくれている."
        "---------------------------------"

        # GPUメモリに送信
        # imgs = imgs.to(device)
        # labels = labels.to(device)
        
        output = model(imgs)
        
        # テストデータで検証
        "There are two ways to evaluate"
        "How to (1)"# ミニバッチごとの集計(E資格講座)
        # _, pred_labels = torch.max(output, axis=1)  # outputsから必要な情報(予測したラベル)のみを取り出す。
        # batch_size = len(labels)  # バッチサイズの確認
        # for i in range(batch_size):  # データ一つずつループ,ミニバッチの中身出しきるまで
        #     total_data_len += 1  # 全データ数を集計
        #     if pred_labels[i] == labels[i]:
        #         total_correct += 1 # 正解のデータ数を集計
        "How to (2)"# predの部分がE資格講座と異なる
        pred = torch.argmax(output, dim=1) # dimが0だとバッチ方向(縦). 1だと分類方向(横:0~9の分類) # イメージ:[batch, 予測]
        batch_size = len(labels)  # バッチサイズの確認
        for i in range(batch_size):  # データ一つずつループ,ミニバッチの中身出しきるまで
            total_data_len += 1  # 全データ数を集計
            if pred[i] == labels[i]:
                total_correct += 1 # 正解のデータ数を集計
        "------------------------------"
    
    # 今回のエポックの正答率と損失を求める
    accuracy = total_correct/total_data_len*100  # 予測精度の算出

    print("正解率: {}".format(accuracy))



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(device)

    train_dataloader, test_dataloader = data_load()

    # num_batches = 100
    model = MLP()

    # GPUを使うにはGPU用のメモリに送信する必要がある
    model.to(device)
    # print(model)

    train_model = False

    if train_model: # 学習
        train(train_dataloader, model)
    else: # 評価
        eval(test_dataloader, model)
    

    end = time.time()
    print("処理時間: {}".format(end-start))
    # 処理時間: 150.52174639701843
    # 処理時間: 158.32472777366638 (原始的なやり方)