import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import time

# CNNモデルのデータ拡張ver.

def data_load():
    "**************"
    "(1)"
    val_transform = transforms.Compose([
        transforms.ToTensor(), # dataをtensor型に変換, channelを先頭にする
        transforms.Normalize((0.5,), (0.5,)) # channelごとに画像の平均値が0.5になるようにする
    ])
    train_transform = transforms.Compose([
        # データ拡張のための前処理
        transforms.RandomHorizontalFlip(), # ランダムに左右を入れける
        transforms.ColorJitter(), # ランダムに画像の色値を変える
        transforms.RandomRotation(10), # ランダムに画像の回転を行う(今回は10度)
        transforms.ToTensor(), # dataをtensor型に変換, channelを先頭にする
        transforms.Normalize((0.5,), (0.5,)) # channelごとに画像の平均値が0.5になるようにする
    ])
    
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    validation_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=val_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
    # data_iter = iter(train_dataloader)
    # imgs, labels = data_iter.__next__()
    # print(labels)
    # print("サイズ: {} ...(batch, channel, height, wide)".format(imgs.size()))
    # img = imgs[0]
    # img_permute = img.permute(1, 2, 0) # 軸の順番を入れ替える(一番右を一番最初にする)
    # img_permute = 0.5 * img_permute + 0.5 # 画像を明るく(0.5倍で0.5シフト)
    # img_permute = np.clip(img_permute, 0, 1)
    # plt.imshow(img_permute)
    # plt.show()

    "***** or *****"
    "(2)"
    # train_dataset = datasets.CIFAR10(root='./data', 
    #                                 train=True, 
    #                                 download=True, 
    #                                 # transform=(transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)))) # こっちだとエラー
    #                                 transform=transforms.Compose([  
    #                                                                 # データ拡張のための前処理
    #                                                                 transforms.RandomHorizontalFlip(), # ランダムに左右を入れける
    #                                                                 transforms.ColorJitter(), # ランダムに画像の色値を変える
    #                                                                 transforms.RandomRotation(10), # ランダムに画像の回転を行う(今回は10度)

    #                                                                 transforms.ToTensor(), # dataをtensor型に変換, channelを先頭にする
    #                                                                 transforms.Normalize((0.5,), (0.5,))])) # Compose:順に処理してくれる
    # train_dataloader = torch.utils.data.DataLoader(train_dataset, 
    #                                                 batch_size=32, # 100, 
    #                                                 shuffle=True, 
    #                                                 num_workers=2)  # ミニバッチごとにデータを纏める(学習時にはshuffle=True)
    #                                                 # num_workers:処理の高速化(default=0)

    # testset = datasets.CIFAR10(root='./data', 
    #                             train=False, 
    #                             download=True, 
    #                             # transform=(transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)))) # こっちだとエラー
    #                             transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])) # Compose:順に処理してくれる
    # validation_dataloader = torch.utils.data.DataLoader(testset, 
    #                                                     batch_size=32, # 100, 
    #                                                     shuffle=False, 
    #                                                     num_workers=2)  # ミニバッチごとにデータを纏める(学習時にはshuffle=False)
    #                                                     # num_workers:処理の高速化(default=0)
    "**************"

    names = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

    # # (2)だとエラー =====> 関数化してif name mainの中に入れたらエラー回避
    # data_iter = iter(train_dataloader)
    # imgs, labels = data_iter.__next__()
    # print(labels)
    # print("サイズ: {} ...(batch, channel, height, wide)".format(imgs.size()))
    # img = imgs[0]
    # img_permute = img.permute(1, 2, 0) # 軸の順番を入れ替える(一番右を一番最初にする)
    # img_permute = 0.5 * img_permute + 0.5 # 画像を明るく(0.5倍で0.5シフト)
    # img_permute = np.clip(img_permute, 0, 1)
    # plt.imshow(img_permute)
    # plt.show()

    return train_dataloader, validation_dataloader

def train(train_dataloader, model):
    "(1)"
    # running_loss = 0.0
    # running_acc = 0.0
    "-----"
    "(2)"
    "----- E資格 -----"
    total_correct = 0
    total_data_len = 0
    total_loss = 0
    "-----------------"

    for imgs, labels in train_dataloader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        # optimizer.zero_grad()
        model.optimizer.zero_grad()
        output = model(imgs)
        # loss = criterion(output, labels)
        loss = model.criterion(output, labels)
        loss.backward()
        "(1)"
        # running_loss += loss.item()
        "-----"
        "(2)"
        total_loss += loss.item()
        "-----"
        pred = torch.argmax(output, dim=1) # dim=1:channel方向, dim=0:バッチ方向
        "(1)"
        # running_acc += torch.mean(pred.eq(labels).float())
        "-----"
        "(2)" # より原始的な方法
        batch_size = len(labels)  # バッチサイズの確認
        for i in range(batch_size):  # データ一つずつループ,ミニバッチの中身出しきるまで
            total_data_len += 1  # 全データ数を集計
            if pred[i] == labels[i]:
                total_correct += 1 # 正解のデータ数を集計
        "-----"
        # optimizer.step()
        model.optimizer.step()
    "(1)"
    # running_loss /= len(train_dataloader)
    # running_acc /= len(train_dataloader)
    # losses.append(running_loss)
    # accs.append(running_acc)
    "-----"
    "(2)"
    accuracy = total_correct/total_data_len*100  # 予測精度の算出
    loss = total_loss/total_data_len  # 損失の平均の算出
    print("epoch: {}, loss: {}, acc: {}".format(epoch, loss, accuracy))
    print("*train finish* 正解率: {}".format(accuracy))
    "-----"

def validation(validation_dataloader, model):
    "(1)"
    # val_running_loss = 0.0
    # val_running_acc = 0.0
    "-----"
    "(2)"
    total_val_correct = 0
    total_val_data_len = 0
    total_val_loss = 0
    "-----"

    for val_imgs, val_labels in validation_dataloader:
        val_imgs = val_imgs.to(device)
        val_labels = val_labels.to(device)
        
        val_output = model(val_imgs)
        # val_loss = criterion(val_output, val_labels)
        val_loss = model.criterion(val_output, val_labels)

        "How to (1)"
        # val_running_loss += val_loss.item()
        # val_pred = torch.argmax(val_output, dim=1)
        # val_running_acc += torch.mean(val_pred.eq(val_labels).float())
        "-----"
        "How to (2)"# predの部分がE資格講座と異なる ==> 今回は同じ
        val_pred = torch.argmax(val_output, dim=1) # dimが0だとバッチ方向(縦). 1だと分類方向(横:0~9の分類) # イメージ:[batch, 予測]
        batch_size = len(val_labels)  # バッチサイズの確認
        for i in range(batch_size):  # データ一つずつループ,ミニバッチの中身出しきるまで
            total_val_data_len += 1  # 全データ数を集計
            if val_pred[i] == val_labels[i]:
                total_val_correct += 1 # 正解のデータ数を集計
        "------------------------------"
    "(1)"
    # val_running_loss /= len(validation_dataloader)
    # val_running_acc /= len(validation_dataloader)
    # val_losses.append(val_running_loss)
    # val_accs.append(val_running_acc)
    # print("epoch: {}, loss: {}, acc: {}, val loss: {}, val acc: {}".format(epoch, running_loss, running_acc, val_running_loss, val_running_acc))
    "-----"
    "(2)"
    # 今回のエポックの正答率と損失を求める
    val_accuracy = total_val_correct/total_val_data_len*100  # 予測精度の算出

    print("*validation finish* 正解率: {}".format(val_accuracy))
    "-----"

class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            # size:32*32

            # 通常ver.
            # **1layer**
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2), # in:3, out:64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2), # sizeは1/2になる # size:16*16
            # **1layer**
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), # in:64, out:128
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2), # sizeは1/2になる # size:8*8
            # **1layer**
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1), # in:128, out:256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2), # sizeは1/2になる # size:4*4
            # **1layer**
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1), # in:256, out:128
            nn.ReLU(inplace=True),

            # **軽量化ver.**
            # # **1layer**
            # nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2), # in:3, out:64
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2), # sizeは1/2になる # size:16*16
            # # **1layer**
            # nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), # in:256, out:128
            # nn.ReLU(inplace=True),
        )
        # ここまでが特徴抽出

        # 全結合
        self.classifier = nn.Linear(in_features=4*4*128, out_features=num_classes) # size(h) * size(w) * out_channels

        # **軽量化ver.**
        # self.classifier = nn.Linear(in_features=16*16*128, out_features=num_classes) # size(h) * size(w) * out_channels


        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=5e-4) # 重み付けが大きくなりすぎないようにする
    
    def forward(self, x):
        x = self.features(x) # 4*4サイズ, 128チャネルの画像が出力
        "(1)"
        # x = x.view(x.size(0), -1) # 1次元のベクトルに変換 # size(0) = batch数 = 32
        # ** x = [batch, c * h * w] になる **
        "-----"
        "(2)"
        # x = x.view(num_batches, -1) # 画像の順番になっているので、バッチを先頭に持ってくる. 後ろは任意.
        # x = x.reshape(-1, 4*4*128)  # 画像データを1次元に変換
        # x = x.reshape(-1, 16*16*128)  # 軽量化ver.
        x = x.reshape(x.size(0), -1)  # 軽量化ver.
        "-----"
        x = self.classifier(x)
        return x


if __name__ == "__main__":

    # CPUのコア数を確認
    print("cpu core num: {}".format(os.cpu_count()))  # コア数 # if name main の中じゃないと何回も呼び出されている
    start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataloader, validation_dataloader = data_load()

    model = CNN(10)
    model.to(device)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4) # 重み付けが大きくなりすぎないようにする

    num_epochs = 1 # 15
    "(1)"
    # losses = []
    # accs = []
    # val_losses = []
    # val_accs = []
    "-----"

    use_learned_model = False

    for epoch in range(num_epochs):
        
        if not use_learned_model:
            # train loop
            train(train_dataloader, model)
            # パラメータの保存
            params = model.state_dict()
            torch.save(params, "model.param")
        else:
            # パラメータの読み込み
            param_load = torch.load("model.param")
            model.load_state_dict(param_load)

        # validation loop
        validation(validation_dataloader, model)


    end = time.time()
    print("処理時間: {}".format(end-start))

    # # パラメータの保存
    # params = model.state_dict()
    # torch.save(params, "model.param")

    # plt.style.use("ggplot")
    # plt.plot(losses, label="train loss")
    # plt.plot(val_losses, label="validation loss")
    # plt.legend()
    # plt.show()