import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import os
import time
from PIL import Image

data_path = "./train"

file_list = os.listdir("./train")
cat_files = [file_name for file_name in file_list if "cat" in file_name]
dog_files = [file_name for file_name in file_list if "dog" in file_name]

transform = transforms.Compose([
    # transforms.Resize((256, 256)), # 入力画像のサイズがバラバラなので、すべて256*256にリサイズする
    transforms.Resize((28, 28)), # 入力画像のサイズがバラバラなので、すべて28*28にリサイズ
    transforms.ToTensor()
])


class CatDogDataset(Dataset):
    def __init__(self, file_list, dir, transform=None):
        # super().__init__()
        self.file_list = file_list
        self.dir = dir
        self.transform = transform
        if "dog" in self.file_list[0]:
            self.label = 1
        else:
            self.label = 0
    
    # 特殊メソッド
    def __len__(self):
        return len(self.file_list) # 画像の枚数を返す
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.dir, self.file_list[idx])
        img = Image.open(file_path)
        if self.transform is not None: # 前処理がある場合は前処理をする
            img = self.transform(img)
        return img, self.label


class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.classifer = nn.Sequential(
            # nn.Linear(28*28, 400),
            nn.Linear(28*28*3, 400),
            nn.ReLU(inplace=True),
            nn.Linear(400, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 2) # 10)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
    
    def forward(self, x):
        x = x.reshape(x.size(0), -1) # データを1次元に変換
        output = self.classifer(x)
        return output

class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            # size:32*32
            # size:28*28

            # 通常ver.
            # **1layer**
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2), # in:3, out:64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2), # sizeは1/2になる # size:16*16 # size:14*14
            # **1layer**
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), # in:64, out:128
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2), # sizeは1/2になる # size:8*8 # size:7*7
            # # **1layer**
            # nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1), # in:128, out:256
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2), # sizeは1/2になる # size:4*4
            # **1layer**
            # nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1), # in:256, out:128
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1), # in:256, out:128
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
        # self.classifier = nn.Linear(in_features=4*4*128, out_features=num_classes) # size(h) * size(w) * out_channels
        self.classifier = nn.Linear(in_features=7*7*64, out_features=num_classes) # size(h) * size(w) * out_channels

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
        x = x.reshape(x.size(0), -1) # データを1次元に変換
        "-----"
        x = self.classifier(x)
        return x

def train(train_dataloader, model):

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

        # 比較用
        # imgs_save

        # 8回ループ
        for imgs, labels in train_dataloader: # 1loopで32(バッチサイズ)個のデータを処理する
            # 画像データを1次元に変換
            # imgs = imgs.view(num_batches, -1) # 画像の順番になっているので、バッチを先頭に持ってくる. 後ろは任意.
            
            
            "これだとimgsと参照しているアドレスが同じで書き換えられてしまうと思ったが大丈夫だった"
            imgs_save = imgs
            
            "***** 入力がカラー画像の時!!!!! 重要 *****"
            # <こっちはグレースケール用> imgs = imgs.reshape(-1, 28*28*1)  # 画像データを1次元に変換 グレースケールではなく、カラーなので、3チャネル!!!!!
            
            "CNNを使うときはモデルの中ですでに一次元に変換しているのでコメントアウト -> MLPもモデルの中に記述"
            # imgs = imgs.reshape(-1, 28*28*3)  # 画像データを1次元に変換 グレースケールではなく、カラーなので、3チャネル!!!!!
            "imgs = [32(batch), 28*28(size) *28(channel)]"

            "***** 入力がカラー画像の時!!!!! 重要 *****"
            
            # imgs = imgs.to(device)
            # labels = labels.to(device)
            
            # optimizer.zero_grad()
            model.optimizer.zero_grad()

            output = model(imgs) # 順伝播
            # loss = criterion(output, labels)

            
            "表示用に追加"
            # print(imgs, len(imgs)) # 96
            # print(labels, len(labels)) # 32
            # print("out:{}, labels:{}".format(output, labels)) # 100, 32
            # red = torch.argmax(output, dim=1)
            # print("red:{}".format(red)) # 100


            loss = model.criterion(output, labels)
            "(1)"
            # runnin_loss += loss.item()
            "-----"
            "(2)"
            total_loss += loss.item()
            "-----"

            "このやり方ならone-hotベクトルにしなくてもいいかも"
            pred = torch.argmax(output, dim=1) # dimが0だとバッチ方向(縦). 1だと分類方向(横:0~9の分類) # イメージ:[batch, 予測]
            # 1バッチ分 = 32 個のデータ分を 0, 1 判定
            # print("  pred:{}".format(pred))
            
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
            # print("total_data:{}".format(total_data_len))
        
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

    # train_iter = iter(train_dataloader)
    # imgs, labels = train_iter.__next__()
    # print(labels)

    # # パラメータの保存
    # params = model.state_dict()
    # torch.save(params, "model.param")
    

    # 最後の予測結果32個と正解ラベル32個を比較
    print("*************************************************************************************************************")
    print("予測結果  : {}".format(pred))
    
    # data_iter = iter(train_dataloader)
    # imgs, labels = data_iter.__next__() # 1バッチ分表示(size=32)
    print("正解ラベル: {}".format(labels))
    print("*************************************************************************************************************")
    grid_imgs = torchvision.utils.make_grid(imgs_save[:32]) # 24]) # 32枚表示
    grid_imgs_arr = grid_imgs.numpy()
    plt.figure(figsize=(16, 24))
    plt.imshow(np.transpose(grid_imgs_arr, (1, 2, 0)))
    plt.show()
    print("アドレス: {}, {}".format(id(imgs), id(imgs_save)))
    print("type: {}".format(type(imgs_save)))

def validation(val_dataloader, model):
    # # パラメータの読み込み
    # param_load = torch.load("model.param")
    # model.load_state_dict(param_load)

    total_correct = 0
    total_data_len = 0

    for imgs, labels in val_dataloader: # train_dataloader:

        # 画像データを一次元に変換
        "There are two ways to change data"
        "How to (1)"# E資格講座

        # imgs = imgs.reshape(-1, 28*28*1)  # 画像データを1次元に変換
        "***** 入力がカラー画像の時!!!!! 重要 *****"
        # imgs = imgs.reshape(-1, 28*28*1)  # 画像データを1次元に変換 グレースケールではなく、カラーなので、3チャネル!!!!!
        
        "CNNを使うときはモデルの中ですでに一次元に変換しているのでコメントアウト -> MLPもモデルの中に記述"
        # imgs = imgs.reshape(-1, 28*28*3)  # 画像データを1次元に変換 グレースケールではなく、カラーなので、3チャネル!!!!!
        "imgs = [32(batch), 28*28(size) *28(channel)]"

        "***** 入力がカラー画像の時!!!!! 重要 *****"
        
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
    
    dir_path = "train/"
    cat_dataset = CatDogDataset(cat_files, dir_path, transform=transform)
    dog_dataset = CatDogDataset(dog_files, dir_path, transform=transform)

    # 二つのデータセットを結合して一つのデータセットにする
    cat_dog_dataset = ConcatDataset([cat_dataset, dog_dataset])

    # DataLoader作成
    data_loader = DataLoader(cat_dog_dataset, batch_size=32, shuffle=True)
    
    data_iter = iter(data_loader)
    imgs, labels = data_iter.__next__() # 1バッチ分表示(size=32)
    # print("imgs: {}".format(imgs))
    print(labels)
    grid_imgs = torchvision.utils.make_grid(imgs[:32]) # 24]) # 32枚表示
    grid_imgs_arr = grid_imgs.numpy()
    plt.figure(figsize=(16, 24))
    plt.imshow(np.transpose(grid_imgs_arr, (1, 2, 0)))
    plt.show()

    "*** Add ***"
    # DataLoder作成　(評価用)
    validation_loader = DataLoader(cat_dog_dataset, batch_size=32, shuffle=False)
    # val_iter = iter(validation_loader)
    # val_imgs, val_labels = val_iter.__next__() # 1バッチ分表示(size=32)
    # print(val_labels)
    # grid_imgs = torchvision.utils.make_grid(val_imgs[:32]) # 24]) # 32枚表示
    # grid_imgs_arr = grid_imgs.numpy()
    # plt.figure(figsize=(16, 24))
    # plt.imshow(np.transpose(grid_imgs_arr, (1, 2, 0)))
    # plt.show()


    # model = MLP()
    model = CNN(2)

    # GPUを使うにはGPU用のメモリに送信する必要がある
    # model.to(device)

    train_model = True # False

    if train_model: # 学習
        train(data_loader, model)
    else: # 評価
        validation(validation_loader, model)

    # print(len(data_loader)) # 8
    # for imgs, labels in data_loader: # 1epochで17回 (32バッチ×17=544)
    #     pass
    # print(len(imgs), len(labels))

    
    # print(imgs[0], labels[0])
    # print(imgs, labels)