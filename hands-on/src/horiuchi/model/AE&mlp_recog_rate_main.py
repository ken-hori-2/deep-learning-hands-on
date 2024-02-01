import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import time
import seaborn as sns
# from torchvision.models import resnet18, ResNet18_Weights


# AE で元画像に似た画像を生成
# mlp でその画像が本来のラベルと一致しているかを判別
# 整理ver.



def data_load():
    "*****"
    "(1)"
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    # ])

    # train_dataset = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    "(2)"
    data_path = "./data"
    train_dataset = datasets.MNIST(root=data_path, 
                                    train=True, 
                                    download=True,
                                    transform=transforms.Compose([                                                                        
                                                                transforms.ToTensor(),                                                                        
                                                                ]))
    train_loader = DataLoader(train_dataset, 
                                batch_size=32, # ミニバッチごとにデータを纏める
                                shuffle=True, 
                                num_workers=2)  # (学習時にはshuffle=True)
                                # num_workers:処理の高速化(default=0)
    "*****"

    # add validation data
    val_dataset = datasets.MNIST(root=data_path, 
                                    train=True, 
                                    download=True,
                                    transform=transforms.Compose([                                                                        
                                                                transforms.ToTensor(),                                                                        
                                                                ]))
    validation_loader = DataLoader(val_dataset, 
                                    batch_size=32, # ミニバッチごとにデータを纏める
                                    shuffle=True, 
                                    num_workers=2)  # (学習時にはshuffle=True)
                                    # num_workers:処理の高速化(default=0)
    
    return train_loader, validation_loader

def train(train_loader, model,     ae_result_imgs, ae_result_labels): # 復元画像とそのラベル(正解=元画像)を保存

    "* add *"
    # ae_result = []
    num = 0


    running_loss = 0.0
    for imgs, labels in train_loader: # 教師は元画像=labelsは使わないので、"_"
    # for imgs, _ in train_loader:
        # imgs = imgs.to(device)
        # labels = labels.to(device)

        # optimizer.zero_grad()
        model.optimizer.zero_grad()
        output = model(imgs)

        # 重要 - これまでと違うところ -
        "***** Warning *****"
        # loss = criterion(output, labels)
        # loss = criterion(output, imgs) # 教師は自分自身 (再構成した画像と元の画像)
        loss = model.criterion(output, imgs) # 教師は自分自身 (再構成した画像と元の画像)
        "***** Warning *****"

        running_loss += loss.item()
        loss.backward()
        # optimizer.step()
        model.optimizer.step()

        "* add *" # outputとlabelsが一致したときのみ保存するのもありかも？->そうするとただの画像認識になってしまうので、どの割合でそれらしい画像を生成できたかは判別できなくなってしまう
        ae_result_imgs.append(output)
        ae_result_labels.append(labels)
        num += 1

    running_loss /= len(train_loader) # ミニバッチの数で割る
    losses.append(running_loss)

    print("epoch: {}, loss: {}".format(epoch, running_loss))

    "* add *"
    print("num: {}".format(num)) # 1875
    print("batch size: {}".format(len(imgs))) # 32
    print("size: {}, {}".format(len(ae_result_imgs[0]), len(ae_result_labels[0]))) # 1875, 1875
    # print("output size: {}, {}".format(len(output), output[0])) # 32
    return ae_result_imgs, ae_result_labels
    
def eval(test_dataloader, model, ae_result_imgs, ae_result_labels):
    # パラメータの読み込み
    param_load = torch.load("mnist_model.param")
    model.load_state_dict(param_load)

    total_correct = 0
    total_data_len = 0

    num = 0 # バッチ数(total_data_lenと同じではない)

    # for imgs, labels in test_dataloader: # train_dataloader:
    for _, _ in test_dataloader: # train_dataloader: # 1875回loop
        imgs = torch.tensor(ae_result_imgs[num])
        labels = torch.tensor(ae_result_labels[num])

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
        
        "***** 重要 *****"
        output = model(imgs) # 0~9の10種類のうちそれぞれ何%かの確率を出力
        "***** 重要 *****"
        
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

        num += 1
    
    # 今回のエポックの正答率と損失を求める
    accuracy = total_correct/total_data_len*100  # 予測精度の算出

    print("[Batch Size 32 × {}loop中] <{}/{}一致> ... 正解率: {}".format(num, total_correct, total_data_len, accuracy))
    # print("imgs:{}".format(ae_result_imgs[0]))
    print("pred:{}, labels:{}".format(pred[0], labels[0]))
    print("1. labels(num):{}".format(labels[0]))
    # print("labels:{}, {} : {}%:max:{}".format(ae_result_labels[0], ae_result_labels[0][0], output[0], pred[0])) # 32
    print("2. labels(%):{} : max:{}".format(output[0], pred[0]))
    # print("labels:{}".format(ae_result_labels))

    print("*****************************************")
    print("num: {}".format(num)) # 1875
    print("batch size: {}".format(len(imgs))) # 32
    print("size: {}, {}".format(len(ae_result_imgs[0]), len(ae_result_labels[0]))) # 1875, 1875
    print("output size: {}, 中身: {}, max: {}".format(len(output), output[0], pred[0])) # 32
    print("*****************************************")


class ConvAE(nn.Module):

    def __init__(self):
        super().__init__()

        self.en = nn.Sequential(
            # size:28*28
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # size:14*14

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # size:7*7 (これ以上2で割り切れない)

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.de = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            nn.Tanh(), # こっちの方が精度がいいらしい
        )

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
    
    def forward(self, x):
        x = self.en(x)
        x = self.de(x)

        return x

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


if __name__ == "__main__":

    start = time.time()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, validation_loader = data_load()

    model = ConvAE()
    # model = model.to(device)

    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 15
    losses = []

    use_learned_model = True # False

    if use_learned_model:
        # パラメータの読み込み
        param_load = torch.load("model.param")
        model.load_state_dict(param_load)

    for epoch in range(num_epochs):

        if not use_learned_model:
            # train loop
            train(train_loader, model)

        else: # 追加
            
            "*add*" # 1epoch 1875*32=60000回ごとに正解率を評価するための配列を作成(出力例: n個一致/60000個中)
            ae_result_imgs = []
            ae_result_labels = []
            ae_result_imgs, ae_result_labels = train(train_loader, model, ae_result_imgs, ae_result_labels)
            eval_mnist_model = MLP()
            eval(validation_loader, eval_mnist_model, ae_result_imgs, ae_result_labels)

            print("***** validation result *****")
    
    
    
    end = time.time()
    print("処理時間: {}".format(end-start))

    if not use_learned_model:
        # パラメータの保存
        params = model.state_dict()
        torch.save(params, "model.param")

    # data_iter = iter(train_loader)
    # imgs, _ = data_iter.__next__()
    # img = imgs[0]
    # img_permute = img.permute(1, 2, 0) # channel lastに軸の順番を変更
    # sns.heatmap(img_permute[:, :, 0]) # heatmapでは2次元で出力するために変換
    # plt.show()

    # x_en = model.en(imgs.to(device)) # modelがGPU上にある場合はGPUに転送
    # x_en2 = x_en[0].permute(1, 2, 0)
    # sns.heatmap(x_en2[:, :, 0].detach().to("cpu")) # forwardして計算しているので、勾配計算を切り離す操作(detach)をする
    # plt.show()

    # x_ae = model(imgs.to(device)) # エンコードしてデコードしたものが返ってくる
    # sns.heatmap(x_ae[0].permute(1, 2, 0).detach().to("cpu")[:, :, 0]) # channel lastに変換, 勾配計算切り離し, GPU上のデータをCPUにコピー, 2次元データに変換
    # plt.show()