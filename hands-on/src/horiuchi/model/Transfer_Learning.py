import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import time
from torchvision.models import resnet18, ResNet18_Weights

def data_load():
    "*****"
    "(1)"
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5,), (0.5,))
    # ])

    # train_dataset = datasets.ImageFolder("./hymenoptera_data/train", transform=transform)
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # data_iter = iter(train_loader)
    # imgs, labels = data_iter.__next__()
    # # print(labels)
    # # print("サイズ: {} ...(batch, channel, height, wide)".format(imgs.size()))
    # # img = imgs[0]
    # # img_permute = img.permute(1, 2, 0) # 軸の順番を入れ替える(一番右を一番最初にする)
    # # img_permute = 0.5 * img_permute + 0.5 # 画像を明るく(0.5倍で0.5シフト)
    # # img_permute = np.clip(img_permute, 0, 1)
    # # plt.imshow(img_permute)
    # # plt.show()
    "(2)"
    data_path = "./hymenoptera_data/train"
    train_dataset = datasets.ImageFolder(data_path, 
                                         transform=transforms.Compose([
                                                                        transforms.Resize((224, 224)),
                                                                        transforms.ToTensor(),
                                                                        transforms.Normalize((0.5,), (0.5,))
                                                                        ]))
    train_loader = DataLoader(train_dataset, 
                                    batch_size=32, # ミニバッチごとにデータを纏める
                                    shuffle=True, 
                                    num_workers=2)  # (学習時にはshuffle=True)
                                    # num_workers:処理の高速化(default=0)
    "*****"

    # add validation data
    val_dataset = datasets.ImageFolder(data_path, 
                                         transform=transforms.Compose([
                                                                        transforms.Resize((224, 224)),
                                                                        transforms.ToTensor(),
                                                                        transforms.Normalize((0.5,), (0.5,))
                                                                        ]))
    validation_loader = DataLoader(val_dataset, 
                                    batch_size=32, # ミニバッチごとにデータを纏める
                                    shuffle=True, 
                                    num_workers=2)  # (学習時にはshuffle=True)
                                    # num_workers:処理の高速化(default=0)
    
    return train_loader, validation_loader

def transfer_train(train_loader, model):

    "*****"
    "(1)"
    losses = []
    accs = []
    # running_loss = 0.0
    # running_acc = 0.0
    "(2)"
    "***** E資格 *****"
    total_correct = 0
    total_data_len = 0
    total_loss = 0
    "*****************"

    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        # optimizer.zero_grad()
        model.optimizer.zero_grad()
        output = model(imgs)
        # criterion(output, labels)
        loss = model.criterion(output, labels)

        "*****"
        "(1)"
        # running_loss += loss.item()
        "(2)"
        total_loss += loss.item()
        "*****"

        pred = torch.argmax(output, dim=1)

        "*****"
        "(1)"
        # running_acc += torch.mean(pred.eq(labels).float())
        "(2)" # より原始的な方法
        batch_size = len(labels)  # バッチサイズの確認
        for i in range(batch_size):  # データ一つずつループ,ミニバッチの中身出しきるまで
            total_data_len += 1  # 全データ数を集計
            if pred[i] == labels[i]:
                total_correct += 1 # 正解のデータ数を集計
        "*****"
        loss.backward()
        # optimizer.step()
        model.optimizer.step()
    "*****"
    "(1)"
    # running_loss /= len(train_loader)
    # running_acc /= len(train_loader)
    # losses.append(running_loss)
    # accs.append(running_acc)
    # print("epoch: {}, loss: {}, acc: {}".format(epoch, running_loss, running_acc))
    "(2)"
    # total_data_len != len(train_loader) ... len(train_loader)はミニバッチ何個分回すか
    accuracy = total_correct/total_data_len*100  # 予測精度の算出
    loss = total_loss/total_data_len  # 損失の平均の算出
    print("epoch: {}, loss: {}, acc: {}".format(epoch, loss, accuracy))
    print("*train finish* 正解率: {}".format(accuracy))
    losses.append(loss)
    accs.append(accuracy)
    "*****"

    return losses, accs


def validation(validation_loader, model):
    "(1)"
    # val_running_loss = 0.0
    # val_running_acc = 0.0
    "-----"
    "(2)"
    total_val_correct = 0
    total_val_data_len = 0
    total_val_loss = 0
    "-----"

    for val_imgs, val_labels in validation_loader:
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


if __name__ == "__main__":

    start = time.time()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, validation_loader = data_load()


    # model = models.resnet18(pretrained=True) # 古い書き方
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1) # 新しい書き方
    for param in model.parameters(): # 既存のモデルのパラメータ変更をしない
        param.requires_grad = False

    model.fc = nn.Linear(in_features=512, out_features=2) # 出力前の線形層を変更
    print(model)


    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.fc.parameters(), lr=0.001) # 出力前の変更した層のみ学習(最適化)
    model.criterion = nn.CrossEntropyLoss()
    model.optimizer = optim.Adam(model.fc.parameters(), lr=0.001) # 出力前の変更した層のみ学習(最適化)
    print(model)
    
    num_epochs = 15

    use_learned_model = True # False

    if use_learned_model:
        # パラメータの読み込み
        param_load = torch.load("model.param")
        model.load_state_dict(param_load)

    for epoch in range(num_epochs):
        
        if not use_learned_model:
            # train loop
            losses, accs = transfer_train(train_loader, model) # part of model

        # validation loop
        validation(validation_loader, model)
    
    end = time.time()
    print("処理時間: {}".format(end-start))
    
    if not use_learned_model:
        # パラメータの保存
        params = model.state_dict()
        torch.save(params, "model.param")
        
        # plt.plot(losses, label="loss")
        # plt.legend()
        # plt.show()
        # plt.plot(accs, label="acc")
        # plt.legend()
        # plt.show()