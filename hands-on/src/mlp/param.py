import torch
from model_minibatch import MLP
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def test(model, data_loader):
    # # # 今は学習時であることを明示するコード
    # model.eval()

    # ### 追記部分1 ###
    # # 正しい予測数、損失の合計、全体のデータ数を数えるカウンターの0初期化
    # total_correct = 0
    # total_data_len = 0
    # ### ###

    # # ミニバッチごとにループさせる,train_loaderの中身を出し切ったら1エポックとなる
    # for batch_imgs, batch_labels in data_loader:
    #     batch_imgs = batch_imgs.reshape(-1, 28*28*1)  # 画像データを1次元に変換
    #     # labels = torch.eye(10)[batch_labels]  # 正解ラベルをone-hotベクトルへ変換

    #     outputs = model(batch_imgs)  # 順伝播(=予測)
        
    #     # ミニバッチごとの集計
    #     _, pred_labels = torch.max(outputs, axis=1)  # outputsから必要な情報(予測したラベル)のみを取り出す。
    #     batch_size = len(batch_labels)  # バッチサイズの確認
    #     for i in range(batch_size):  # データ一つずつループ,ミニバッチの中身出しきるまで
    #         total_data_len += 1  # 全データ数を集計
    #         if pred_labels[i] == batch_labels[i]:
    #             total_correct += 1 # 正解のデータ数を集計

    # # 今回のエポックの正答率と損失を求める
    # accuracy = total_correct/total_data_len*100  # 予測精度の算出
    # return accuracy
    # ### ###

    # eval
    model.eval()
    num_batches = 5
    total_correct = 0
    total_data_len = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for imgs, labels in data_loader:
        # imgs = imgs.view(num_batches, -1) # 画像の順番になっているので、バッチを先頭に持ってくる. 後ろは任意.
        imgs = imgs.to(device)
        labels = labels.to(device)
        imgs = imgs.reshape(-1, 28*28*1)  # 画像データを1次元に変換
        
        output = model(imgs)
        
        
        # pred = torch.argmax(output, dim=1) # dimが0だとバッチ方向(縦). 1だと分類方向(横:0~9の分類) # イメージ:[batch, 予測]

        # ミニバッチごとの集計
        _, pred_labels = torch.max(output, axis=1)  # outputsから必要な情報(予測したラベル)のみを取り出す。
        batch_size = len(labels)  # バッチサイズの確認
        for i in range(batch_size):  # データ一つずつループ,ミニバッチの中身出しきるまで
            total_data_len += 1  # 全データ数を集計
            if pred_labels[i] == labels[i]:
                total_correct += 1 # 正解のデータ数を集計
    
    # 今回のエポックの正答率と損失を求める
    accuracy = total_correct/total_data_len*100  # 予測精度の算出
    return accuracy
    ### ###
        
    
    # runnin_loss /= len(train_dataloder)
    # running_acc /= len(train_dataloder)
    # losses.append(runnin_loss)
    # accs.append(running_acc)
    # print("epoch: {}, loss: {}, acc: {}".format(epoch, runnin_loss, running_acc))



model = MLP()



# パラメータの読み込み
param_load = torch.load("model.prm")
model.load_state_dict(param_load)


# train_dataset = datasets.MNIST(root="./data")
# train_dataloder = DataLoader(train_dataset, shuffle=True)
testset = datasets.MNIST(root='./test_data', 
                             train=False, 
                             download=True, 
                             transform=transforms.ToTensor())  # 検証用データセット
test_loader = torch.utils.data.DataLoader(testset, 
                                    batch_size=100, 
                                    shuffle=False, 
                                    num_workers=2)  # ミニバッチごとにデータを纏める(学習時にはshuffle=False)
# 評価
test_acc = test(model, test_loader)
print(test_acc)