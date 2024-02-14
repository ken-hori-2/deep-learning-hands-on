#ライブラリのインポート
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer

#ランダムシードの設定
fix_seed = 2023
np.random.seed(fix_seed)
torch.manual_seed(fix_seed)

#デバイスの設定
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

multi_data = False # True

if not multi_data:
    "*** 変更前 ***"
    df = pd.read_csv("test_small.csv",sep=",")
    # df = pd.read_csv("gouseizyusi.csv",sep=",")
    # df = pd.read_csv("kabuka_small.csv",sep=",")
    df.columns = ["date", "actions"]
    from datetime import datetime as dt
    df.date = df.date.apply(lambda d: dt.strptime(str(d), "%Y/%m/%d"))
    print("df : ", df)
    # plt.plot(df['date'], df['actions'])
    # plt.show()

    # #機械受注長期時系列
    # machinary_order_data = 'machine_data.csv'
    # df = pd.read_csv(machinary_order_data, sep=",")
    # print(df)
    # df.columns = ["年","月","風水力機械","運搬機械","産業用ロボット","金属加工機械","化学機械","冷凍機械","合成樹脂","繊維機械","建設機械","鉱山機械","農林用機械","その他"] # ["datetime","id","value"]
    # plt.plot(df["産業用ロボット"]) # .values) # , df['actions'])
    # plt.show()
else:
    "*** 変更後 ***"
    import pandas as pd
    df1 = pd.read_csv("kabuka_small_add_day.csv",sep=",")
    df2 = pd.read_csv("gouseizyusi.csv",sep=",")
    df3 = pd.read_csv("test_small.csv",sep=",") # pd.read_csv("kabuka_small.csv",sep=",")
    df1.columns = ["date", "actions"]
    df2.columns = ["date", "actions"]
    df3.columns = ["date", "actions"]
    from datetime import datetime as dt
    df1.date = df1.date.apply(lambda d: dt.strptime(str(d), "%Y/%m/%d"))
    df2.date = df2.date.apply(lambda d: dt.strptime(str(d), "%Y/%m/%d"))
    df3.date = df3.date.apply(lambda d: dt.strptime(str(d), "%Y/%m/%d"))
    # 時系列データを結合して入力とする
    # input_data = torch.stack((time_series1, time_series2, time_series3), dim=1)
    # input_data = torch.stack((df1, df2, df3), dim=1)
    print("*****")
    print(df1)
    print("*****")
    print(df2)
    print("*****")
    print(df3)
    print("*****")

# データのロードと実験用の整形
class AirPassengersDataset(Dataset):
    def __init__(self, flag, seq_len, pred_len):
        #学習期間と予測期間の設定
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        #訓練用、評価用、テスト用を分けるためのフラグ
        type_map = {'train': 0, 'val': 1, 'test':2}
        self.set_type = type_map[flag]

        self.__read_data__()

    def __read_data__(self):
        
        if not multi_data:
            #seabornのデータセットから飛行機の搭乗者数のデータをロード
            df_raw = df # sns.load_dataset('flights')
            # print("df : ", df_raw)

            "*****"
            #訓練用、評価用、テスト用で呼び出すデータを変える
            # # Qitta
            # border1s = [0, 12 * 9 - self.seq_len, 12 * 11 - self.seq_len]
            # border2s = [12 * 9, 12 * 11, 12 * 12]
            # # border1s = [0, 12 * 4 - self.seq_len, 12 * 11 - self.seq_len] # 0, 48-36=12, 132-36=96
            # # border2s = [12 * 4, 12 * 11, 12 * 6] # 48, 132, 144

            # border1s = [0, 12 * 4 - self.seq_len, 12 * 11 - self.seq_len] # 0, 48-36=12, 132-36=96
            # border2s = [12 * 4, 12 * 11, 12 * 6] # 48, 132, 144

            data_len = len(df_raw)
            print("data len : {}".format(data_len))
            border1s = [0, 56-self.seq_len, 50]
            border2s = [56, 38+30, 74]


            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            # data = df_raw[['passengers']].values
            data = df_raw[['actions']].values
            print("d : ", data)
            ss = StandardScaler()
            data = ss.fit_transform(data)
            print("border1: {}, boredr2: {}".format(border1, border2))
            self.data = data[border1:border2]

            # print("len: {}".format(len(data)))
            # self.data = data[int(len(data)*0.7):len(data)-int(len(data)*0.7)] # *0.3]
            "*****"
            # # 試しにかっこを一つ削除
            # data = df_raw[['actions']].values # *0.001 大きすぎるとnanになる
            # # data = df_raw['actions'].values # *0.001 大きすぎるとnanになる

            
            # self.data = data
            "*****"
            print(self.data)
        else:
            input_data = torch.stack((torch.tensor(df1['actions'].values), torch.tensor(df2['actions'].values), torch.tensor(df3['actions'].values)), dim=1)
            data = input_data
            self.data = data
    
    def __getitem__(self, index):
        #学習用の系列と予測用の系列を出力
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        src = self.data[s_begin:s_end]
        tgt = self.data[r_begin:r_end]

        return src, tgt
    
    def __len__(self):
        # print("data : ", self.data)
        # print("len : ", len(self.data) - self.seq_len - self.pred_len + 1)
        return len(self.data) - self.seq_len - self.pred_len + 1

# DataLoaderの定義
def data_provider(flag, seq_len, pred_len, batch_size):
    #flagに合ったデータを出力
    data_set = AirPassengersDataset(flag=flag, 
                                    seq_len=seq_len, 
                                    pred_len=pred_len
                                   )
    #データをバッチごとに分けて出力できるDataLoaderを使用
    data_loader = DataLoader(data_set,
                             batch_size=batch_size, 
                             shuffle=True
                            )
    
    print("data set : {}".format(data_set))
    
    return data_loader

flag = "train"
src_len = 18 # 36 # 3年分のデータから
tgt_len = 6 # 12 # 1年先を予測する
batch_size = 1

data = data_provider(flag, src_len, tgt_len, batch_size)

# import matplotlib.pyplot as plt

# plt.plot(data, label="inputdata")
# # plt.plot(input_data, label=("kabuka","gouseizyusi","robotics"))
# plt.legend()
# plt.show()
# # plt.savefig("multi_input.png")

print(data)