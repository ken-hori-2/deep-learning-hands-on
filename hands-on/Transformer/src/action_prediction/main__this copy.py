
# 【PyTorch】Transformerによる時系列予測

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
# fix_seed = 2023
# np.random.seed(fix_seed)
# torch.manual_seed(fix_seed)

#デバイスの設定
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')




"main.pyの整理ver."



# メモ
# 学習率の大きさは予測したいデータの大きさによって変える
# 一定以上誤差が大きくなるとnanになってしまう


multi_data = False # True

if not multi_data:
    "*** 変更前 ***"
    df = pd.read_csv("test_small.csv",sep=",")

    df = pd.read_csv("test_only_action.csv",sep=",")


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
            # #訓練用、評価用、テスト用で呼び出すデータを変える
            # border1s = [0, 12 * 4 - self.seq_len, 12 * 11 - self.seq_len] # 0, 108-3, 132-3
            # border2s = [12 * 4, 12 * 11, 12 * 12]
            # border1 = border1s[self.set_type]
            # border2 = border2s[self.set_type]
            # # data = df_raw[['passengers']].values
            # data = df_raw[['actions']].values
            # print("d : ", data)
            # ss = StandardScaler()
            # data = ss.fit_transform(data)
            # print("border1: {}, boredr2: {}".format(border1, border2))
            # self.data = data[border1:border2]
            "*****"
            data_len = len(df_raw)
            # print("data len : {}".format(data_len))
            border1s = [0, 56-self.seq_len, 50] # [train0, val0, test0] # test0~test1=24
            border2s = [56, 38+30, 74]          # [train1, val1, test1]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            data = df_raw[['actions']].values
            ss = StandardScaler()
            data = ss.fit_transform(data)
            # print("border1: {}, boredr2: {}".format(border1, border2))
            self.data = data[border1:border2]

            # 試しにかっこを一つ削除
            data = df_raw[['actions']].values # *0.001 大きすぎるとnanになる
            # data = df_raw['actions'].values # *0.001 大きすぎるとnanになる

            
            self.data = data
            "*****"
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
                            #  shuffle=True # これが原因
                            shuffle=False # これが原因
                            )
    
    return data_loader

# エンべディングの定義
#位置エンコーディングの定義
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout: float = 0.1, max_len: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

#モデルに入力するために次元を拡張する
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = nn.Linear(c_in, d_model) 

    def forward(self, x):
        x = self.tokenConv(x)
        return x

# Transformerの定義
class Transformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers,
        d_model, d_input, d_output,
        dim_feedforward = 512, dropout = 0.1, nhead = 8):
        
        super(Transformer, self).__init__()
        

        #エンべディングの定義
        self.token_embedding_src = TokenEmbedding(d_input, d_model)
        self.token_embedding_tgt = TokenEmbedding(d_output, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        #エンコーダの定義
        encoder_layer = TransformerEncoderLayer(d_model=d_model, 
                                                nhead=nhead, 
                                                dim_feedforward=dim_feedforward,
                                                dropout=dropout,
                                                batch_first=True,
                                                activation='gelu'
                                               )
        encoder_norm = LayerNorm(d_model)
        self.transformer_encoder = TransformerEncoder(encoder_layer, 
                                                      num_layers=num_encoder_layers,
                                                      norm=encoder_norm
                                                     )
        
        #デコーダの定義
        decoder_layer = TransformerDecoderLayer(d_model=d_model, 
                                                nhead=nhead, 
                                                dim_feedforward=dim_feedforward,
                                                dropout=dropout,
                                                batch_first=True,
                                                activation='gelu'
                                               )
        decoder_norm = LayerNorm(d_model)
        self.transformer_decoder = TransformerDecoder(decoder_layer, 
                                                      num_layers=num_decoder_layers, 
                                                      norm=decoder_norm)
        
        #出力層の定義
        self.output = nn.Linear(d_model, d_output)
        

    def forward(self, src, tgt, mask_src, mask_tgt):
        #mask_src, mask_tgtはセルフアテンションの際に未来のデータにアテンションを向けないためのマスク
        
        embedding_src = self.positional_encoding(self.token_embedding_src(src))
        memory = self.transformer_encoder(embedding_src, mask_src)
        
        embedding_tgt = self.positional_encoding(self.token_embedding_tgt(tgt))
        outs = self.transformer_decoder(embedding_tgt, memory, mask_tgt)
        
        output = self.output(outs)
        return output

    def encode(self, src, mask_src):
        return self.transformer_encoder(self.positional_encoding(self.token_embedding_src(src)), mask_src)

    def decode(self, tgt, memory, mask_tgt):
        return self.transformer_decoder(self.positional_encoding(self.token_embedding_tgt(tgt)), memory, mask_tgt)

# マスクの定義
def create_mask(src, tgt):
    
    seq_len_src = src.shape[1]
    seq_len_tgt = tgt.shape[1]

    mask_tgt = generate_square_subsequent_mask(seq_len_tgt).to(device)
    mask_src = generate_square_subsequent_mask(seq_len_src).to(device)

    return mask_src, mask_tgt


def generate_square_subsequent_mask(seq_len):
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
    return mask

# 訓練、評価の処理を定義
def train(model, data_provider, optimizer, criterion):
    model.train()
    total_loss = []
    for src, tgt in data_provider:
        
        src = src.float().to(device)
        tgt = tgt.float().to(device)

        input_tgt = torch.cat((src[:,-1:,:],tgt[:,:-1,:]), dim=1)

        mask_src, mask_tgt = create_mask(src, input_tgt)

        output = model(
            src=src, tgt=input_tgt, 
            mask_src=mask_src, mask_tgt=mask_tgt
        )

        optimizer.zero_grad()

        loss = criterion(output, tgt)
        loss.backward()
        total_loss.append(loss.cpu().detach())
        optimizer.step()
        
    return np.average(total_loss)


def evaluate(flag, model, data_provider, criterion):
    model.eval()
    total_loss = []
    # # print(data_provider)
    # # print(data_provider.shape())
    # val_iter = iter(data_provider)
    # val_imgs, val_labels = val_iter.__next__() # 1バッチ分表示(size=32)
    # print("*****")
    # print(val_labels)
    # print("***************")
    # Iter = iter(data_provider)
    # xdata, ydata = next(Iter) #教師データ、ラベルデータ
    # print(xdata.shape, type(xdata))
    # print(xdata) 
    # #torch.Size([3, 64]) <class 'torch.Tensor'>
    # print("*****")

    # print(ydata.shape)
    # print(ydata)
    # #torch.Size([3]) tensor([0, 1, 2])
    # print("*****")

    for src, tgt in data_provider:
        
        src = src.float().to(device)
        tgt = tgt.float().to(device)

        seq_len_src = src.shape[1]
        mask_src = (torch.zeros(seq_len_src, seq_len_src)).type(torch.bool)
        mask_src = mask_src.float().to(device)
    
        memory = model.encode(src, mask_src)
        outputs = src[:, -1:, :]
        seq_len_tgt = tgt.shape[1]
    
        for i in range(seq_len_tgt - 1):
        
            mask_tgt = (generate_square_subsequent_mask(outputs.size(1))).to(device)
        
            output = model.decode(outputs, memory, mask_tgt)
            output = model.output(output)

            outputs = torch.cat([outputs, output[:, -1:, :]], dim=1) # ここで予測を1つずつ追加している
            # print("TEST 2024/02/10")
        
        loss = criterion(outputs, tgt)
        total_loss.append(loss.cpu().detach())
        print("out : ", output)
        
    if flag=='test':
        print("src: ", src)
        true = torch.cat((src, tgt), dim=1)
        print("true: ", true)
        pred = torch.cat((src, output), dim=1)
        # pred = torch.cat((src, outputs), dim=1) # ???
        print("target: ", tgt)
        print("output: ", output.detach().numpy())
        print("pred  : ", pred.detach().numpy())
        # plt.plot(true.squeeze().cpu().detach().numpy(), label='true')
        # plt.plot(pred.squeeze().cpu().detach().numpy(), label='pred')
        # plt.style.use("ggplot")
        plt.plot(df['date'][0:-1:1], df['actions'][0:-1:1], label='true')
        "***"
        "main__this.py ver."
        index = df["date"][-(seq_len_src)-2:-(seq_len_tgt):1] # -37 ~ -11
        index2 = df["date"][-(seq_len_tgt)-1:-2:1] # -11 ~ の予測
        plt.plot(index, pred.squeeze().cpu().detach().numpy()[-(seq_len_src)-1:-(seq_len_tgt-1):1], label='src(input)') # 'true') # , color="blue") # true.detach())
        plt.plot(index2, pred.squeeze().cpu().detach().numpy()[-(seq_len_tgt):-1:1], label='pred(output)')
        index_tgt = df["date"][-(seq_len_tgt)-1:-1:1] # target
        plt.plot(index_tgt, true.squeeze().cpu().detach().numpy()[-(seq_len_tgt)-1:-1:1], label='tgt(true)', alpha=0.5) # , color="green") # true.detach())
        "***"
        plt.legend()
        plt.grid(True)
        
        # コメントアウト
        # pred = torch.cat((src, outputs), dim=1) # ???
        # print("target: ", tgt)
        # print("output: ", output.detach().numpy())
        # print("pred  : ", pred.detach().numpy())
        # # plt.plot(true.squeeze().cpu().detach().numpy(), label='true')
        # # plt.plot(pred.squeeze().cpu().detach().numpy(), label='pred')
        # # plt.legend()

        # data_len = src_len + tgt_len # -1 # 47
        # # plt.style.use("ggplot")
        # plt.plot(df['date'][0:-1:1], df['actions'][0:-1:1], label='true')
        # "***"
        # "main__.py ver."
        # index = df["date"][-(seq_len_src)-1:-(seq_len_tgt):1] # -37 ~ -11
        # index2 = df["date"][-(seq_len_tgt)-1:-1:1] # -11 ~ の予測
        # # plt.plot(df["date"][-48:data_len:1], pred.squeeze().cpu().detach().numpy()) # [-(seq_len_src)-1:-(seq_len_tgt -1):1], label='src(input)') # 'true') # , color="blue") # true.detach())
        # # # plt.plot(index2, pred.squeeze().cpu().detach().numpy()) # [-(seq_len_tgt -1)-1:-1:1], label='pred(output)') # , color="green") # true.detach())
        # # index_tgt = df["date"][-(seq_len_tgt -1)-1:-1:1] # target
        # # plt.plot(df["date"][-48:data_len:1], true.squeeze().cpu().detach().numpy()) # [-(seq_len_tgt )-1:-2:1], label='tgt(true)', alpha=0.5) # , color="green") # true.detach())
        # plt.plot(index, pred.squeeze().cpu().detach().numpy()[-(seq_len_src)-1:-(seq_len_tgt):1], label='src(input)') # 'true') # , color="blue") # true.detach())
        # plt.plot(index2, pred.squeeze().cpu().detach().numpy()[-(seq_len_tgt)-1:-1:1], label='pred(output)')
        # index_tgt = df["date"][-(seq_len_tgt)-1:-1:1] # target
        # plt.plot(index_tgt, true.squeeze().cpu().detach().numpy()[-(seq_len_tgt)-1:-1:1], label='tgt(true)', alpha=0.5) # , color="green") # true.detach())
        # "***"
        # plt.legend()
        # plt.grid(True)
        # コメントアウト

        
        
        
        # y軸を行動に変更, 次の行動を予測
        " *** ADD *** "
        # next_action_list = [ "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R"]
        next_action_list = [ "sleep", "wakeup", "breakfast", "tooth brush", "go work", "coffee", "lunch", "coffee", "tooth brush", "drink alchol", "back home", "bath", "TV/Youtube", "study(C++)", "study(AWS)", "study(ML)", "study(other)", "go bed", "TV/Youtube", "sleeping"]
        plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], next_action_list)
        print("outputs:", outputs)
        print("output: ", output)
        print("target: ", tgt)
        print("**********")
        next_action = output.detach().numpy()
        print("next action pred1: ", next_action[0][0][0])
        print("next action pred2: ", next_action[0][1][0])
        print("next action pred3: ", next_action[0][2][0])
        # print("next action pred1: ", int(next_action[0][0]))
        # next_action_list = [ "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R"]
        print("(t+1)next action is ... ", next_action_list[round(next_action[0][0][0])])
        print("(t+2)next action is ... ", next_action_list[round(next_action[0][1][0])])
        print("(t+3)next action is ... ", next_action_list[round(next_action[0][2][0])])
        " *** ADD *** "

        plt.savefig('pred_2_2.png')
        # plt.savefig('pred.pdf')
    
    # print("test : {}".format(seq_len_src))
        
    return np.average(total_loss)

if __name__ == "__main__":
    
    # パラメータなどの定義
    d_input = 1
    d_output = 1
    d_model = 512
    nhead = 8
    dim_feedforward = 2048
    num_encoder_layers = 1
    num_decoder_layers = 1
    dropout = 0.01
    src_len = 36 # 18 # 3年分のデータから
    tgt_len = 12 # 6 # 1年先を予測する
    batch_size = 1
    epochs = 3 # 30 # 100 # 5 # 0 # 30 # 5 # 0 # 30+70 # 300
    best_loss = float('Inf')
    best_model = None

    model = Transformer(num_encoder_layers=num_encoder_layers,
                        num_decoder_layers=num_decoder_layers,
                        d_model=d_model,
                        d_input=d_input, 
                        d_output=d_output,
                        dim_feedforward=dim_feedforward,
                        dropout=dropout, nhead=nhead
                    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    model = model.to(device)

    criterion = torch.nn.MSELoss() # オーバーフローに注意

    optimizer = torch.optim.RAdam(model.parameters(), lr=0.0001) # 0.00001) # 今回は値が大きいので小さすぎると全然変わらなくなる

    # 訓練と評価用データにおける評価
    valid_losses = []
    for epoch in range(1, epochs + 1):
        
        loss_train = train(
            model=model, data_provider=data_provider('train', src_len, tgt_len, batch_size), optimizer=optimizer,
            criterion=criterion
        )
            
        loss_valid = evaluate(
            flag='val', model=model, data_provider=data_provider('val', src_len, tgt_len, batch_size), criterion=criterion
        )
        
        if epoch%10==0:
        # if epoch%1==0:
            print('[{}/{}] train loss: {:.2f}, valid loss: {:.2f}'.format(
                epoch, epochs,
                loss_train, loss_valid,
            ))
            
        valid_losses.append(loss_valid)
        
        if best_loss > loss_valid:
            best_loss = loss_valid
            best_model = model

    # [10/30] train loss: 0.10, valid loss: 0.31
    # [20/30] train loss: 0.07, valid loss: 0.40
    # [30/30] train loss: 0.05, valid loss: 0.30
    

    print("data provider : ", data_provider)

    # テスト用データにおける予測
    r = evaluate(flag='test', model=best_model, data_provider=data_provider('test', src_len, tgt_len, batch_size), criterion=criterion)
    print(r)
    # 0.5851381
