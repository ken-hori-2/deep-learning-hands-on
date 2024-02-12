import torch
import torch.nn as nn

# class PositionwiseFeedForward(nn.Module):
#     def __init__(self, d_model, d_ff, dropout=0.1):
#         """
#         Args:
#             d_model (int): The dimension of the model (also the input and output dimension).
#             d_ff (int): The dimension of the feed-forward hidden layer.
#             dropout (float): Dropout probability.
#         """
#         super(PositionwiseFeedForward, self).__init__()
#         # 入力にseq_len(文章の長さ)は含まれない＝各単語ごとに処理される
#         self.w_1 = nn.Linear(d_model, d_ff) # 入力層 -> 中間層
#         self.w_2 = nn.Linear(d_ff, d_model) # 中間層 -> 出力層
#         self.dropout = nn.Dropout(dropout) # ニューロンをランダムに無効化する
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         """
#         Args:
#             x (Tensor): Input tensor, shape [batch_size, seq_len, d_model]

#         Returns:
#             Tensor: Output tensor, shape [batch_size, seq_len, d_model]
#         """
#         return self.w_2(self.dropout(self.relu(self.w_1(x)))) # seq_len(文章の長さ)は含まれない＝各単語ごとに処理される

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Args:
            d_model (int): The dimension of the model (also the input and output dimension).
            d_ff (int): The dimension of the feed-forward hidden layer.
            dropout (float): Dropout probability.
        """
        super(PositionwiseFeedForward, self).__init__()
        # 入力にseq_len(文章の長さ)は含まれない＝各単語ごとに処理される

        self.features = nn.Sequential( 
            nn.Linear(d_model, d_ff), # 入力層 -> 中間層 # self.w_1 = 
            nn.ReLU(), # self.relu = 
            nn.Dropout(dropout), # ニューロンをランダムに無効化する # self.dropout = 
            nn.Linear(d_ff, d_model), # 中間層 -> 出力層 # self.w_2 = 
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor, shape [batch_size, seq_len, d_model]

        Returns:
            Tensor: Output tensor, shape [batch_size, seq_len, d_model]
        """
        output = self.features(x)
        
        return output


# ハイパーパラメータの設定
batch_size = 8
seq_len = 50
d_model = 512  # モデルの次元数
d_ff = 2048  # フィードフォワード隠れ層の次元数

# PositionwiseFeedForwardモジュールのインスタンス化
positionwise_ff = PositionwiseFeedForward(d_model, d_ff)

# ランダムなテンソルを生成
x = torch.randn(batch_size, seq_len, d_model)

# forwardメソッドを呼び出し
output = positionwise_ff(x)

# 出力の表示
print(output.size())  # 出力テンソルのサイズを表示: [batch_size, seq_len, d_model]