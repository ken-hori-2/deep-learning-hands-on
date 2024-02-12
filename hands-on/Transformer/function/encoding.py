import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term) # 0から始まって2つおき＝偶数　はsinの式
        self.encoding[:, 1::2] = torch.cos(position * div_term) # 1から始まって1つおき＝奇数　はcosの式
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor, shape [batch_size, seq_len, d_model] # [バッチサイズ, 文章の長さ, 入力として渡されるベクトルの要素数]

        Returns:
            Tensor: Output tensor, shape [batch_size, seq_len, d_model]
        """
        # Add positional encoding to the input tensor
        x = x + self.encoding[:, :x.size(1), :] # 入力に対して位置情報を加える
        return x


# ハイパーパラメータの設定
batch_size = 8
seq_len = 50
d_model = 512  # モデルの次元数 # 各ベクトルの長さ＝各単語を表すベクトルの長さ

# PositionalEncodingモジュールのインスタンス化
positional_encoding = PositionalEncoding(d_model)

# ランダムなテンソルを生成
x = torch.randn(batch_size, seq_len, d_model)

# forwardメソッドを呼び出し
encoded_x = positional_encoding(x)

# エンコードされたデータの表示
print(encoded_x.size())  # エンコードされたテンソルのサイズを表示: [batch_size, seq_len, d_model]

import torch
import matplotlib.pyplot as plt

# PositionalEncodingクラスのインスタンス化
d_model = 128  # モデルの次元数
positional_encoding = PositionalEncoding(d_model)

# ダミーの入力テンソルを作成
x = torch.zeros(1, 100, d_model)  # ここでは、seq_lenを100としています

# PositionalEncodingを適用
encoded_x = positional_encoding(x)

# エンコードされたポジショナルエンコーディングをNumPy配列に変換
encoded_x = encoded_x.squeeze(0).detach().numpy()

# グラフをプロット
plt.figure(figsize=(10, 10))
plt.pcolormesh(encoded_x.transpose(), cmap='viridis')
plt.xlabel('Sequence Position')
plt.ylabel('Embedding Dimension')
plt.colorbar()
plt.show()
