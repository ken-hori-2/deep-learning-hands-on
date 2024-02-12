import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.scaling_factor = torch.rsqrt(torch.tensor(d_k, dtype=torch.float))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q (Tensor): Queries tensor, shape [batch_size, n_head, seq_len, d_k].
            K (Tensor): Keys tensor, shape [batch_size, n_head, seq_len, d_k].
            V (Tensor): Values tensor, shape [batch_size, n_head, seq_len, d_v].
            mask (Tensor, optional): Mask tensor, shape [batch_size, 1, 1, seq_len].

        Returns:
            Tensor: Output tensor, shape [batch_size, n_head, seq_len, d_v].
            Tensor: Attention weights tensor, shape [batch_size, n_head, seq_len, seq_len].
        """

        # Compute scaled dot-product attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scaling_factor

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 1, float('-inf'))

        # Compute attention weights
        attn_weights = self.softmax(attn_scores)

        # Compute weighted sum of values
        output = torch.matmul(attn_weights, V)

        return output, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.W_Q = nn.Linear(d_model, d_k * n_head, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_head, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_head, bias=False)
        self.W_O = nn.Linear(d_v * n_head, d_model, bias=False)

        self.attention = ScaledDotProductAttention(d_k)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # Linear projections
        Q = self.W_Q(Q).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.n_head, self.d_v).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]

        # Apply Scaled Dot Product Attention
        x, attn = self.attention(Q, K, V, mask=mask)  # [batch_size, n_head, seq_len, d_v]

        # Concatenate and apply final linear
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.d_v)  # [batch_size, seq_len, n_head * d_v]
        output = self.W_O(x)  # [batch_size, seq_len, d_model]

        return output, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Args:
            d_model (int): The dimension of the model (also the input and output dimension).
            d_ff (int): The dimension of the feed-forward hidden layer.
            dropout (float): Dropout probability.
        """
        # super(PositionwiseFeedForward, self).__init__()
        # self.w_1 = nn.Linear(d_model, d_ff)
        # self.w_2 = nn.Linear(d_ff, d_model)
        # self.dropout = nn.Dropout(dropout)
        # self.relu = nn.ReLU()

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
        # return self.w_2(self.dropout(self.relu(self.w_1(x))))
        
        output = self.features(x)
        
        return output


# Encoderでは6回処理が繰り返される : そのうちの1回の処理 = 上記の各機能を1つにまとめたもの
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, d_k, d_v)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Self-attention sublayer
        attn_output, _ = self.self_attn(x, x, x, mask) # クエリとキーとバリューを渡しているが、selfアテンションなのですべてx(自分自身)
        x = x + self.dropout(attn_output) # スルーした入力(迂回処理)に上記のアテンションのドロップアウトしたものを加算
        x = self.layer_norm1(x)

        # Feed-forward sublayer
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.layer_norm2(x)

        return x


class Encoder(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v, d_ff, num_layers, dropout=0.1):
        super(Encoder, self).__init__()

        # 内包表記を使ってEncoderレイヤーが6つ入ったリストを作成して、それをモジュールリストにまとめている
        self.layers = nn.ModuleList(
                                    [EncoderLayer(d_model, n_head, d_k, d_v, d_ff, dropout) for _ in range(num_layers)]
                                    )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        x = self.layer_norm(x)
        return x


# パラメータ設定
d_model = 512  # 埋め込みの次元数
n_head = 8     # アテンションヘッドの数
d_k = d_v = 64 # キーと値の次元数
d_ff = 2048    # フィードフォワードネットワークの内部次元数
num_layers = 6 # エンコーダの層の数
batch_size = 32  # バッチサイズ
seq_len = 100  # 入力シーケンスの長さ
dropout = 0.1  # ドロップアウト率

# エンコーダのインスタンス化
encoder = Encoder(d_model, n_head, d_k, d_v, d_ff, num_layers, dropout)

# モデルを評価モードに設定（ドロップアウトなどが無効になる）
encoder.eval()

# ダミーの入力データとマスクの生成
input_tensor = torch.rand(batch_size, seq_len, d_model)  # ダミーの入力テンソル
mask = torch.zeros(batch_size, seq_len, seq_len)  # マスク（ここでは全てマスクなし）

# エンコーダに入力データを通す
with torch.no_grad():  # 勾配計算を行わない
    encoded_output = encoder(input_tensor, mask)

print("Encoded output shape:", encoded_output.shape)  # 出力テンソルの形状を表示: [batch_size, seq_len, d_model]
print("Encoded output:", encoded_output)