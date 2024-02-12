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

        # print("Q:", Q.shape)  # デバッグ用
        # print("K:", K.shape)  # デバッグ用
        # print("mask:", mask.shape)  # デバッグ用
        # print("attn_scores:", attn_scores.shape)  # デバッグ用

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

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, d_k, d_v)
        self.enc_dec_attn = MultiHeadAttention(d_model, n_head, d_k, d_v)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # Self-attention sublayer with target mask
        self_attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout(self_attn_output)
        x = self.layer_norm1(x)

        # Encoder-decoder attention sublayer with source mask
        enc_dec_attn_output, _ = self.enc_dec_attn(x, enc_output, enc_output, src_mask)
        x = x + self.dropout(enc_dec_attn_output)
        x = self.layer_norm2(x)

        # Feed-forward sublayer
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.layer_norm3(x)

        return x

class Decoder(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v, d_ff, num_layers, dropout=0.1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(
                        [DecoderLayer(d_model, n_head, d_k, d_v, d_ff, dropout) for _ in range(num_layers)]
                        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        x = self.layer_norm(x)
        return x

# パラメータ設定
d_model = 512  # 埋め込みの次元数
n_head = 8     # アテンションヘッドの数
d_k = d_v = 64 # キーと値の次元数
d_ff = 2048    # フィードフォワードネットワークの内部次元数
num_layers = 6 # デコーダの層の数
batch_size = 32  # バッチサイズ
tgt_seq_len = 100  # デコーダへの入力のシーケンスの長さ
src_seq_len = 120  # エンコーダからの出力のシーケンスの長さ
dropout = 0.1  # ドロップアウト率

# デコーダのインスタンス化
decoder = Decoder(d_model, n_head, d_k, d_v, d_ff, num_layers, dropout)

# モデルを評価モードに設定（ドロップアウトなどが無効になる）
decoder.eval()

# ダミーの入力
input_tensor = torch.rand(batch_size, tgt_seq_len, d_model)  # デコーダへの入力
enc_output = torch.rand(batch_size, src_seq_len, d_model)    # エンコーダからの出力

# マスクの設定
src_mask = torch.zeros(batch_size, tgt_seq_len, src_seq_len)
tgt_mask = torch.triu(torch.ones(batch_size, tgt_seq_len, tgt_seq_len), diagonal=1)  # 一部を0に

# デコーダにデータを通す
with torch.no_grad():  # 勾配計算を行わない
    decoded_output = decoder(input_tensor, enc_output, src_mask, tgt_mask)

print("Decoded output shape:", decoded_output.shape)  # 出力テンソルの形状を表示: [batch_size, tgt_seq_len, d_model]
print("Decoded output:", decoded_output)