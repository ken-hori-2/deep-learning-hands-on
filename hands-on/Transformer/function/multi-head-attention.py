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

        self.W_Q = nn.Linear(d_model, d_k * n_head, bias=False) # クエリ
        self.W_K = nn.Linear(d_model, d_k * n_head, bias=False) # キー

        self.W_V = nn.Linear(d_model, d_v * n_head, bias=False) # バリュー
        
        self.W_O = nn.Linear(d_v * n_head, d_model, bias=False) # アウトプット

        self.attention = ScaledDotProductAttention(d_k)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # Linear projections
        Q = self.W_Q(Q).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.n_head, self.d_v).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch_size, 1, 1, seq_len]

        # Apply Scaled Dot Product Attention
        x, attn = self.attention(Q, K, V, mask=mask)  # [batch_size, n_head, seq_len, d_v]

        # Concatenate and apply final linear
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.d_v)  # [batch_size, seq_len, n_head * d_v]
        output = self.W_O(x)  # [batch_size, seq_len, d_model]

        return output, attn




# ハイパーパラメータ
batch_size = 8
seq_len = 10
d_model = 512  # 入力特徴の次元数
n_head = 8  # Attention headの数
d_k = 64  # キー/クエリベクトルの次元数
d_v = 64  # 値ベクトルの次元数

# MultiHeadAttentionモジュールのインスタンス化
multi_head_attention = MultiHeadAttention(d_model, n_head, d_k, d_v)

# ランダムなテンソルを生成
Q = torch.randn(batch_size, seq_len, d_model)
K = torch.randn(batch_size, seq_len, d_model)
V = torch.randn(batch_size, seq_len, d_model)

# オプショナル: マスクの作成
# このマスクは、最初の5つの位置だけをアンマスクし、残りの位置をマスクします。
mask = torch.ones(batch_size, 1, seq_len)
mask[:, :, :5] = 0

# forwardメソッドを呼び出し
output, attn_weights = multi_head_attention(Q, K, V, mask)

# 出力とAttention weightを表示
print(output.size())  # 出力テンソルのサイズを表示: [batch_size, seq_len, d_model]
print(attn_weights.size())  # Attention weightテンソルのサイズを表示: [batch_size, n_head, seq_len, seq_len]