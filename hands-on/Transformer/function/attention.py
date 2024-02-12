import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k): # d_k : キーとクエリ
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
            attn_scores = attn_scores.masked_fill(mask == 1, float('-inf')) # ほぼ0になる
            print(attn_scores)

        # Compute attention weights
        attn_weights = self.softmax(attn_scores) # 合計を1にする

        # Compute weighted sum of values
        output = torch.matmul(attn_weights, V)

        return output, attn_weights


# ハイパーパラメータ
batch_size = 8
n_head = 4
seq_len = 10
d_k = 64
d_v = 128

# ScaledDotProductAttentionモジュールのインスタンス化
scaled_dot_product_attention = ScaledDotProductAttention(d_k)

# ランダムなテンソルを生成
Q = torch.randn(batch_size, n_head, seq_len, d_k)
K = torch.randn(batch_size, n_head, seq_len, d_k)
V = torch.randn(batch_size, n_head, seq_len, d_v)

# マスクの作成
# このマスクは、最初の5つの位置だけをアンマスクし、残りの位置をマスクします。
mask = torch.ones(batch_size, 1, 1, seq_len)
mask[:, :, :, :5] = 0

# forwardメソッドを呼び出し
output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

# 出力とAttention weightを表示
print(output.size())  # 出力テンソルのサイズを表示: [batch_size, n_head, seq_len, d_v]
print(attn_weights.size())  # Attention weightテンソルのサイズを表示: [batch_size, n_head, seq_len, seq_len]