import torch
import torch.nn as nn

# ハイパーパラメータの設定
d_model = 512  # モデルの次元数 # 各単語を表すベクトルの次元数
eps = 1e-6  # イプシロン（数値安定性のため） # 徐算の際に分母に0のような小さい値が入ると結果がおかしくなってしまうため

# LayerNormのインスタンス化
layer_norm = nn.LayerNorm(normalized_shape=d_model, eps=eps)

# ランダムなテンソルを生成
batch_size = 8
seq_len = 50
x = torch.randn(batch_size, seq_len, d_model)

# LayerNormを適用
normalized_x = layer_norm(x)

# 結果の表示
print(normalized_x.size())  # 出力テンソルのサイズを表示: [batch_size, seq_len, d_model]