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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor, shape [batch_size, seq_len, d_model]

        Returns:
            Tensor: Output tensor, shape [batch_size, seq_len, d_model]
        """
        # Add positional encoding to the input tensor
        x = x + self.encoding[:, :x.size(1), :]
        return x

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
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.layer_norm1(x)

        # Feed-forward sublayer
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.layer_norm2(x)

        return x

class Encoder(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v, d_ff, num_layers, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_head, d_k, d_v, d_ff, dropout) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        x = self.layer_norm(x)
        return x


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
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_head, d_k, d_v, d_ff, dropout) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        x = self.layer_norm(x)
        return x


class Transformer(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v, d_ff, num_encoder_layers, num_decoder_layers, src_vocab_size, tgt_vocab_size, max_src_seq_len, max_tgt_seq_len, dropout=0.1):
        super(Transformer, self).__init__()

        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_src_seq_len)
        self.encoder = Encoder(d_model, n_head, d_k, d_v, d_ff, num_encoder_layers, dropout)

        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_decoder = PositionalEncoding(d_model, max_tgt_seq_len)
        self.decoder = Decoder(d_model, n_head, d_k, d_v, d_ff, num_decoder_layers, dropout)

        self.output_linear = nn.Linear(d_model, tgt_vocab_size)
        self.softmax = nn.Softmax(dim=-1)

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model

    def forward(self, src, tgt, src_mask, tgt_mask):
        src_emb = self.encoder_embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
        src_emb = self.pos_encoder(src_emb)
        enc_output = self.encoder(src_emb, src_mask)

        tgt_emb = self.decoder_embedding(tgt) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
        tgt_emb = self.pos_decoder(tgt_emb)
        dec_output = self.decoder(tgt_emb, enc_output, src_mask, tgt_mask)

        output = self.output_linear(dec_output)
        return self.softmax(output)

if __name__ == "__main__":

    # Transformer モデルのパラメータ
    d_model = 512
    n_head = 8
    d_k = d_v = 64
    d_ff = 2048
    num_encoder_layers = 6
    num_decoder_layers = 6
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    max_src_seq_len = 100
    max_tgt_seq_len = 100
    dropout = 0.1

    # Transformer モデルのインスタンス化
    transformer_model = Transformer(
        d_model=d_model,
        n_head=n_head,
        d_k=d_k,
        d_v=d_v,
        d_ff=d_ff,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        max_src_seq_len=max_src_seq_len,
        max_tgt_seq_len=max_tgt_seq_len,
        dropout=dropout
    )

    # モデルを評価モードに設定（ドロップアウトなどが無効になる）
    transformer_model.eval()

    # ランダムなデータの生成
    batch_size = 32
    src_seq_len = max_src_seq_len
    tgt_seq_len = max_tgt_seq_len

    # ソースとターゲットのシーケンスをランダムに生成
    src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))

    # ソースとターゲットのマスクを生成（ここでは実際にはマスクしていない）
    src_mask = torch.zeros(batch_size, src_seq_len, src_seq_len)
    tgt_mask = torch.zeros(batch_size, tgt_seq_len, tgt_seq_len)

    # モデルを通じてデータを伝播させる
    with torch.no_grad():  # 勾配計算を行わない
        output = transformer_model(src, tgt, src_mask, tgt_mask)

    # 出力の確認
    print(output.size())  # 期待される出力サイズ: [batch_size, tgt_seq_len, tgt_vocab_size]
    print(output)