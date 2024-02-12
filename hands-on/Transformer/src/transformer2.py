import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.norm = norm

    def forward(self, src, mask=None):
        for layer in self.layers:
            src = layer(src, mask)
        if self.norm is not None:
            src = self.norm(src)
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        if self.norm is not None:
            tgt = self.norm(tgt)
        return tgt

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_emb = self.src_embed(src)
        tgt_emb = self.tgt_embed(tgt)
        memory = self.encoder(src_emb, src_mask)
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_mask=src_mask)
        return output
    # def forward(self, src):
    #     src = src.permute(1, 0, 2)  # バッチの次元を最初に移動
    #     output = self.encoder(src)
    #     output = output[-1]  # 最後の時刻の出力を取得
    #     output = self.decoder(output)
    #     return output

# Usage Example:
# Define model parameters
# d_model = 512  # Embedding dimension
d_model = 128  # モデルの次元

# nhead = 8  # Number of attention heads
nhead = 2  # 注意機構のヘッド数

num_encoder_layers = 6  # Number of encoder layers
num_decoder_layers = 6  # Number of decoder layers
# dim_feedforward = 2048  # Feedforward dimension
dim_feedforward = 256  # フィードフォワードの次元

dropout = 0.1  # Dropout rate

src_vocab_size = 10000
tgt_vocab_size = 10000

# Define embedding layers
src_embed = nn.Embedding(src_vocab_size, d_model)
tgt_embed = nn.Embedding(tgt_vocab_size, d_model)

# Define transformer model
encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
transformer = Transformer(encoder, decoder, src_embed, tgt_embed, None)
model = transformer

# Forward pass
src = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
# Example input sequence
tgt = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
# Example target sequence
# output = transformer(src, tgt)
# print(output)

# Example data
# 仮想的なデータです。実際のデータを使用する場合は、データの読み込みと前処理が必要です。
time_data = torch.tensor([
                        [0.1, 0.2, 0.3, 0.4, 0.5], 
                        [0.2, 0.3, 0.4, 0.5, 0.6]
                        ])  # 時間情報
location_data = torch.tensor([
                            [1, 2, 3, 4, 5], 
                            [5, 4, 3, 2, 1]
                            ])  # 位置情報
action_data = torch.tensor([
                            [0, 1, 0, 1, 0], 
                            [1, 0, 1, 0, 1]
                            ])  # 行動

import torch.optim as optim
# 損失関数とオプティマイザ
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(transformer.parameters(), lr=0.001)

# 学習
num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()
    # output = model(torch.cat((time_data.unsqueeze(2), location_data.unsqueeze(2), action_data.unsqueeze(2)), dim=2))
    output = transformer(time_data, action_data) # .long() # src, tgt)
    loss = criterion(output, torch.argmax(action_data, dim=1))
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# # 予測
# with torch.no_grad():
#     next_time = torch.tensor([[0.6], [0.7]])  # 予測する次の時間情報
#     next_location = torch.tensor([[6], [0]])  # 予測する次の位置情報
#     next_input = torch.cat((next_time, next_location), dim=1).unsqueeze(2)
#     predicted_action = torch.argmax(model(next_input), dim=1)
#     print("Predicted actions:", predicted_action)
