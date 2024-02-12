import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from Transformer import Transformer

# 簡単な翻訳データセットを作成する
class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src_sentence = self.src_sentences[idx]
        tgt_sentence = self.tgt_sentences[idx]
        src_indices = [self.src_vocab[token] for token in src_sentence.split()]
        tgt_indices = [self.tgt_vocab[token] for token in tgt_sentence.split()]
        return torch.tensor(src_indices), torch.tensor(tgt_indices)

# 簡単な語彙とデータセットを作成
src_vocab = {'hello': 0, 'world': 1, 'EOS': 2}
tgt_vocab = {'こんにちは': 0, '世界': 1, 'EOS': 2}
src_sentences = ['hello world EOS', 'world hello EOS']
tgt_sentences = ['こんにちは 世界 EOS', '世界 こんにちは EOS']

# データセットとデータローダーの作成
dataset = TranslationDataset(src_sentences, tgt_sentences, src_vocab, tgt_vocab)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# モデルのインスタンス化
transformer = Transformer(
    d_model=512, n_head=8, d_k=64, d_v=64, d_ff=2048,
    num_encoder_layers=3, num_decoder_layers=3,
    src_vocab_size=len(src_vocab), tgt_vocab_size=len(tgt_vocab),
    max_src_seq_len=10, max_tgt_seq_len=10, dropout=0.1
)

# 損失関数と最適化手法
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(transformer.parameters(), lr=0.001)

# 訓練ループ
history = {"train_loss": []}
for epoch in range(10):  # 実際にはもっと多くのエポックが必要です
    transformer.train()
    total_loss = 0
    for src, tgt in dataloader:
        # モデルの出力を計算
        output = transformer(src, tgt[:, :-1], None, None)

        # 損失を計算
        loss = criterion(output.view(-1, output.size(-1)), tgt[:, 1:].reshape(-1))

        # 勾配を初期化
        optimizer.zero_grad()

        # バックプロパゲーション
        loss.backward()

        # パラメータの更新
        optimizer.step()

        total_loss += loss.item()
        history["train_loss"].append(loss.item())

    print(f'Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}')

# result = transformer(src_vocab, tgt_vocab, )
# print(result)

import matplotlib.pyplot as plt

plt.plot(history["train_loss"])
plt.show()