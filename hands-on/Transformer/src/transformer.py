import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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

# モデルパラメータ
d_model = 128  # モデルの次元
nhead = 2  # 注意機構のヘッド数
num_layers = 2  # レイヤー数
dim_feedforward = 256  # フィードフォワードの次元
dropout = 0.1  # ドロップアウト率

# トランスフォーマーモデル定義
class TransformerModel(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout):
        super(TransformerModel, self).__init__()
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout), num_layers)
        self.decoder = nn.Linear(d_model, 2)  # 2つのクラスに分類するための線形層

    def forward(self, src):
        src = src.permute(1, 0, 2)  # バッチの次元を最初に移動
        output = self.encoder(src)
        output = output[-1]  # 最後の時刻の出力を取得
        output = self.decoder(output)
        return output

# モデルのインスタンス化
model = TransformerModel(d_model, nhead, num_layers, dim_feedforward, dropout)

# 損失関数とオプティマイザ
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学習
num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(torch.cat((time_data.unsqueeze(2), location_data.unsqueeze(2), action_data.unsqueeze(2)), dim=2))
    loss = criterion(output, torch.argmax(action_data, dim=1))
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# 予測
with torch.no_grad():
    next_time = torch.tensor([[0.6], [0.7]])  # 予測する次の時間情報
    next_location = torch.tensor([[6], [0]])  # 予測する次の位置情報
    next_input = torch.cat((next_time, next_location), dim=1).unsqueeze(2)
    predicted_action = torch.argmax(model(next_input), dim=1)
    print("Predicted actions:", predicted_action)
