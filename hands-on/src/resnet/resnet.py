import torch
import torch.nn as nn

# resnetのように既存のモデルを出力層のみを変えて自分のデータ用に学習させる方法

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 残差ブロック用
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) # in_channelsではなく、out_channelsであることに注意
        
        # shortcut用
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0) # あるチャネル層を持ったものを別のものに変える

        # 共通点
        self.bn = nn.BatchNorm2d(out_channels)
        
        # 活性化関数
        self.relu = nn.ReLU()
    
    def shortcut(self, x): # 出力チャネル数を変える
        x = self.conv3(x)
        x = self.bn(x)
        return x
    
    def forward(self, x): # 残差ブロック
        identity = x
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn(x)

        # x += identity # チャネル数が違うので単純に足せない
        x += self.shortcut(identity)
        
        # x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.linear = nn.Linear(in_features=28*28*64, out_features=10) # 28*28の画像で64チャネルと想定 10クラス分類の想定
        self.layer = self._make_layer(block, 3, 3, 64) # 今回は3層積み重ねる想定 最初のin_channel=カラー画像(3チャネル)想定 out_channelは64を想定
    
    def _make_layer(self, block, num_residual_block, in_channels, out_channels):
        layers = []
        for i in range(num_residual_block):

            if i == 0: # 最初は画像データ(inを入力してoutを出力)
                layers.append(block(in_channels, out_channels))

            else: # 2層目以降は前の層のoutを入力してoutを出力する
                layers.append(block(out_channels, out_channels))
        
        return nn.Sequential(*layers) # 複数のブロックを積み重ねたモデルを作成
    
    def forward(self, x):
        x = self.layer(x) # チャネル数が64の画像が出力
        x = x.view(x.size(0), -1) # そのままでは全結合できないので、
        # x = x.reshape(x.size(0), -1) # [batch_num, c*h*w]の1次元データに変換
        x = self.linear(x)
        return x


if __name__ == "__main__":

    model = ResNet(ResidualBlock)

    
    print(model)

    x_test = torch.randn(32, 3, 28, 28)

    # x_test = x_test.permute(1, 2, 3, 0) # channel lastに軸の順番を変更
    # x_test = x_test[:, :, 0]
    out_put = model(x_test)
    # print(out_put)
    print(out_put.size())