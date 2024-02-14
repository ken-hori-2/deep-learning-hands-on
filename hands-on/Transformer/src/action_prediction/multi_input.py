import torch

# 3つの時系列データを作成
# time_series1 = torch.randn(10)  # 時系列データ1
# time_series2 = torch.randn(10)  # 時系列データ2
# time_series3 = torch.randn(10)  # 時系列データ3

import pandas as pd
df1 = pd.read_csv("kabuka_small_add_day.csv",sep=",")
df2 = pd.read_csv("gouseizyusi.csv",sep=",")
df3 = pd.read_csv("test_small.csv",sep=",") # pd.read_csv("kabuka_small.csv",sep=",")
df1.columns = ["date", "actions"]
df2.columns = ["date", "actions"]
df3.columns = ["date", "actions"]
from datetime import datetime as dt
df1.date = df1.date.apply(lambda d: dt.strptime(str(d), "%Y/%m/%d"))
df2.date = df2.date.apply(lambda d: dt.strptime(str(d), "%Y/%m/%d"))
df3.date = df3.date.apply(lambda d: dt.strptime(str(d), "%Y/%m/%d"))

# 時系列データを結合して入力とする
# input_data = torch.stack((time_series1, time_series2, time_series3), dim=1)
# input_data = torch.stack((df1, df2, df3), dim=1)
print("*****")
print(df1)
print("*****")
print(df2)
print("*****")
print(df3)
print("*****")
input_data = torch.stack((torch.tensor(df1['actions'].values), torch.tensor(df2['actions'].values), torch.tensor(df3['actions'].values)), dim=1)


# 入力データの形状を表示
print("Input shape:", input_data.shape)

# print("*****")
# print(time_series1)
# print("*****")
# print(time_series2)
# print("*****")
# print(time_series3)
# print("*****")
# print("*****")
# # print(df1)
# print("*****")
# print(df2)
# print("*****")
# print(df3)
# print("*****")
print(input_data)

import matplotlib.pyplot as plt
# plt.plot(time_series1, label="1")
# plt.plot(time_series2, label="2")
# plt.plot(time_series3, label="3")
# plt.legend()
# plt.show()

# plt.plot(input_data, label="inputdata")
plt.plot(input_data, label=("kabuka","gouseizyusi","robotics"))
plt.legend()
# plt.show()
plt.savefig("multi_input.png")
