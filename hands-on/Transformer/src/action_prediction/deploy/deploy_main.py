
import matplotlib.pyplot as plt
import numpy as np
# plt.plot(np.random.randn(df_sort.shape[0]), df_sort*0.01)
# plt.show()

import torch
import random


# 入力
import pandas as pd

df = pd.read_csv("date_and_actions2_mmddhhmm.csv",sep=",")
df.columns = ["date", "actions"] # ["date", "actions"]
from datetime import datetime as dt
df.date = df.date.apply(lambda d: dt.strptime(str(d), '%m%d%H%M')) # , errors='coerce')) # "%Y/%m/%d"))
# df.date = pd.to_datetime(df['date'], format='%m/%d-%H-%M') # '%Y年%m月%d日 %H時%M分')
# pd.to_datetime(df.date).dt.strftime('%m/%d-%H-%M') # '%Y-%m-%d')
# df['date'] = pd.to_datetime(df['date'], format='%m/%d-%H-%M') # ,errors='coerce')

# df.date = df.date.apply(lambda d: dt.strptime(str(d), "%Y%m%d%H%M%S"))

print("df : ")
print(df)
# csv = dict(df) # [date, action]
# print(csv)

# データ追加
# print("df.shape[0] : ", df.shape[0])
# df.loc[df.shape[0]] = ['buy dinner', 1930]
# df.loc[df.shape[0]] = ['coffee', 1230] # ['A5', 'B5', 'C5']

# "---------- add test ----------"
# # df.loc[df.shape[0]] = ['coffee', 1230] # ['A5', 'B5', 'C5']
# df.loc[df.shape[0]] = ['coffee', 1230] # ['A5', 'B5', 'C5']

sunday_data = [
    [2250300, "sleep"], 
    [2250600, "wakeup"], 
    [2250700, "breakfast"], 
    [2250715, "tooth brush"], 
    [2250755, "coffee"], 
    [2251200, "lunch"], 
    [2251230, "tooth brush"], 
    [2251400, "going out"], 
    [2252100, "back home"], 
    [2252130, "bath"], 
    [2252200, "TV/Youtube"], 
    [2252230, "study(AWS)"], 
    [2252300, "study(other)"], 
    [2252320, "go bed"], 
    [2252330, "TV/Youtube"], 
    [2252400, "sleep"], # 日曜日は外出したりする
]
Sunday = pd.DataFrame(
    sunday_data,
    #  index=[df.shape[0]]
    columns=['date', 'actions']
)
Sunday['date'] = pd.to_datetime(Sunday['date'], format='%m%d%H%M',errors='coerce')

monday_data = [
    [2260400, "sleep"], 
    [2260600, "wakeup"], 
    [2260700, "breakfast"], 
    [2260715, "tooth brush"], 
    [2260755, "coffee"], 
    [2260900, "go work"],
    [2261200, "lunch"], 
    [2261230, "tooth brush"], 
    [2261500, "meeting"], 
    [2261800, "study(English)"],
    [2261900, "gym"],
    [2262100, "back home"], 
    [2262130, "bath"], 
    [2262200, "study(ML)"], 
    [2262320, "go bed"], 
    [2262330, "TV/Youtube"], 
    [2262400, "sleep"],
]
Monday = pd.DataFrame(
    monday_data,
    #  index=[df.shape[0]]
    columns=['date', 'actions']
)
Monday['date'] = pd.to_datetime(Monday['date'], format='%m%d%H%M',errors='coerce')

wednesday_data = [
    [2270200, "sleep"], 
    [2270600, "wakeup"], 
    [2270700, "go work"],
    [2270730, "study(English)"],
    [2270755, "coffee"], 
    [2270815, "tooth brush"], 
    [2271200, "lunch"], 
    [2271230, "tooth brush"], 
    [2271500, "meeting"], 
    [2271800, "study(English)"],
    [2271900, "gym"],
    [2272100, "back home"], 
    [2272130, "bath"], 
    [2272200, "go bed"], 
    [2272230, "Movie"], 
    [2272330, "sleep"],
]
Wednesday = pd.DataFrame(
    wednesday_data,
    #  index=[df.shape[0]]
    columns=['date', 'actions']
)
Wednesday['date'] = pd.to_datetime(Wednesday['date'], format='%m%d%H%M',errors='coerce')

# Monday = {"sleep", "wakeup", "drink water", "go work", "coffee", "work", "study(C++)", "lunch", "coffee", "tooth brush", "back home", "gym", "buy dinner at 711", "bath", "TV/Youtube", "study(C++)", "study(AWS)", "study(ML)", "study(other)", "go bed", "TV/Youtube", "sleeping"}
# Wednesday = {"sleep", "wakeup", "go work", "coffee", "lunch", "coffee", "tooth brush", "study(ML)", "back home", "gym", "buy dinner at ox", "bath", "study(C++)", "go bed", "TV/Youtube", "sleeping"}
# Friday = {"sleep", "wakeup", "go work", "coffee", "lunch", "coffee", "tooth brush", "drink alchol", "back home", "bath", "TV/Youtube", "go bed", "TV/Youtube", "sleeping"} # 金曜日は飲み会に行ったりする
print('Sunday : ', Sunday)
df = pd.concat([df, Sunday], axis=0) # 1) # 日曜日のデータを行方向に追加
print('Monday : ', Monday)
df = pd.concat([df, Monday], axis=0) # 1) # 月曜日のデータを行方向に追加
print('Wednesday : ', Wednesday)
df = pd.concat([df, Wednesday], axis=0)

# total_data_pred = lambda x: int(df['date'].replace("1900-2-24", ""))
# # total_data_pred.reset_index(drop=True)
# # # total_data_pred.drop_duplicates(keep='last', subset='actions')
# print("total data pred : ")
# print(total_data_pred)
# plt.plot(total_data_pred)

total_data = df
total_data = total_data.reset_index(drop=True)
# total_data = pd.concat([total_data, 'idx'], axis=1)
# total_data.set_index('idx',drop=False)
# total_data = total_data.drop_duplicates(keep='last', subset='actions')
total_data.to_csv('outfile_TotalData.csv')
"---------- add test ----------"


print("df : ")
print(df)
df = df.drop_duplicates(keep='last', subset='actions') # 重複したデータを削除
print("df drop : ")
print(df)


df_sort = df.sort_values(['date']) # date順にソート
print("sorted df : ")
print(df_sort)

"*****"
# # indexをソート順でつける場合
# # df_sort["idx"] = 0
df_sort = df_sort.reset_index(drop=True) # reset_index(inplace=True, drop=True) 
# # df_sort = df.drop_duplicates(subset='actions', keep='last', ignore_index=True)
# # df_sort.columns = ["idx", "actions", "date"]
" or "
# indexをつけない場合
# df_sort = df_sort.set_index('actions') # indexをactionsに変更
"*****"


# 予測対象のリスト(縦軸)の出力
df_sort.to_csv('outfile_addData.csv')





pred = [random.randint(0, 2272330) for p in range(df_sort.shape[0])] # 0, 10)]
# pred = [ 724, 1752, 1692,  172,  907,  357,  674, 2044,  164, 1048, 1706,   51,
#           569, 1231,  586,  891, 2329]
# pred = [600,
#         700,
#         755,
#         900,
#         1200,
#         1230,
#         1400,
#         1900,
#         1930,
#         2100,
#         2130,
#         2200,
#         2230,
#         2300,
#         2320,
#         2330,
#         2400,
#         ] # データが時刻順かテスト
        
# pred = [random.randint(0, 24) for p in range(df_sort.shape[0])]
# pred = [ 3, 11,  5,  3, 23, 24,  3,  3, 16, 20, 18,  0,  8,  1, 13, 18,  3]


output = torch.tensor([pred])
# print(output)
print(df_sort.shape[0])
x = np.linspace(0, df_sort.shape[0], df_sort.shape[0])
print(x)
print(output)
# plt.plot(x, output.numpy()[0])
plt.plot(df_sort['date'], df_sort['actions'])

# # df_sort.columns = ["actions", "date"]
# # next_action_list = [ "sleep", "wakeup", "breakfast", "tooth brush", "go work", "coffee", "lunch", "coffee", "tooth brush", "drink alchol", "back home", "bath", "TV/Youtube", "study(C++)", "study(AWS)", "study(ML)", "study(other)", "go bed", "TV/Youtube", "sleeping"]
# # plt.yticks(x, df_sort['actions']) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], next_action_list)
# plt.yticks(df_sort['date'], df_sort['actions'])

print(df_sort['date']) # print(df_sort.set_index('date'))

plt.show()






pre = total_data # ソート前のdfを保存
print("pre : ", pre)
total_data = total_data.sort_values(['date']) # date順にソート
print("sorted total data : ")
print(total_data)
# print(pre.sort_values(['actions']))
# plt.plot(total_data['date'], total_data['actions'])
total_data.to_csv('test.csv')

# depluoy_test_adddata.pyのoutfile_addData.csvより
# y軸（予測対象）
y = [
    
    "wakeup",
    "go work",
    "breakfast",
    "coffee",
    "working",
    "lunch",
    "tooth brush",
    "going out",
    "meeting",
    "study(English)",
    "gym",
    "buy dinner",
    "back home",
    "bath",
    "study(ML)",
    "go bed",
    "study",
    "Movie",
    "study(AWS)",
    "study(other)",
    "TV/Youtube",
    "sleep",
]

# y = 2*total_data.index()

plt.plot(total_data['date'], pre['actions'])
# plt.plot(total_data['actions'])
# plt.yticks(pre['actions'], df_sort['actions'])

# plt.yticks(total_data['actions'], y)

plt.show()