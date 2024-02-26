
import matplotlib.pyplot as plt
import numpy as np
# plt.plot(np.random.randn(df_sort.shape[0]), df_sort*0.01)
# plt.show()

import torch
import random


# 入力
import pandas as pd

# df = pd.read_csv("date_and_actions2_mmddhhmm.csv",sep=",")







# date_and_actions2_mmddhhmm.csv を next_action_listのようにラベルに変換したver.
df = pd.read_csv("input.csv",sep=",")
df.columns = ["date", "actions"]
from datetime import datetime as dt
df.date = df.date.apply(lambda d: dt.strptime(str(d), '%m%d%H%M%S'))

print("df : ")
print(df)

total_data = df
total_data = total_data.reset_index(drop=True)
# total_data = pd.concat([total_data, 'idx'], axis=1)
# total_data.set_index('idx',drop=False)
# total_data = total_data.drop_duplicates(keep='last', subset='actions')
total_data.to_csv('outfile_TotalData.csv')
next_action_list = {
    "wakeup":0,
    "go work":1,
    "breakfast":2,
    "coffee":3,
    "working":4,
    "lunch":5,
    "tooth brush":6,
    "going out":7,
    "meeting":8,
    "study(English)":9,
    "gym":10,
    "buy dinner":11,
    "back home":12,
    "bath":13,
    "study(ML)":14,
    "go bed":15,
    "study":16,
    "Movie":17,
    "study(AWS)":18,
    "study(other)":19,
    "TV/Youtube":20,
    "sleep":21,
}
total_data = total_data.replace(
        'wakeup', 0
    ).replace(
        'go work', 1
    ).replace(
        'breakfast', 2
    ).replace(
        'coffee', 3
    ).replace(
        'working', 4
    ).replace(
        'lunch', 5
    ).replace(
        'tooth brush', 6
    ).replace(
        'going out', 7
    ).replace(
        'meeting', 8
    ).replace(
        'study(English)', 9
    ).replace(
        'gym', 10
    ).replace(
        'buy dinner', 11
    ).replace(
        'back home', 12
    ).replace(
        'bath', 13
    ).replace(
        'study(ML)', 14
    ).replace(
        'go bed', 15
    ).replace(
        'study', 16
    ).replace(
        'Movie', 17
    ).replace(
        'study(AWS)', 18
    ).replace(
        'study(other)', 19
    ).replace(
        'TV/Youtube', 20
    ).replace(
        'sleep', 21
    )

print("replace : ")
print(total_data)
total_data.to_csv('replace.csv')
replace = total_data
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
plt.plot(df_sort['date'], df_sort['actions'])
print(df_sort['date']) # print(df_sort.set_index('date'))
plt.show()






# pre = total_data # ソート前のdfを保存
# print("pre : ", pre)
# total_data = total_data.sort_values(['date']) # date順にソート
# print("sorted total data : ")
# print(total_data)
# # print(pre.sort_values(['actions']))
# # plt.plot(total_data['date'], total_data['actions'])
# total_data.to_csv('test.csv')

# depluoy_test_adddata.pyのoutfile_addData.csvより
# y軸（予測対象）
# next_action_list = [
#     "wakeup",
#     "go work",
#     "breakfast",
#     "coffee",
#     "working",
#     "lunch",
#     "tooth brush",
#     "going out",
#     "meeting",
#     "study(English)",
#     "gym",
#     "buy dinner",
#     "back home",
#     "bath",
#     "study(ML)",
#     "go bed",
#     "study",
#     "Movie",
#     "study(AWS)",
#     "study(other)",
#     "TV/Youtube",
#     "sleep",
# ]
# next_action_list = {
#     "wakeup":0,
#     "go work":1,
#     "breakfast":2,
#     "coffee":3,
#     "working":4,
#     "lunch":5,
#     "tooth brush":6,
#     "going out":7,
#     "meeting":8,
#     "study(English)":9,
#     "gym":10,
#     "buy dinner":11,
#     "back home":12,
#     "bath":13,
#     "study(ML)":14,
#     "go bed":15,
#     "study":16,
#     "Movie":17,
#     "study(AWS)":18,
#     "study(other)":19,
#     "TV/Youtube":20,
#     "sleep":21,
# }


# replace.set_index() # 'date')
print(replace)
replace = replace.sort_values(['date'])
print(replace)
# plt.plot(total_data['date'], pre['actions'])
plt.plot(replace['date'], replace['actions'], label='actions', color='orange')
plt.legend()
plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], next_action_list)

plt.show()