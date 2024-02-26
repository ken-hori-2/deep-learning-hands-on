

next_action_list = [ "sleep", "wakeup", "breakfast", "tooth brush", "go work", "coffee", "lunch", "coffee", "tooth brush", "drink alchol", "back home", "bath", "TV/Youtube", "study(C++)", "study(AWS)", "study(ML)", "study(other)", "go bed", "TV/Youtube", "sleeping"]

flag = 'train'
type_map = {'train': 0, 'val': 1, 'test':2}
set_type = type_map[flag]
print(set_type)

next_action_list = {"sleep", "wakeup", "breakfast", "tooth brush", "go work", "coffee", "lunch", "coffee", "tooth brush", "drink alchol", "back home", "bath", "TV/Youtube", "study(C++)", "study(AWS)", "study(ML)", "study(other)", "go bed", "TV/Youtube", "sleeping"}

print(next_action_list)


# 日にちごとにデータ集計（csv作成）して和集合で結合？→予測したい縦軸の部分
# 学習データはそのまま使いたい（dateとactions）ので、各日にちのcsvデータを結合して使う
# 和集合
# print(a | b)  #実行結果：{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12}

# 以下説明
# set型のオブジェクトは波括弧{}で生成できる。カンマ区切りで要素を書く。

# 重複する値は無視されて、一意な値のみが要素として残る。また、setは順序をもたないので、生成時の順序は保持されない。


# 予測したい行動 これらを重複がないように辞書に追加 # 1週間分のデータを集めて辞書にする＝予測したい行動は重複しない
Sunday = {"sleep", "wakeup", "breakfast", "tooth brush", "coffee", "lunch", "tooth brush", "going out", "back home", "bath", "TV/Youtube", "study(AWS)", "study(other)", "go bed", "TV/Youtube", "sleeping"} # 日曜日は外出したりする
Monday = {"sleep", "wakeup", "drink water", "go work", "coffee", "work", "study(C++)", "lunch", "coffee", "tooth brush", "back home", "gym", "buy dinner at 711", "bath", "TV/Youtube", "study(C++)", "study(AWS)", "study(ML)", "study(other)", "go bed", "TV/Youtube", "sleeping"}
Wednesday = {"sleep", "wakeup", "go work", "coffee", "lunch", "coffee", "tooth brush", "study(ML)", "back home", "gym", "buy dinner at ox", "bath", "study(C++)", "go bed", "TV/Youtube", "sleeping"}
Friday = {"sleep", "wakeup", "go work", "coffee", "lunch", "coffee", "tooth brush", "drink alchol", "back home", "bath", "TV/Youtube", "go bed", "TV/Youtube", "sleeping"} # 金曜日は飲み会に行ったりする

week_data = (Sunday | Monday | Wednesday | Friday)
print("week : ", week_data)
print("len : ", len(week_data))
print("len : ", len(Sunday) + len(Monday) + len(Wednesday) + len(Friday))

template = {'wakeup', 'work', 'sleep'}
add = {"study", "back home", "gym", "buy dinner", "bath", 'wakeup'}
print(template)
template.update(add) # これで重複なく更新できる
# template = dict.fromkeys(template, 0)

ans = tuple(template) # 集合のままだと順序を加味できないので、ラベルを順につけられないのでいったんタプルに変換して順番を加味してラベルを付けてから辞書型に変換する
ans = dict([(e, i) for i, e in enumerate(ans)]) # dict([(i, e) for i, e in enumerate(template)])
print(ans)

# template = {'wakeup':0, 'work':1, 'sleep':2}
# add = {"study":3, "back home":4, "gym":5, "buy dinner":6, "bath":7, 'wakeup':0}

# print(template)
# template.update(add) # これで重複なく更新できる
# print(template)


# メモ
# 予測したい行動に対するラベルは一日の時刻でいいかも→そうすれば縦軸も時系列順になる（下から上:0:00~23:59）
# ということはvalueはintじゃなくて文字列でいいかも
memo = {'wakeup' : '0630', 'work' : '0900', 'sleep' : '2350'} # {'wakeup' : 0630, 'work' : 0900, 'sleep' : 2350}
print(memo)
print(memo["sleep"])
print(memo.values())
# memo = lambda x: int(memo.values().replace("'", ""))


dict_data = {'wakeup' : '0630', 'work' : '0900', 'sleep' : '2350'} # memo
# dict_data.update(zip('study', '2200'))
dict_data['study'] = '2200'
print(dict_data)
new_dict = {}
# for k, v in enumerate(dict_data.keys(), dict_data.values()): # keys():
#     # new_dict[i] = k # int(dict_data[k])
#     # print(k)
#     # for k in dict_data.keys():
#     # new_dict[int(k)] = int(dict_data[k])
#     # new_dict[k] = int(v)
#     new_dict = (k, v)
# print(new_dict)

import pprint

# valuesを整数変換
dic = {k: int(v) for k, v in dict_data.items()} # v.replace(',', '')) for k, v in dict_data.items()}
print(f"new: {dic}")
print(dic.values())
sorted_dic = sorted(dic.items(), key=lambda x : x[1]) # dic.values()))
print("sort dic : ")
pprint.pprint(sorted_dic) # f"new sorted: {sorted_dic}")


# 行動を1日の時刻ごとに並び替えるのには以下が応用できそう
# ranks = [
#     {'url': 'example.com', 'rank': '11,279'},
#     {'url': 'facebook.com', 'rank': '2'},
#     {'url': 'google.com', 'rank': '1'}
# ]
# results = sorted(ranks, key=lambda x: int(x["rank"].replace(",", "")))
# print(results)




ranks = [
    {'action': 'wakeup', 'date': '0630'},
    {'action': 'work', 'date': '0900'},
    {'action': 'sleep', 'date': '2350'},
    {'action': 'study', 'date': '2200'}
]
results = sorted(ranks, key=lambda x: int(x["date"].replace(",", "")))
print("result : ")
pprint.pprint(results)
# results = sorted(csv, key=lambda x: int(x["date"].replace(",", "")))
# print("result : ")
# pprint.pprint(results)






# 入力
import pandas as pd

df = pd.read_csv("date_and_actions2.csv",sep=",")
df.columns = ["actions", "date"] # ["date", "actions"]
from datetime import datetime as dt
# df.date = df.date.apply(lambda d: dt.strptime(str(d), "%Y/%m/%d"))
print("df : ")
print(df)
# csv = dict(df) # [date, action]
# print(csv)

# データ追加
print("df.shape[0] : ", df.shape[0])
df.loc[df.shape[0]] = ['buy dinner', 1930]
df.loc[df.shape[0]] = ['coffee', 1230] # ['A5', 'B5', 'C5']

"---------- add test ----------"
# df.loc[df.shape[0]] = ['coffee', 1230] # ['A5', 'B5', 'C5']
df.loc[df.shape[0]] = ['coffee', 1230] # ['A5', 'B5', 'C5']

sunday_data = [
    ["sleep",0], 
    ["wakeup",600], 
    ["breakfast",700], 
    ["tooth brush",715], 
    ["coffee",755], 
    ["lunch",1200], 
    ["tooth brush",1230], 
    ["going out",1400], 
    ["back home",2100], 
    ["bath",2130], 
    ["TV/Youtube",2200], 
    ["study(AWS)",2230], 
    ["study(other)",2300], 
    ["go bed",2320], 
    ["TV/Youtube",2330], 
    ["sleep",2400], # 日曜日は外出したりする
]
Sunday = pd.DataFrame(
    sunday_data,
    #  index=[df.shape[0]]
    columns=['actions', 'date']
)

# Monday = {"sleep", "wakeup", "drink water", "go work", "coffee", "work", "study(C++)", "lunch", "coffee", "tooth brush", "back home", "gym", "buy dinner at 711", "bath", "TV/Youtube", "study(C++)", "study(AWS)", "study(ML)", "study(other)", "go bed", "TV/Youtube", "sleeping"}
# Wednesday = {"sleep", "wakeup", "go work", "coffee", "lunch", "coffee", "tooth brush", "study(ML)", "back home", "gym", "buy dinner at ox", "bath", "study(C++)", "go bed", "TV/Youtube", "sleeping"}
# Friday = {"sleep", "wakeup", "go work", "coffee", "lunch", "coffee", "tooth brush", "drink alchol", "back home", "bath", "TV/Youtube", "go bed", "TV/Youtube", "sleeping"} # 金曜日は飲み会に行ったりする
print('Sunday : ', Sunday)
df = pd.concat([df, Sunday], axis=0) # 1) # 日曜日のデータを行方向に追加
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
df_sort.to_csv('outfile.csv')
import matplotlib.pyplot as plt
import numpy as np
# plt.plot(np.random.randn(df_sort.shape[0]), df_sort*0.01)
# plt.show()

import torch
import random

pred = [random.randint(0, 2400) for p in range(df_sort.shape[0])] # 0, 10)]
pred = [ 724, 1752, 1692,  172,  907,  357,  674, 2044,  164, 1048, 1706,   51,
          569, 1231,  586,  891, 2329]
pred = [600,
        700,
        755,
        900,
        1200,
        1230,
        1400,
        1900,
        1930,
        2100,
        2130,
        2200,
        2230,
        2300,
        2320,
        2330,
        2400,
        ] # データが時刻順かテスト
        
# pred = [random.randint(0, 24) for p in range(df_sort.shape[0])]
# pred = [ 3, 11,  5,  3, 23, 24,  3,  3, 16, 20, 18,  0,  8,  1, 13, 18,  3]


output = torch.tensor([pred])
# print(output)
print(df_sort.shape[0])
x = np.linspace(0, df_sort.shape[0], df_sort.shape[0])
print(x)
print(output)
plt.plot(x, output.numpy()[0])

# # df_sort.columns = ["actions", "date"]
# # next_action_list = [ "sleep", "wakeup", "breakfast", "tooth brush", "go work", "coffee", "lunch", "coffee", "tooth brush", "drink alchol", "back home", "bath", "TV/Youtube", "study(C++)", "study(AWS)", "study(ML)", "study(other)", "go bed", "TV/Youtube", "sleeping"]
# # plt.yticks(x, df_sort['actions']) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], next_action_list)
plt.yticks(df_sort['date'], df_sort['actions'])

print(df_sort['date']) # print(df_sort.set_index('date'))

plt.show()