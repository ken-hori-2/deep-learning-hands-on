import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
from sklearn.preprocessing import MinMaxScaler

#vizualize the datasets in seaborn
sns.get_dataset_names()

#loading the dataset
flight_dataset = sns.load_dataset('flights')

#To vizualize the loaded dataset
print(flight_dataset)

# #plotting the data
# plt.plot(flight_dataset['passengers'])
# plt.title('Month vs Passenger')
# plt.ylabel('Total Passengers')
# plt.xlabel('Months')
# plt.grid(True)
# plt.show()




# 自分のデータ
import pandas as pd

# CSV読み込み
# df = pd.read_csv("data_test.csv",sep=",")
df = pd.read_csv("data_action.csv",sep=",")
# df.columns = ["year", "month", "passengers"] # ["datetime","id","value"]
df.columns = ["date", "youbi", "actions"] # ["datetime","id","value"]
print(df) # .head())

# plt.plot(df['passengers'])
# plt.title('Month vs Passenger')
# plt.ylabel('Total Passengers')
# plt.xlabel('Months')
# plt.grid(True)
# plt.show()






# # add
from datetime import datetime as dt

# # df.datetime = df.datetime.apply(lambda d: dt.strptime(str(d), "%Y%m%d%H%M%S"))
df.date = df.date.apply(lambda d: dt.strptime(str(d), "%Y%m%d%H%M%S"))
print(df)


# # df_by_id= df.groupby("id")["value"].count().reset_index()
# df_by_id= df.groupby("youbi")["actions"].count().reset_index()
# print(df_by_id)


# import seaborn as sns
# import matplotlib.pyplot as plt
# id_df = pd.DataFrame(df_by_id)
# # sns.distplot(id_df.value, kde=False, rug=False, axlabel="record_count",bins=10)
# # sns.distplot(id_df.actions, kde=False, rug=False, axlabel="record_count",bins=10)

# start_datetime_by_id = df.groupby(["youbi"])["date"].first().reset_index()
# df_date = pd.DataFrame(start_datetime_by_id)
# print(df_date)

# sns.distplot(df.date.dt.month, kde=False, rug=False, axlabel="record_generate_date",hist_kws={"range": [1,30]}, bins=30)
# sns.distplot(df.date, df.actions, kde=False, rug=False, axlabel="date",bins=10)
# plt.plot(df['date'], df['actions'])
plt.plot(df['youbi'], df['actions'])
plt.title('youbi vs actions')
plt.ylabel('action')
plt.xlabel('date')
plt.grid(True)

plt.show()
