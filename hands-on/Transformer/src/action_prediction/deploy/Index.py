import pandas as pd
import pprint

df = pd.read_csv("pred_date_actions.csv",sep=",")
df = df.drop(df.columns[0], axis=1)
df = df.reset_index(drop=True)
print(df)
# df = df.set_index('date',drop=True)

print(df)
df = df.to_dict()
pprint.pprint(df)
# print(df['date'])
# print(df[1900-02-27 07:03:00])

print("*****")
key = 10
print("key = ", key)
print("date : ", df['date'][key])
print("action : ", df['actions'][key])
print("*****")