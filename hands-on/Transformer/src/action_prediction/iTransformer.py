import pandas as pd
import matplotlib.pyplot as plt

#機械受注長期時系列
machinary_order_data = 'machine_data.csv'
df_machinary_order_data = pd.read_csv(machinary_order_data, sep=",")
print(df_machinary_order_data)
df_machinary_order_data.columns = ["年","月","風水力機械","運搬機械","産業用ロボット","金属加工機械","化学機械","冷凍機械","合成樹脂","繊維機械","建設機械","鉱山機械","農林用機械","その他"] # ["datetime","id","value"]
# plt.plot(df_machinary_order_data)
plt.plot(df_machinary_order_data["産業用ロボット"]) # .values) # , df_machinary_order_data['actions'])
plt.show()

# df_machinary_order_data.columns = ["年","月","風水力機械","運搬機械","産業用ロボット","金属加工機械","化学機械","冷凍機械","合成樹脂","繊維機械","建設機械","鉱山機械","農林用機械","その他"] # ["datetime","id","value"]
# plt.plot(df_machinary_order_data)
plt.plot(df_machinary_order_data["合成樹脂"]) # .values) # , df_machinary_order_data['actions'])
plt.show()

# machinary_order_data_2 = 'machine_data_2.csv'
# df_machinary_order_data_2 = pd.read_csv(machinary_order_data_2, sep=",")
# print(df_machinary_order_data_2)
# df_machinary_order_data_2.columns = ["年","月","原子力原動機","火水力原動機","重電機","内燃機関","発電機","その他重電機","電子計算機等","通信機","電子応用装置","電気計測器"] # ["datetime","id","value"]
# # plt.plot(df_machinary_order_data)
# plt.plot(df_machinary_order_data_2["重電機"].values)

# plt.show()