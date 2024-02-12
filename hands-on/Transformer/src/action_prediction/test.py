import pandas as pd

machinary_order_data = 'machine_data.csv'
df_machinary_order_data = pd.read_csv(machinary_order_data) # , sep=",")
# df_machinary_order_data.columns = ["年","月","風水力機械","運搬機械","産業用ロボット","金属加工機械","化学機械","冷凍機械","合成樹脂","繊維機械","建設機械","鉱山機械","農林用機械","その他"] # ["datetime","id","value"]

for i in df_machinary_order_data: # .columns:
    print(i)
    i.replace('"', '')