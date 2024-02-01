from flask import Flask

app = Flask(__name__) # __name__という名称を付けてFlaskのインスタンス化
    # __name__はpythonがもともと確保している変数（予約語）であり、
    # 特別開発者が値を代入するコードを書かなくても最初からファイルの名称が代入されています。
    # 今回の場合は__name__には"app"(実行ファイル名)が代入されます。


# @~でルート設定して、その直下の関数が実行される = 2つで1セット
@app.route("/") # 「引数として渡しているパスに対してアクセスが来たら、この直下の関数を実行する」
    # 今回の場合は"/"、つまりhttp://127.0.0.1:5000/
    # （一番最後のスラッシュがルートを表している）にアクセスが来たら以下の関数が実行されることになります。

# **********
# 以下別例
@app.route('/ringo')
    # 今回の場合はhttp://127.0.0.1:5000/ringoに対してアクセスがきたら、
    # apple関数が実行されることになります。
def apple():
    return "apple"
# **********

def hello_world():
    return "Hello, World!"

app.run() # Flaskアプリケーションを起動させています。
    # 起動後は先ほど説明した@app.route('/’)が効力を持つようになります。