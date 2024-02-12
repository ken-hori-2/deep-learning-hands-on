

# Classの書き方と継承

from typing import Any


class Car(): # object):

    # 変数（メンバ） = ガソリン，スピード，室内温度
    def __init__(self, ini_gasoline, ini_speed, ini_temp):
        self.gasoline = ini_gasoline
        self.speed = ini_speed
        self.temp = ini_temp
    
    # 関数（メソッド） = 機能
        # 走る[ガソリンを減らす, スピード上昇]
    def run(self):
        self.gasoline -= 0.1
        self.speed += 0.3
    
        # エアコン
    def aircon(self, target_temp):
        self.temp = target_temp


# Carクラスを継承してハイブリッドカーを定義
class HybridCar(Car):
    def __init__(self, ini_gasoline, ini_speed, ini_temp):
        super().__init__(ini_gasoline, ini_speed, ini_temp) # 親クラスに変数の初期化（super=親クラス）
        self.state = "stopped"
    
    # run部分のみ書き換え = オーバーライド（一部上書き）
    def run(self):
        self.gasoline -= 0.5
        self.speed += 0.3
        self.state = "runnning"
    
    def __call__(self):
        print("I'm " + self.state)
    
    def __len__(self):
        return 4 # len(self)




if __name__ == "__main__":

    my_car = Car(10, 0, 20)

    my_car.run()

    print("Car's Speed: {}".format(my_car.speed))

    my_car2 = HybridCar(10, 0, 20)
    print("Car's gasoline: {}".format(my_car2.gasoline))

    my_car2.run()

    print("Hybrid's Speed: {}".format(my_car2.speed))

    print("Hybrid's gasoline: {}".format(my_car2.gasoline))

    print("**********")
    my_car3 = HybridCar(10, 0, 20)
    
    # __call__ で定義した内容を実行
    my_car3()
   
    # __len__　で定義した内容を実行
    print(my_car3.__len__())
    