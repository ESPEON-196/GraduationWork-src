import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import replacement
import allocation
from my_module import netread_part
from my_module import calculator
from my_module import topologymaking
import time

# パラメータ設定
N = 200             #ノード数
K = 20               # エッジサーバーの数
C = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])   # 受付の数
MYU = np.array([300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300])     # 処理率
SIM = 100         # シミュレーション回数の定義

T_0 = 3.0     # シミュレーテッドアニーリング:初期温度
T_f = 0.01      # シミュレーテッドアニーリング:最終温度
gamma = 0.9   # シミュレーテッドアニーリング:冷却速度


rho_avg = np.empty(SIM)
rho_var = np.empty(SIM)
W_avg = np.empty(SIM)
#W_var = np.empty(SIM)
Wmax = np.empty(SIM)
Dmax = np.empty(SIM)
Pmax = np.empty(SIM)


'''
# ノード間の最小ホップ数行列dの用意
if __name__ == '__main__':
    name = input('トポロジーの名前を入力：')
    d = np.array(netread_part.netread(name))
    N = len(d)  # ノード数
    # print(d)
'''

f1 = open('Wmax.txt', 'w')
f2 = open('Dmax.txt', 'w')
f3 = open('Davg.txt', 'w')
f4 = open('Pmax.txt', 'w')
f5 = open('rhobar.txt', 'w')

# ここからシミュレーション開始
time_sta = time.time()
for Dis_avg in range(20, 101, 80): # Dis_avg:ノード間の平均距離
    f1.writelines(['Distance average:', str(Dis_avg), '\n'])
    f2.writelines(['Distance average:', str(Dis_avg), '\n'])
    f3.writelines(['Distance average:', str(Dis_avg), '\n'])
    f4.writelines(['Distance average:', str(Dis_avg), '\n'])
    f5.writelines(['Distance average:', str(Dis_avg), '\n'])
    for lg in range(16, 25):  #lamb_avg:平均の到着率
        lamb_avg = lg * 0.75
        rho = lamb_avg / 30
        Davg = np.zeros(SIM)
        Pavg = np.zeros(SIM)
        for simulation in range(SIM):
            
            d = topologymaking.topologymake2(N, Dis_avg)
            #print(d)

            # 各ノードの到着率を乱数で用意
            while True:
                lamb = np.random.poisson(lamb_avg, N)  
                if np.all(lamb > 0):
                    break
            lamb = lamb.astype(float)
            lamb /= (np.sum(lamb))
            lamb *= lamb_avg * N  # ノードごとの到着率
            #for i in range(N): 
            #lamb[i] = np.random.normal(lamb_avg, np.sqrt(lamb_var))

            # 変数初期化
            Eta = 0.0                      # 各エッジサーバの負荷の最大値(これの最小化を目指す)
            Eta_last = np.inf              # Etaの値保持用
            S = np.zeros(N, dtype=int)     # エッジが配置されているなら1, いないなら0
            y = np.full(N, 65536)          # ノードiが割り当てられたエッジサーバのノード番号
            Delay = np.empty(N)
            T = T_0                        # ループ用
            E = np.full(K, 65536)          # エッジサーバのあるノード番号
            Xc = np.zeros(N, dtype=int)    # ノードiに置かれたエッジサーバのプロセッサ数
            Xm = np.zeros(N, dtype=int)    # ノードiに置かれたエッジサーバの処理率
            count = 0

            # ここから提案手法プログラム
            
            # エッジサーバ仮配置 (到着率が多い順にK個)
            for i in range(K):
                MAX = 0.0
                for j in range(N):
                    if lamb[j] >= MAX and S[j] != 1:
                        MAX = lamb[j]
                        E[i] = j
                S[E[i]] = 1
                Xc[E[i]] = C[i]
                Xm[E[i]] = MYU[i]

            # ループ開始
            while T > T_f:
                # 各ノードのエッジサーバ割り当て決定
                y = allocation.allocation_new_ver1(S, N, K, lamb, d, Xc, Xm)

                #どのエッジサーバに割り当てられたか確認用
                for i in range(N):
                    #print('ノード', i, ' → エッジサーバノード', y[i])
                    if y[i] == 65536:
                        print('Allocation error')
                        exit()

                lambofcluster = calculator.lambdaofcluster(lamb, N, y)  # クラスタの到着率の合計
                # pofcluster = calculator.lossofcluster(lambofcluster, MYU, N, C)  # クラスタの呼損率
                Wofcluster = calculator.SystemTime(lambofcluster, Xm, N, Xc) # クラスタの平均系内滞在時間

                #エッジサーバをクラスタ内で再配置(割り当ても同時に変更)
                replacement.replacement(S, y, N, d, Xc, Xm)

                #エッジサーバをクラスタ内で再配置(割り当ても同時に変更)
                #replacement.replacement(S, y, N, load, Xc, Xm)

                # 合計遅延時間 (ms)
                for i in range(N):
                    Delay[i] = d[i][y[i]] + (Wofcluster[y[i]] * 1000)

                #遅延の最大値
                Eta = max(Delay)

                #シミュレーテッドアニーリング
                if Eta < Eta_last:
                    #Eta, S, yをそれぞれ保持
                    Eta_last = Eta
                    S_last = S.copy()
                    Xc_last = Xc.copy()
                    Xm_last = Xm.copy()
                    y_last = y.copy()

                else:
                    delta = Eta - Eta_last
                    P = min(1, np.exp(-delta/T))
                    judge = np.random.choice([True, False], p=[P, (1-P)])
                    print(P)
                    print(judge)
                    if judge:
                        Eta_last = Eta
                        S_last = S.copy()
                        Xc_last = Xc.copy()
                        Xm_last = Xm.copy()
                        y_last = y.copy()
                    else:
                        S = S_last.copy()
                        Xc = Xc_last.copy()
                        Xm = Xm_last.copy()
                        y = y_last.copy()
                T *= gamma
                count += 1


            lambofcluster = calculator.lambdaofcluster(lamb, N, y)  # クラスタの到着率の合計
            Wofcluster = calculator.SystemTime(lambofcluster, Xm, N, Xc) # クラスタの平均系内滞在時間
            print('simuration', simulation + 1)

            rho_sum = 0.0
            for j in range(N):
                if S[j] == 1:
                    print('クラスタ', j, 'の総到着率:', lambofcluster[j])
                    rho_sum += lambofcluster[j] / (Xc[j] * Xm[j])
            
            W_sum = 0.0
            for j in range(N):
                if S[j] == 1:
                    print('クラスタ', j, 'の平均系内滞在時間[ms]:', (Wofcluster[j] * 1000))
                    W_sum += Wofcluster[j] * 1000
            
            p = 0
            for i in range(N):
                Delay[i] = d[i][y[i]] + (Wofcluster[y[i]] * 1000) 
                Davg[simulation] += Delay[i] * lamb[i]
                Pavg[simulation] += d[i][y[i]] * lamb[i]
                if p < d[i][y[i]]:
                    p = d[i][y[i]]

            Dmax[simulation] = max(Delay)
            Davg[simulation] /= (np.sum(lamb))
            Pmax[simulation] = p
            Pavg[simulation] /= (np.sum(lamb))

            rho_avg[simulation] = rho_sum / K
            rho_varsum = 0.0
            Wmax[simulation] = max(Wofcluster) * 1000
            W_avg[simulation] = W_sum / K
            
            #W_varsum = 0.0
            for j in range(N):
                if S[j] == 1:
                    rho_varsum += (lambofcluster[j] / (Xc[j] * Xm[j]) - rho_avg[simulation]) ** 2
                    #W_varsum += (Wofcluster[j] * 1000 - W_avg[simulation]) ** 2
            rho_var[simulation] = rho_varsum / K
            #W_var[simulation] = W_varsum / K
            
            print('最大系内滞在時間[ms]:', (max(Wofcluster) * 1000))
            print('最大遅延時間[ms]:', max(Delay))
            print('平均遅延時間[ms]:', Davg[simulation])
            print()   
        
    
        # loss_sum_avg = np.sum(loss_sum) / SIM
        rho_avg_avg = np.sum(rho_avg) / SIM
        rho_var_avg = np.sum(rho_var) / SIM
        #W_var_avg = np.sum(W_var) / SIM
        Wmax_avg = np.sum(Wmax) / SIM
        Dmax_avg = np.sum(Dmax) / SIM
        Davg_avg = np.sum(Davg) / SIM
        Pmax_avg = np.sum(Pmax) / SIM
        Pavg_avg = np.sum(Pavg) / SIM

        # print('the average of loss', SIM, 'simulations is...', loss_sum_avg)
        print('the average of rho_avg', SIM, 'simulations is...', rho_avg_avg)
        print('the average of rho_var', SIM, 'simulations is...', rho_var_avg)
        print('the average of max wait delay[ms]', SIM, 'simulations is...', Wmax_avg)
        print('the average of max prop delay[ms]', SIM, 'simulations is...', Pmax_avg)
        print('the average of avg delay', SIM, 'simulations is...', Pavg_avg)
        print('the average of max delay[ms]', SIM, 'simulations is...', Dmax_avg)
        print('the average of avg delay', SIM, 'simulations is...', Davg_avg)

        
        f1.writelines([str(rho), ' ', str(Wmax_avg), '\n'])
        f2.writelines([str(rho), ' ', str(Dmax_avg), '\n'])
        f3.writelines([str(rho), ' ', str(Davg_avg), '\n'])
        f4.writelines([str(rho), ' ', str(Pmax_avg), '\n'])
        f5.writelines([str(rho), ' ', str(rho_var_avg), '\n'])

    f1.write('\n')
    f2.write('\n')
    f3.write('\n')
    f4.write('\n')
    f5.write('\n')

f1.close()
f2.close()
f3.close()
f4.close()
f5.close()

time_end = time.time()
tim = time_end - time_sta

print(tim)