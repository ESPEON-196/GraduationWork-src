import numpy as np

# プロセッサ数、処理率が全て同じとき
def allocation_new_ver0(S, N, K, lamb, d):
    # [ノード番号, 到着率]の2次元配列を作成
    lamb_list = np.empty((N, 2))
    for i in range(N):
        lamb_list[i][0] = i
        lamb_list[i][1] = lamb[i]
    lamb_list_sorted = lamb_list[np.argsort(lamb_list[:, 1])[::-1]]

    y = [65536 for i in range(N)]
    lambsup = np.full(N, np.sum(lamb) / K)
    # エッジサーバの到着率は先に行う
    for i in range(N):
        if S[i] == 1:
            lambsup[i] -= lamb[i]
            y[i] = i
        else:
            lambsup[i] = -np.inf # エッジサーバが置かれてないノードの閾値は下のargmaxに引っかからないように設定

    # 残りのノードを到着率が高い順に割り当て
    for i in range(N):
        if S[int(lamb_list_sorted[i][0])] != 1:
            MIN = np.inf
            for j in range(N):
                if S[j] == 1:
                    if lamb_list_sorted[i][1] < lambsup[j]:
                        L = lamb_list_sorted[i][1] * d[int(lamb_list_sorted[i][0])][j]
                        if L < MIN:
                            y[int(lamb_list_sorted[i][0])] = j
                            MIN = L
            if y[int(lamb_list_sorted[i][0])] == 65536:
                y[int(lamb_list_sorted[i][0])] = np.argmax(lambsup) # 分散重視ならこっち
                '''
                for j in range(N): # 閾値超えたとき、一番近いノードへやるならこっち
                    if S[j] == 1:
                        L = lamb_list_sorted[i][1] * d[int(lamb_list_sorted[i][0])][j]
                        if L < MIN:
                            y[int(lamb_list_sorted[i][0])] = j
                            MIN = L
                '''
            
            lambsup[y[int(lamb_list_sorted[i][0])]] -= lamb_list_sorted[i][1]

    return y

def allocation_new_ver1(S, N, K, lamb, d, Xc, Xm):
    # [ノード番号, 到着率, パラメータ]の2次元配列を作成
    lamb_list = np.empty((N, 2))
    alpha = np.zeros(K)            

    for i in range(N):
        lamb_list[i][0] = i
        lamb_list[i][1] = lamb[i]
    lamb_list_sorted = lamb_list[np.argsort(lamb_list[:, 1])[::-1]]

    y = [65536 for i in range(N)]
    
    #lambsup = np.full(N, (Xc * Xm * (np.sum(lamb) / np.sum(Xc * Xm)))) #要検討
    lambsup = np.full(N, (Xc * Xm - ((np.sum(Xc * Xm) - np.sum(lamb)) / K)))
    
    # エッジノードの到着率は先に行う
    for i in range(N):
        if S[i] == 1:
            lambsup[i] -= lamb[i]
            y[i] = i
        else:
            lambsup[i] = -np.inf # エッジサーバが置かれてないノードの閾値は下のargmaxに引っかからないように設定

    # 残りのノードをパラメータが高い順に割り当て
    for i in range(N):
        if S[int(lamb_list_sorted[i][0])] != 1:
            MIN = np.inf
            for j in range(N):
                if S[j] == 1:
                    if lamb_list_sorted[i][1] < lambsup[j]:
                        L = d[int(lamb_list_sorted[i][0])][j]
                        if L < MIN:
                            y[int(lamb_list_sorted[i][0])] = j
                            MIN = L
            if y[int(lamb_list_sorted[i][0])] == 65536:
                y[int(lamb_list_sorted[i][0])] = np.argmax(lambsup) # 分散重視ならこっち
                '''
                for j in range(N): # 閾値超えたとき、一番近いノードへやるならこっち
                    if S[j] == 1:
                        L = lamb_list_sorted[i][1] * d[int(lamb_list_sorted[i][0])][j]
                        if L < MIN:
                            y[int(lamb_list_sorted[i][0])] = j
                            MIN = L
                '''
            
            lambsup[y[int(lamb_list_sorted[i][0])]] -= lamb_list_sorted[i][1]

    return y

def allocation_new_ver2(S, N, K, lamb, d, Xc, Xm):
    # [ノード番号, 到着率, パラメータ]の2次元配列を作成
    lamb_list = np.empty((N, 3))
    alpha = np.zeros(K)            

    for i in range(N):
        lamb_list[i][0] = i
        lamb_list[i][1] = lamb[i]
        if S[i] != 1:
            Enum = 0  # カウント変数
            for j in range(N):
                if S[j] == 1:
                    alpha[Enum] = d[i][j]
            #lamb_list[i][2] = np.mean(alpha) - min(alpha)  
            lamb_list[i][2] = max(alpha)  # こっちかな？


    lamb_list_sorted = lamb_list[np.argsort(lamb_list[:, 2])[::-1]]  # 配列をパラメータが高い順にソート

    y = [65536 for i in range(N)]
    
    #lambsup = np.full(N, (Xc * Xm * (np.sum(lamb) / np.sum(Xc * Xm)))) #要検討
    lambsup = np.full(N, (Xc * Xm - ((np.sum(Xc * Xm) - np.sum(lamb)) / K)))
    
    # エッジノードの到着率は先に行う
    for i in range(N):
        if S[i] == 1:
            lambsup[i] -= lamb[i]
            y[i] = i
        else:
            lambsup[i] = -np.inf # エッジサーバが置かれてないノードの閾値は下のargmaxに引っかからないように設定

    # 残りのノードをパラメータが高い順に割り当て
    for i in range(N):
        if S[int(lamb_list_sorted[i][0])] != 1:
            MIN = np.inf
            for j in range(N):
                if S[j] == 1:
                    if lamb_list_sorted[i][1] < lambsup[j]:
                        L = d[int(lamb_list_sorted[i][0])][j]
                        if L < MIN:
                            y[int(lamb_list_sorted[i][0])] = j
                            MIN = L
            if y[int(lamb_list_sorted[i][0])] == 65536:
                y[int(lamb_list_sorted[i][0])] = np.argmax(lambsup) # 分散重視ならこっち
                '''
                for j in range(N): # 閾値超えたとき、一番近いノードへやるならこっち
                    if S[j] == 1:
                        L = lamb_list_sorted[i][1] * d[int(lamb_list_sorted[i][0])][j]
                        if L < MIN:
                            y[int(lamb_list_sorted[i][0])] = j
                            MIN = L
                '''
            
            lambsup[y[int(lamb_list_sorted[i][0])]] -= lamb_list_sorted[i][1]

    return y