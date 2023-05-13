import numpy as np
import random

# 距離を平等にしたい！
# zはエッジサーバが配置されてたノード
# i, jはエッジzに割り当てられているノードでjが新エッジ配置候補
def replacement(S, y, N, d, Xc, Xm):
    X = []
    for k in range(N):
        if S[k] == 1:
            X += [k]
    s = random.choice(X)
    print(s)
    S[s] = 0
    MAX = 0
    for j in range(N):
        if y[j] == s:
            a = np.zeros(N)
            for i in range(N):
                if y[i] == s:
                    a[i] = d[i][j]
            if MAX <= 1 / max(a):
                MAX = 1 / max(a)
                safter = j
    S[safter] = 1
    if s != safter:
        Xc[safter] = Xc[s]
        Xm[safter] = Xm[s]
        Xc[s] = 0
        Xm[s] = 0

    # print(s, '→', safter)
    
    for i in range(N):
        if y[i] == s:
            y[i] = safter

    return S
                       