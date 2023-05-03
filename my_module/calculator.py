import math
import numpy as np


def lambdaofcluster(lamb, N, y):
    lambofcluster = np.zeros(N)
    for j in range(N):
        for i in range(N):
            if y[i] == j:
                lambofcluster[j] += lamb[i]
    
    return lambofcluster

def lossofcluster(lambofcluster, MYU, N, C):
    denominator = np.zeros(N)
    for k in range(C + 1):
        denominator += (lambofcluster ** k) * np.float64((MYU ** (C - k)) * math.factorial(C) / math.factorial(k))
    pofcluster = lambofcluster ** C / denominator

    return pofcluster

# プロセッサ数、処理率が全て同じとき
def SystemTime_old(lambofcluster, MYU, N, C):
    rho = lambofcluster / MYU
    denominator = np.zeros(N) # p0
    L = np.zeros(N)
    W = np.empty(N)
    for k in range(1, C):
        denominator += (rho ** k) / math.factorial(k)
        L += (rho ** k) / math.factorial(k - 1)
    denominator += (rho ** C) / (math.factorial(C - 1) * (C - rho)) + 1  # (rho^0/0!) = 1をここで足してる
    L += (rho ** C) * (C ** 2 - (C - 1) * rho) / (math.factorial(C - 1) * ((C - rho) ** 2))
    W = L / (denominator * lambofcluster)
    for i in range(N):
        if lambofcluster[i] == 0:
            W[i] = 0

        elif W[i] < 0 or W[i] > 0.2:
            W[i] = 0.2

    return W


def SystemTime(lambofcluster, MYU, N, C):
    rho = lambofcluster / MYU
    denominator = np.zeros(N) # p0
    L = np.zeros(N)
    W = np.zeros(N)
    for i in range(N):
        for k in range(1, C[i]):
            denominator[i] += (rho[i] ** k) / math.factorial(k)
            L[i] += (rho[i] ** k) / math.factorial(k - 1)
        if C[i] > 0:
            denominator[i] += (rho[i] ** C[i]) / (math.factorial(C[i] - 1) * (C[i] - rho[i])) + 1  # (rho^0/0!) = 1をここで足してる
            L[i] += (rho[i] ** C[i]) * (C[i] ** 2 - (C[i] - 1) * rho[i]) / (math.factorial(C[i] - 1) * ((C[i] - rho[i]) ** 2))
            W[i] = L[i] / (denominator[i] * lambofcluster[i])
            
    return W