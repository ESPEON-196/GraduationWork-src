import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def calc_size_of_GC(G):
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    G0 = G.subgraph(Gcc[0])
    return len(G0)


def topologymake(N, Distance):
    SIZE = 0 # 全ノードが全ノードにいけるときSIZE=ノード数
    while SIZE < N:
        #G = nx.erdos_renyi_graph(N, 3 / (N - 1))  # ランダムネットワーク←微妙
        G = nx.watts_strogatz_graph(N, 4, 0.2)  # スモールワールドネットワーク←多分これ
        for (u,v) in G.edges():
            if np.random.rand() < 0.1: 
                G.remove_edge(u, v)
        #G = nx.barabasi_albert_graph(N, 2)  # スケールフリーネットワーク←微妙
        SIZE = calc_size_of_GC(G)

    # nx.draw(G, with_labels = True)
    # plt.show()
    # plt.close()
    
    #ここまでで、ネットワークトポロジが完成、これ以降どういう分析するかは自由
    d = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            d[i][j] = nx.shortest_path_length(G, source=i, target=j)
            #d[i][j] = nx.shortest_path_length(G, source=i, target=j, weight='weight')
            
    print(nx.info(G)) # 種々の情報を出力
    print('平均最短経路', nx.average_shortest_path_length(G))
    print('クラスタ係数：', nx.average_clustering(G)) # ネットワーク全体のクラスタ係数を出力
    #main関数にｄ配列を返す。
    return d

def topologymake2(N, Distance):
    SIZE = 0 # 全ノードが全ノードにいけるときSIZE=ノード数
    while SIZE < N:
        #G = nx.erdos_renyi_graph(N, 3 / (N - 1))  # ランダムネットワーク←微妙
        G = nx.watts_strogatz_graph(N, 4, 0.2)  # スモールワールドネットワーク←多分これ
        for (u,v) in G.edges():
            if np.random.rand() < 0.1: 
                G.remove_edge(u, v)
        #G = nx.barabasi_albert_graph(N, 2)  # スケールフリーネットワーク←微妙
        SIZE = calc_size_of_GC(G)

    # リンクに重み付け(伝搬遅延[ms], RTTのため2倍)
    for (u,v) in G.edges(): 
        r = 0
        while r <= 0:
            r = np.random.poisson(Distance)
        G[u][v]['weight'] = (r / 200) * 2.0  # 伝搬遅延[ms] = (平均距離[km]の乱数(正規分布)) / 光速の約7割[km/ms]
        #print(G[u][v])

    # nx.draw(G, with_labels = True)
    # plt.show()
    # plt.close()
    
    #ここまでで、ネットワークトポロジが完成、これ以降どういう分析するかは自由
    d = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            d[i][j] = nx.shortest_path_length(G, source=i, target=j, weight='weight')
            
    print(nx.info(G)) # 種々の情報を出力
    print('平均最短経路', nx.average_shortest_path_length(G))
    print('クラスタ係数：', nx.average_clustering(G)) # ネットワーク全体のクラスタ係数を出力
    #main関数にｄ配列を返す。
    return d