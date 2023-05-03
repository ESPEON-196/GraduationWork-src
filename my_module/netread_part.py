import networkx as nx
import matplotlib.pyplot as plt

def netread(name):
    name += '.txt'
    f = open(name, 'r', encoding='utf-8')

    datalist = f.readlines()
    node_num = int(datalist[4].strip())

    d = [[0 for j in range(node_num)] for i in range(node_num)]
    array = [[0 for j in range(node_num)] for i in range(node_num)]
    #array_apart =[0 for i in range(node_num)]
    array_apart_str = [0 for i in range(node_num)]

    for i in range(node_num):
        listnum = i + 6
        array_apart_str = datalist[listnum].strip()
        array_apart_str = array_apart_str.split(',')
        for j in range(node_num):
            array[i][j] = int(array_apart_str[j])

    f.close()

    #ここまでで、txtファイルからのインポートが終了

    G = nx.Graph()
    for i in range(node_num):
        G.add_node(i)

    for i in range(node_num):
        for j in range(node_num):
            if array[i][j] != 0:
                G.add_edge(i, j)
                array[j][i] = 0

    #ここまでで、ネットワークトポロジが完成、これ以降どういう分析するかは自由

    for i in range(node_num):
        for j in range(node_num):
            d[i][j] = nx.shortest_path_length(G, source=i, target=j)
    #main関数にｄ配列を返す。
    return d

