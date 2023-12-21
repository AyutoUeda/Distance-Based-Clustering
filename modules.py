import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import math


def hierachical_clustering(data, time_analyze, threshold, method="single"):
    """階層クラスタリングを実行する関数
    
    任意の時間における座標データに対し、階層クラスタリングを実行.
      
    その時点におけるクラスタ数とどのクラスタに属するかを表す各点のラベルを返す.  
    
    Args:
        data (```dataframe```): 座標データ
        time_analyze (````int````): 分析する秒数（行数） 
        threshold (```int```): クラスタ間の距離の閾値  
        method (```str```): 距離の測定方法(default="single")  
        
    Returns:
        model.n_clusters_ (```int```): クラスタ数
        model.labels_ (```list```): 各点のラベル
        coordinate_data (```list```): 指定した時間（行数）における各点の座標
        model (```model```): モデル
        
    Examples:
        >>> n_clusters, labels, coordinate_data, model = hierachical_clustering(data, 0, 100, "single")
        
    """
    coordinate_data = []
    
    # 指定した時間（行数）における各点の座標を取得
    for i in range(0, data.shape[1], 2):
      x = data.iloc[time_analyze][i]
      y = data.iloc[time_analyze][i+1]
      coordinate_data.append([int(x), int(y)])

    # list --> array
    X = np.array(coordinate_data)
    
    # クラスタリング
    model = AgglomerativeClustering(
        n_clusters=None, linkage=method, compute_full_tree=True, distance_threshold=threshold
    )

    model.fit(X)
    
    return model.n_clusters_, model.labels_, coordinate_data, model

def exception_size1(labels: list):
    """クラスタサイズが1のものを除いた際のクラスタ数を返す関数"""
    
    clusters_size = np.bincount(labels) # <-- ラベル別のクラスタサイズ
    # print(clusters_size, "<--ラベル別のクラスタサイズ")

    n = sum(x>1 for x in clusters_size) 
    # print(n, "<--クラスタサイズが1より大きいものの数")
    
    return clusters_size, n

def calculate_radius(coordinate, center_coordinate):
    """半径を計算する関数
    
    Args:
        coordinate (```list```): 各点の座標リスト [[x1,y1], [x2,y2], ....]
        center_coordinate (```list```): 中心座標 [x, y]
        
    Returns:
        R (```float```): 半径
        
    
    """

    points = np.array(coordinate)
    center = np.array(center_coordinate)

    R = 0

    for point in points:
        a = np.abs(point-center) # ベクトルの差
        r = math.sqrt(a[0]**2 + a[1]**2)
        
        # 半径の更新
        if r > R:
            R = r
    return R

def calculate_center(coordinates: list):
    """クラスタの中心座標を計算する関数
    
    Args:
        coordinates (```list```): 各点の座標リスト [[x1,y1], [x2,y2], ....]
    
    Returns:
        center_coordinate (```list```): 中心座標 [x, y]
    
    """
    
    x_sum = 0
    y_sum = 0

    for coordinate in coordinates:
        x_sum += coordinate[0]
        y_sum += coordinate[1]

    center_x = x_sum / len(coordinates)
    center_y = y_sum / len(coordinates)
    center_coordinate = [round(center_x, 2), round(center_y, 2)]

    return center_coordinate

def calculate_cluster_centers(labels, coordinate, c_size):
    """クラスタの中心をすべて計算する関数
    
    Args:
        labels (```list```): 各点のクラスタラベル
        coordinate (```list```): 各点の座標リスト [[x1,y1], [x2,y2], ....]
        c_size (```list```): 各クラスタのサイズリスト(0〜)
        
    Returns:
        cluster_centers (```list```): 各クラスタの中心座標リスト [[x1,y1], [x2,y2], ....]
        list_of_radius (```list```): 各クラスタの半径リスト [r1, r2, ....]
        
    """
    
    centers = []
    list_of_radius = []
    for i, size in enumerate(c_size): # i --> ラベル番号
        if size > 1:
            points = []
            for j, label in enumerate(labels):
                # ラベルが一致したら座標をpointsに追加
                if i == label:
                    points.append(coordinate[j])
                    
            # 中心座標を計算
            center = calculate_center(points) 
            centers.append(center)
            
            # 半径を計算
            R = calculate_radius(points, center)
            list_of_radius.append(R)
            
            points = []

    return centers, list_of_radius

if __name__ == "__main__":
    
    data = pd.DataFrame([[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 54,33], 
                        [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]])
    n_clusters, labels, coordinate_data, model = hierachical_clustering(data, 0, 2, "single")

    # labels = [0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 3]
    exception_size1(labels)

    centers, list_of_radius = calculate_cluster_centers(labels, coordinate_data, np.bincount(labels))
