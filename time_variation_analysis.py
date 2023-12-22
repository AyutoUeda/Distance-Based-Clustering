"""
時間ごとにクラスタ数、最大クラスタサイズ、最小クラスタサイズ、最大半径、最小半径を調べる。
クラスタ数はクラスタサイズが1より大きいものの数を表す。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from modules import hierachical_clustering, calculate_cluster_centers


def time_variation(data, threshold=35, method="single"):
    """各特徴に対して時間変化を調べる関数
    
    Args:
        data (```dataframe```): 座標データ
        threshold (```int```): クラスタ間の距離の閾値(default=35)
        method (```str```): 距離の測定方法(default="single")
        
    Returns:
        time_variation_dict (```dict```): 時間変化を調べた結果  
        - n_clusters (```list```): クラスタ数  
        - max_clustersize (```list```): 最大クラスタサイズ  
        - min_clustersize (```list```): 最小クラスタサイズ(size1を除く)  
        - max_radius (```list```): 最大半径  
        - min_radius (```list```): 最小半径  
        
    Notes:
        n_clustersはクラスタサイズが1より大きいものの数を表す。
        
    """
    time_variation_dict = {
        "n_clusters":[], 
        "max_clustersize":[],
        "min_clustersize":[],
        "max_radius":[],
        "min_radius":[],
    }  

    for time_analyze in range(0, len(data)):    
        _, labels, coordinate_data, _ = hierachical_clustering(data, 
                                                                time_analyze, 
                                                                threshold=threshold, 
                                                                method=method)
        
        clusters_size = np.bincount(labels) # <-- ラベル別のクラスタサイズ
        except_size1 = [x for x in clusters_size if x > 1]
        
        n_withoutsize1 = sum(x>1 for x in clusters_size) # <--クラスタサイズが1より大きいものの数
        time_variation_dict["n_clusters"].append(n_withoutsize1)
        
        time_variation_dict["max_clustersize"].append(max(clusters_size))
        time_variation_dict["min_clustersize"].append(min(except_size1)) # サイズ1のクラスタを除いた最小クラスタサイズ
        
        _, list_of_radius = calculate_cluster_centers(labels, coordinate_data, clusters_size)
        
        time_variation_dict["max_radius"].append(max(list_of_radius))
        time_variation_dict["min_radius"].append(min(list_of_radius))
        
    return time_variation_dict
    

if __name__ == "__main__":
    data = pd.read_csv("all_result.csv")
    
    timevariation_dict = time_variation(data, threshold=35, method="single")

    # =====save=====
    time_keys_and_values = {
        "n_clusters":timevariation_dict["n_clusters"],
        "max_clustersize":timevariation_dict["max_clustersize"],
        "min_clustersize":timevariation_dict["min_clustersize"],
        "max_radius":timevariation_dict["max_radius"],
        "min_radius":timevariation_dict["min_radius"],
    }
    
    
    df = pd.DataFrame(time_keys_and_values)
    # print(df)
    
    df.to_csv("time_variation.csv", index=False)
