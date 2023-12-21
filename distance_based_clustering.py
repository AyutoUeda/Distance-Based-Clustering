from modules import hierachical_clustering, calculate_cluster_centers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_coordinate(data, lower_threshold, upper_threshold, time_analyze, method="single"): 
    """指定された行数（時間）における座標データとクラスタをプロットする関数
    
    Args:
        data (```dataframe```): 座標データ
        lower_threshold (```int```): 閾値の下限
        upper_threshold (```int```): 閾値の上限
        time_analyze (```int```): 分析する秒数（行数） 
        method (```str```): 距離の測定方法(default="single")
    
    """

    for distance_threshold in range(lower_threshold, upper_threshold+1):
        
        n_clusters, labels, coordinate_data, model = hierachical_clustering(data, 
                                                                            time_analyze, 
                                                                            distance_threshold, 
                                                                            "single")

        X = np.array(coordinate_data) # 各点の座標データ np.array
        
        
        # 位置とクラスタ番号の描画
        fig, ax = plt.subplots(figsize=(6,6))
        plt.scatter(X[:,0], X[:,1], c=model.labels_) #cmap="viridis")



        clusters_size = np.bincount(labels) # <-- ラベル別のクラスタサイズ

        n = sum(x>1 for x in clusters_size) # <--クラスタサイズが1より大きいものの数

        # print(clusters_size)


        # ラベルのプロット
        for j in range(60):
            if clusters_size[labels[j]] > 1:
                plt.annotate(labels[j], (X[j][0], X[j][1]))
        
        # =====クラスタの中心座標と半径の描画=====
        cluster_centers, list_of_radius = calculate_cluster_centers(labels, X, clusters_size) 
        
        # 中心座標の描画
        cnt = np.array(cluster_centers)
        plt.scatter(cnt[:,0], cnt[:,1], s=100, marker="x", color="red")
        
        # 半径の描画
        maxradius = max(list_of_radius)
        for center, r in zip(cluster_centers, list_of_radius):
            if r == maxradius:
                color = "red"
                c = patches.Circle(center, r, alpha=0.2,facecolor=color, edgecolor='black')
                ax.add_patch(c)
            else:
                c = patches.Circle(center, r, alpha=0.2,facecolor='blue', edgecolor='black')
                ax.add_patch(c)
    
        
        
        plt.title("distance based (%s) \n distance_threshold = %d, n_clusters=%d (all=%d) \n maxradius=%1.2f" %(method, distance_threshold, n, n_clusters, maxradius))

        plt.xlim(0, 430)
        plt.ylim(0, 430)
        plt.show()
        
if __name__ == "__main__":
    data = pd.read_csv("all_result.csv", header=None)
    plot_coordinate(data, 33, 35, 10000, "single")