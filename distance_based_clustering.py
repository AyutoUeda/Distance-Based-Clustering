from modules import hierachical_clustering, calculate_cluster_centers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_coordinate(data, lower_threshold, upper_threshold, time_analyze, method="single", save_fig=False): 
    """指定された行数（時間）における座標データとクラスタをプロットする関数
    
    Args:
        data (```dataframe```): 座標データ
        lower_threshold (```int```): 閾値の下限
        upper_threshold (```int```): 閾値の上限
        time_analyze (```int```): 分析する秒数（行数） 
        method (```str```): 距離の測定方法(default="single")
        save_fig (```bool```): 図を保存するかどうか(default=False)
    
    """
    # 図の書式設定
    plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
    plt.rcParams['xtick.labelsize'] = 9 # 軸だけ変更されます。
    plt.rcParams['ytick.labelsize'] = 9 # 軸だけ変更されます
    plt.rcParams['xtick.direction'] = 'in' # x axis in
    plt.rcParams['ytick.direction'] = 'in' # y axis in 
    plt.rcParams['axes.linewidth'] = 1.0 # axis line width
    plt.rcParams['axes.grid'] = True # make grid
    plt.rcParams["legend.fancybox"] = False # 丸角
    plt.rcParams["legend.framealpha"] = 1 # 透明度の指定、0で塗りつぶしなし
    plt.rcParams["legend.edgecolor"] = 'black' # edgeの色を変更
    plt.rcParams["legend.handlelength"] = 1 # 凡例の線の長さを調節
    plt.rcParams["legend.labelspacing"] = 5. # 垂直方向の距離の各凡例の距離
    plt.rcParams["legend.handletextpad"] = 3. # 凡例の線と文字の距離の長さ
    plt.rcParams["legend.markerscale"] = 2 # 点がある場合のmarker scale
    plt.rcParams["legend.borderaxespad"] = 0. # 凡例の端とグラフの端を合わせる
    

    for distance_threshold in range(lower_threshold, upper_threshold+1):
        
        n_clusters, labels, coordinate_data, model = hierachical_clustering(data, 
                                                                            time_analyze, 
                                                                            distance_threshold, 
                                                                            "single")

        X = np.array(coordinate_data) # 各点の座標データ np.array
        
        
        # 位置とクラスタ番号の描画
        fig, ax = plt.subplots(figsize=(6,6))
        plt.scatter(X[:,0], X[:,1], c="steelblue") #cmap="viridis")



        clusters_size = np.bincount(labels) # <-- ラベル別のクラスタサイズ

        n_withoutsize1 = sum(x>1 for x in clusters_size) # <--クラスタサイズが1より大きいものの数

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
    
        
        
        plt.title("Hierarchical Clustering (%s) \n distance_threshold = %d, n_clusters=%d (all=%d) \n maxradius=%1.2f" 
                  %(method, distance_threshold, n_withoutsize1, n_clusters, maxradius))

        plt.xlim(0, 430)
        plt.ylim(0, 430)
        # plt.show()
        
        fig_path = "HC_{}s_threshold{}.png".format(time_analyze, distance_threshold)
        
        if save_fig:
            fig.savefig(fig_path, bbox_inches="tight", pad_inches=0.05)
        
if __name__ == "__main__":
    data = pd.read_csv("all_result.csv", header=None)
    plot_coordinate(data, 33, 35, 10000, "single", save_fig=False)