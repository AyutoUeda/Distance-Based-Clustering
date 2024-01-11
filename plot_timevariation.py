"""
【時間変化を出力するためのプログラム】

1. time_variation.csvを読み込む
2. 各列をプロットする

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("time_variation.csv")

df_ = df.values

print(df_)


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


selected_time = [s/10 for s in range(0, len(df_))]

# ===== the number of clusters  =====
plt.figure(figsize=(18, 5))
plt.plot(selected_time, df_[:,0])
plt.title("The Number of Clusters \n distance threshold = 35")
plt.xlabel("Time (s)")
plt.xlim(0, len(df_)/10)
# plt.show()
plt.savefig("outputs/time_variation/number_of_clusters.png", dpi=300)

# ===== the maximum of cluster size =====
plt.figure(figsize=(18, 5))
plt.plot(selected_time, df_[:,1])
plt.title("The Maximum Cluster Size \n distance threshold = 35")
plt.xlabel("Time (s)")
plt.xlim(0, len(df_)/10)
plt.ylim(2,22)
plt.yticks(np.arange(2, 22, 1))
# plt.show()
plt.savefig("outputs/time_variation/maximum_cluster_size.png", dpi=300)

# ===== the maximum radius =====
plt.figure(figsize=(18, 5))
plt.plot(selected_time, df_[:,3])
plt.title("The Maximum Radius \n distance threshold = 35")
plt.xlabel("Time (s)")
plt.ylabel("Radius (pixel)")
plt.xlim(0, len(df_)/10)
# plt.show()
plt.savefig("outputs/time_variation/maximum_radius.png", dpi=300)
