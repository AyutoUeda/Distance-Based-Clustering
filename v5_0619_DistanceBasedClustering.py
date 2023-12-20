# ## 0619 やること  
# 1. 最大半径の時間変化をフーリエ変換  
# 2. 中心位置のコントロール  
# 3. 計算時間の拡大 -> 0秒目から


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

# データ準備
df = pd.read_csv("all_result.csv")

from IPython.core.display import display, HTML
display(HTML("<style>div.output_scroll { height: unset; }</style>"))


# %% [markdown] slideshow={"slide_type": "slide"} toc-hr-collapsed=true
# # 1.関数作成  
# ## 1.1 階層クラスタリング
# 分析する行と距離の閾値を設定→モデルとラベルを返す

# %% slideshow={"slide_type": "slide"}
def hierachical_clustering(method="single", time, threshold):
    """階層クラスタリングを実行する関数
    
    Args:
        method (str): 距離の測定方法(default="single")  
        time (int): 分析する秒数（行数） 
        threshold (int): クラスタ間の距離の閾値  
        
    Returns:
        model.n_clusters_ (int): クラスタ数
        model.labels_ (list): 各点のラベル
        coordinate_data (list): 指定した時間（行数）における各点の座標
        model (model): モデル
        
    """
    coordinate_data = []
    
    # 何秒目をクラスタリングするか
    for i in range(0, 120, 2):
      x = df.iloc[time][i]
      y = df.iloc[time][i+1]
      coordinate_data.append([int(x), int(y)])

    # list --> array
    X = np.array(coordinate_data)

    model = AgglomerativeClustering(
        n_clusters=None, linkage=str, compute_full_tree=True, distance_threshold=threshold
    )

    model.fit(X)
    
    return model.n_clusters_, model.labels_, coordinate_data, model


# ## 1.2 クラスタサイズが1のものを除く関数

def exception_size1(labels):
    labels = hierachical_clustering("single",10000,35)[1] # 各点のラベル
    print(labels)

    clusters_size = np.bincount(labels) # <-- ラベル別のクラスタサイズ
    print(clusters_size, "<--ラベル別のクラスタサイズ")

    n = sum(x>1 for x in clusters_size) 
    print(n, "<--クラスタサイズが1より大きいものの数")
    
    return clusters_size, n

labels = hierachical_clustering("single",10000, 35)[1]
exception_size1(labels)

# %% [markdown]
# **cluster_sizeの見方**  
# [9 2 6 2 1 5 2 1 1 1 1 3 4 1 4 1 1 1 1 1 1 1 1 1 1 1 2 1 1 2]  
# →クラスタナンバー0に分類されているものが9個、クラスタナンバー1に分類されているものが2個....

# %% [markdown]
# ## 1.3 クラスタ半径を求める関数

# %% [markdown]
# **半径を求めるために使用した式(クラスタの一番端を採用)**  →  
# $$
#     \textrm{中心座標　} \ \vec{r_g}= (x_g, y_g)\\
#     \textrm{点の座標　} \ \vec{r_p}= (x_p, y_p)\\
#     R=\sqrt{(x_p - x_g)^2+(y_p - y_g)^2}
# $$

# %% [markdown]
# $$
#     \textrm{中心座標　} \ \vec{r_g}= (x_g, y_g)\\
#     \textrm{クラスタ半径　} \ \vec{R} = \sqrt{\sum_{i=1}^{n}(\vec{r_i}-\vec{r_g})^2} 
# $$

# %% [markdown]
# $$
#     \sqrt{\sum_{i=1}^{n}((x_i-a)^2 + (y_i-a)^2)}
# $$

# %%
import math

def calculate_radius(arg1, arg2):
    '''
    引数
    arg1 --> 各点の座標リスト [[x1,y1], [x2,y2], ....]
    arg2 --> 中心座標 [x, y]
    '''
    points = np.array(arg1)
    center = np.array(arg2)

    R = 0

    for point in points:
        a = np.abs(point-center) # ベクトルの差
        r = math.sqrt(a[0]**2 + a[1]**2)
        if r > R:
            R = r
    return R


# %% [markdown]
# ## 1.4 中心座標を求める関数

# %% [markdown]
# ### 1.4.1中心座標を計算

# %%
def calculate_center(arg1):
    points = arg1
    x_sum = 0
    y_sum = 0

    for point in points:
        x_sum += point[0]
        y_sum += point[1]

    center_x = x_sum / len(points)
    center_y = y_sum / len(points)
    center_coordinate = [round(center_x, 2), round(center_y, 2)]

    return center_coordinate


# %% [markdown]
# **引数**には座標リスト [[x,y], [x,y], ...]  
# **戻り値**は中心座標 [xxx, yyy]

# %% [markdown]
# ### 1.4.2 クラスタごとに中心座標とその半径を算出する関数 calculate_cluster_centers

# %%
# test data
labels = hierachical_clustering("single",10000,35)[1] # 各点のラベル
coordinate = hierachical_clustering("single",10000,35)[2] # 各点の座標
c_size = [9, 2, 6, 2, 1, 5, 2, 1, 1, 1, 1, 3, 4, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2]

def calculate_cluster_centers(labels, coordinate, c_size):
    centers = []
    list_of_radius = []
    for i, size in enumerate(c_size): # i --> ラベル番号
        if size > 1:
            points = []
            for j, label in enumerate(labels):
                if i == label:
                    # print(labels[j], tstcoordinate[j])
                    points.append(coordinate[j])

            center = calculate_center(points) # 中心座標を計算
            # print(points, "\n", center)
            # print("--------------------------------")
            centers.append(center)
            
            R = calculate_radius(points, center)
            list_of_radius.append(R)
            
            points = []

    return centers, list_of_radius


# %% [markdown]
# **引数**  
# labels --> 各点のクラスタラベル（要素60個）  
# coordinate --> 各点の座標データ (要素60)  
# c_size --> ラベルごとのクラスタサイズ　([9,3,2,1...] <-- クラスタラベル1のサイズは9、クラスタラベル2のサイズは2...)

# %%
ccc=np.array(coordinate)
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
plt.scatter(ccc[:,0], ccc[:,1])
plt.axis('square')
plt.xlim(0,430)
plt.ylim(0,430)

# %%
# 位置とクラスタ番号の描画
fig = plt.figure(figsize=(6,6))

tstX = np.array(coordinate)
plt.scatter(tstX[:,0], tstX[:,1], c=labels) #cmap="viridis")

clusters_size = np.bincount(labels) # <-- ラベル別のクラスタサイズ

n = sum(x>1 for x in clusters_size) # <--クラスタサイズが1より大きいものの数

print(clusters_size)
print(n)

# ラベルのプロット
for i in range(60):
    # plt.annotate(labels[i], (tstX[i][0], tstX[i][1]))

    if clusters_size[labels[i]] > 1:
        plt.annotate(labels[i], (tstX[i][0], tstX[i][1]))

cnt = np.array(calculate_cluster_centers(labels, coordinate, c_size)[0])
plt.scatter(cnt[:,0], cnt[:,1], s=100, marker="x", color="red")

plt.show()

# %% [markdown]
# # ■実験結果

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Test(plot_coordinate)

# %% [markdown]
# 時間→固定  
# 閾値→変動  
# 閾値別のクラスタとその中心、半径をプロット  
# 半径が最大のものを赤く表示

# %% slideshow={"slide_type": "slide"}
import matplotlib.patches as patches

def plot_coordinate(str, num1, num2, num3): 
    '''
    [引数]
    str --> 距離の算出方法（single, ward ...)
    num1 --> 閾値の下限
    num2 --> 閾値の上限
    num3 --> 何行目をクラスタリングするか
    '''
    for i in range(num1, num2+1):
        distance_threshold = i
        model = hierachical_clustering(str,num3, i)[3]
        
        nclusters = model.n_clusters_ # クラスタサイズ1を含めた全クラスタ数

        X = np.array(hierachical_clustering(str,num3, i)[2]) # 各点の座標データ
        
        
        # 位置とクラスタ番号の描画
        fig, ax = plt.subplots(figsize=(6,6))
        plt.scatter(X[:,0], X[:,1], c=model.labels_) #cmap="viridis")

        labels = model.labels_

        clusters_size = np.bincount(labels) # <-- ラベル別のクラスタサイズ

        n = sum(x>1 for x in clusters_size) # <--クラスタサイズが1より大きいものの数

        # print(clusters_size)


        # ラベルのプロット
        for j in range(60):
            if clusters_size[labels[j]] > 1:
                plt.annotate(labels[j], (X[j][0], X[j][1]))
        
        # クラスタの中心座標
        centers = calculate_cluster_centers(labels, X, clusters_size)[0]    
        cnt = np.array(centers)
        plt.scatter(cnt[:,0], cnt[:,1], s=100, marker="x", color="red")
        
        # クラスタ半径
        radius = calculate_cluster_centers(labels, X, clusters_size)[1]
        maxradius = max(radius)
        for center, r in zip(centers, radius):
            if r == maxradius:
                color = "red"
                c = patches.Circle(center, r, alpha=0.2,facecolor=color, edgecolor='black')
                ax.add_patch(c)
            else:
                c = patches.Circle(center, r, alpha=0.2,facecolor='blue', edgecolor='black')
                ax.add_patch(c)
    
        
        
        plt.title("distance based (%s) \n distance_threshold = %d, n_clusters=%d (all=%d) \n maxradius=%1.2f" %(str, distance_threshold, n, nclusters, maxradius))

        plt.xlim(0, 430)
        plt.ylim(0, 430)
        plt.show()


# %% [markdown]
# plot_coordinate --> クラスタサイズが1のものを除く、クラスタの中心をプロットする、クラスタ半径をプロットする

# %%
# plot_coordinate("single", 30, 35, 70000)

# %% [markdown]
# ## 時間ごとのクラスタと半径プロット(plot_cootdinate_by_time)

# %%
def plot_coordinate_by_time(str, num1, num2, num3):
    '''
    [引数]
    str --> 距離の算出方法（single, ward ...)
    num1 --> 閾値
    num2 --> スタートタイム
    num3 --> エンドタイム
    '''
    distance_threshold = num1
    
    for i in range(num2, num3+1):
        _, _, coordinatedata, model = hierachical_clustering(str, i, distance_threshold)
        
        nclusters = model.n_clusters_ # クラスタサイズ1を含めた全クラスタ数

        # X = np.array(hierachical_clustering(str, i, distance_threshold)[2]) # 各点の座標データ
        
        X = np.array(coordinatedata)
        
        # 位置とクラスタ番号の描画
        fig, ax = plt.subplots(figsize=(6,6))
        plt.scatter(X[:,0], X[:,1], c=model.labels_) #cmap="viridis")

        labels = model.labels_

        clusters_size = np.bincount(labels) # <-- ラベル別のクラスタサイズ

        n = sum(x>1 for x in clusters_size) # <--クラスタサイズが1より大きいものの数

        # print(clusters_size)


        # ラベルのプロット
        for j in range(60):
            if clusters_size[labels[j]] > 1:
                plt.annotate(labels[j], (X[j][0], X[j][1]))
        
        # クラスタの中心座標
        centers = calculate_cluster_centers(labels, X, clusters_size)[0]    
        cnt = np.array(centers)
        plt.scatter(cnt[:,0], cnt[:,1], s=100, marker="x", color="red")
        
        # クラスタ半径
        radius = calculate_cluster_centers(labels, X, clusters_size)[1]
        maxradius = max(radius)
        for center, r in zip(centers, radius):
            if r == maxradius:
                color = "red"
                c = patches.Circle(center, r, alpha=0.2,facecolor=color, edgecolor='black')
                ax.add_patch(c)
            else:
                c = patches.Circle(center, r, alpha=0.2,facecolor='blue', edgecolor='black')
                ax.add_patch(c)
    
        
        
        plt.title("distance based (time = %d (s)) \n distance_threshold = %d, n_clusters=%d (all=%d) \n maxradius=%1.2f" 
                  %(i/10, distance_threshold, n, nclusters, maxradius))

        plt.xlim(0, 430)
        plt.ylim(0, 430)
        # plt.savefig("outputs/hierarchical_clustering/hierarchical_time_{}.png".format(int(i/10)), bbox_inches='tight')
        plt.show()


# %% [markdown]
# **1000秒目から12000秒まで1000秒おきにクラスタと中心をプロット**  
# -　時間の経過とともにクラスタが端による  
# -　時間の経過とともにクラスタ半径が小さくなる

# %%
# 1000秒目から12000秒目まで（1000秒
for i in range(10000, 130000, 10000):
    plot_coordinate_by_time("single", 35, i, i)

# %% [markdown]
# ## 半径の最大値の時間変化

# %% [markdown]
# クラスタの最大半径とその時のクラスタサイズを描画

# %% [markdown]
# ### 最大半径を持つクラスタの中心（左）と最小半径を持つクラスタの中心（中）とクラスタに含まれない粒子（右）をプロット

# %% [markdown]
# - 半径が最大のものほど端による

# %%
# %%time

time_start = 0
time_end = 120000
span = 10000

list_of_max_radius = []
list_of_min_radius = []
list_of_max_clustersize = []
list_of_min_clustersize = []

for i in range(time_start, time_end, span):
    y_max_radius = [] # 最大半径
    y_max_clustersize = [] # 最大半径を持つクラスタのサイズ
    coordinate_of_max_center = []
    
    y_min_radius = [] # 最小半径
    y_min_clustersize = [] # 最小半径を持つクラスタのサイズ
    coordinate_of_min_center = []
    
    coordinate_of_singles = []
    
    for j in range(i, i+span):
        distance_threshold = 35
        model = hierachical_clustering("single", j, distance_threshold)[3]

        nclusters = model.n_clusters_ # クラスタサイズ1を含めた全クラスタ数

        X = np.array(hierachical_clustering("single", j, distance_threshold)[2]) # 各点の座標データ

        labels = model.labels_
        
        clusters_size = np.bincount(labels) # <-- ラベル別のクラスタサイズ
        
        # -----サイズ1の粒子の位置-----
        labels_of_clustersizeis1 = []
        for k, s in enumerate(clusters_size):
            if s == 1:
                labels_of_clustersizeis1.append(k)

        index_of_clustersizeis1 = []
        for k in labels_of_clustersizeis1:
            index_of_clustersizeis1.append(labels.tolist().index(k))

        coordinate_of_size1 = []
        for k in index_of_clustersizeis1:
            coordinate_of_size1.append(X.tolist()[k])

        coordinate_of_singles += coordinate_of_size1
        
        
        # -----クラスタの中心座標-----
        centers = calculate_cluster_centers(labels, X, clusters_size)[0]    
        cnt = np.array(centers)

        # クラスタ半径
        radius = calculate_cluster_centers(labels, X, clusters_size)[1]
        
        # クラスタサイズが1以上
        clusters_size_without_size1 = clusters_size[clusters_size != 1]
        
        
        # -----max-----
        maxradius = max(radius)
        max_index = radius.index(maxradius)
        
        y_max_radius.append(maxradius)
        y_max_clustersize.append(clusters_size_without_size1[max_index])

        coordinate_of_max_center.append(centers[max_index])
        
        # -----min-----
        minradius = min(radius)
        min_index = radius.index(minradius)
        
        y_min_radius.append(minradius)
        y_min_clustersize.append(clusters_size_without_size1[min_index])

        coordinate_of_min_center.append(centers[min_index])
    
    list_of_max_radius += y_max_radius
    list_of_min_radius += y_min_radius
    
    list_of_max_clustersize += y_max_clustersize
    list_of_min_clustersize += y_min_clustersize
    
    
    fig = plt.figure(figsize=(18, 6))
    # plot max centers
    ax1 = fig.add_subplot(1,3,1)
    values = np.array(coordinate_of_max_center)
    ax1.scatter(values[:, 0],values[:, 1], marker="x", alpha=0.5)
    ax1.set_title("Location of the cluster center with the maximum radius \n time=%d - %d (s)" %(i/10, (i+span)/10))
    ax1.set_xlim(0, 430)
    ax1.set_ylim(0, 430)
    
    # plot min centers
    ax2 = fig.add_subplot(1,3,2)
    min_values = np.array(coordinate_of_min_center)
    ax2.scatter(min_values[:, 0],min_values[:, 1], marker="x", color="red", alpha=0.5)
    ax2.set_title("Location of the cluster center with the minimum radius \n time=%d - %d (s)" %(i/10, (i+span)/10))
    ax2.set_xlim(0, 430)
    ax2.set_ylim(0, 430)
    
    # plot singles
    ax3 = fig.add_subplot(1,3,3)
    single_values = np.array(coordinate_of_singles)
    ax3.scatter(single_values[:, 0],single_values[:, 1], marker=".", color="black", alpha=0.5)
    ax3.set_title("Location of singles \n time=%d - %d (s)" %(i/10, (i+span)/10))
    ax3.set_xlim(0, 430)
    ax3.set_ylim(0, 430)

# %% [markdown]
# test of plotting singles

# %%
model = hierachical_clustering("single", 10000, 35)[3]

nclusters = model.n_clusters_ # クラスタサイズ1を含めた全クラスタ数

X = np.array(hierachical_clustering("single", 10000, 35)[2]) # 各点の座標データ
X = X.tolist()
labels = model.labels_.tolist()

clusters_size = np.bincount(labels) # <-- ラベル別のクラスタサイズ

labels_of_clustersizeis1 = []
for i, s in enumerate(clusters_size):
    if s == 1:
        labels_of_clustersizeis1.append(i)

index_of_clustersizeis1 = []
for i in labels_of_clustersizeis1:
    index_of_clustersizeis1.append(labels.index(i))

coordinate_of_size1 = []
for i in index_of_clustersizeis1:
    coordinate_of_size1.append(X[i])

print(coordinate_of_size1)

test_values = np.array(coordinate_of_size1)
plt.figure(figsize=(6,6))
min_values = np.array(coordinate_of_min_center)
plt.scatter(test_values[:, 0],test_values[:, 1], marker="x", color="red", alpha=0.5)
plt.xlim(0, 430)
plt.ylim(0, 430)

# %% [markdown]
# ### 最大半径の時間変化とその時のクラスタサイズ

# %% [markdown]
# - 3000〜4000秒あたりから半径が安定し始める
# - クラスタサイズも半径に一致する形で変化

# %%
#グラフを表示する領域を，figオブジェクトとして作成。
fig = plt.figure(figsize = (25,9))

#グラフを描画するsubplot領域を作成。
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

# selected time
x = [s/10 for s in range(time_start, time_end)]

ax1.plot(np.array(x), np.array(list_of_max_radius))
ax2.plot(np.array(x), np.array(list_of_max_clustersize))

ax1.set_ylabel("maximum value of radius")
ax2.set_ylabel("clustersize")
ax1.set_xlabel("time (s)")
ax2.set_xlabel("time (s)")

ax1.set_xlim(time_start/10, time_end/10)
ax2.set_xlim(time_start/10, time_end/10)

# ax1.set_xlim(time_start/10, time_end/10)
# ax2.set_xlim(time_start/10, time_end/10)

# ax1.set_xticks(np.arange(time_start,time_end,1000))

plt.suptitle("the time variation of maximum value of radius and clustersize \n distance threthold=35", fontsize=20)
plt.show()

# %% [markdown]
# #### 最大半径のフーリエ変換

# %%
# 実数部
fk = np.fft.fft(list_of_max_radius-np.mean(list_of_max_radius))
plt.plot((fk.real)**2+(fk.imag)**2)
plt.xlim(0,300)
plt.ylim(0, 1.5e10)

# %% [markdown] slideshow={"slide_type": "slide"}
# ## クラスタ数と距離（クラスタサイズが1より大きいもののみ）

# %% [markdown]
# 距離の閾値とクラスタの関係性

# %% slideshow={"slide_type": "slide"}
'''
・クラスタ間の距離を変化させる
・処理する時間は固定
'''
time = 20000
clusters1 = [] # クラスタ数 （クラスタサイズ１を除く）
clusters_including1 = []
dis = [s for s in np.arange(30, 51, 0.5)]
for i in np.arange(30, 51, 0.5):
    nclusters, labels = hierachical_clustering("single",time,i)[0], hierachical_clustering("single",time,i)[1] # 各点のラベル

    clusters_size = np.bincount(labels) # ラベル別のクラスタサイズ

    n = sum(x>1 for x in clusters_size) # クラスタサイズが1より大きいもののみカウント
    clusters1.append(n)
    
    clusters_including1.append(nclusters)

print(clusters1)


X = dis
Y = clusters1
plt.figure(figsize=(6,6))
plt.plot(X, Y, label="number of clusters")
plt.plot(X, clusters_including1, "--",label="clusters including size 1")
plt.xlabel("distance_threshold")
plt.ylabel("n_clusters")
plt.xlim(30,50)
plt.title("Relationship between distance_threshold (0.5steps) and n_clusters \n time=%d" %time)
plt.legend()
plt.show()

# %% [markdown]
# ある一点におけるクラスタ数と距離の閾値の関係（クラスタサイズが1のものを除く）

# %% [markdown] slideshow={"slide_type": "slide"}
# ## 時間ごとのクラスタ数の変化(クラスタサイズが1より大きいもの）

# %% [markdown]
# - 4000秒辺りから安定
# - 時間の経過とともにクラスタ数は減少

# %% slideshow={"slide_type": "slide"}
# %%time
'''
・時間ごとのクラスタ数の表示
'''
start = 0
end = 120000
clusters2 = []
selected_time = [s for s in range(start, end)]

# for i in range(30, 36):
#     c = []
#     for j in range(start, end):
#         labels = hierachical_clustering("single",j,i)[1] # 各点のラベル

#         clusters_size = np.bincount(labels) # ラベル別のクラスタサイズ

#         n = sum(x>1 for x in clusters_size) # クラスタサイズが1より大きいもののみカウント
#         c.append(n)
#     clusters2.append(c)

for i in range(start, end):
    labels = hierachical_clustering("single",i,35)[1] # 各点のラベル

    clusters_size = np.bincount(labels) # ラベル別のクラスタサイズ

    n = sum(x>1 for x in clusters_size) # クラスタサイズが1より大きいもののみカウント
    clusters2.append(n)

# %% [markdown]
# **memo**  
# 50000行→6min

# %%
# distance = [s for s in range(30, 36)]
# for i in range(len(distance)):
#     mean = np.mean(clusters2[i]) # クラスタサイズの平均値

#     plt.figure(figsize=(15,4))
#     plt.plot(selected_time, clusters2[i])
#     plt.plot([start, end],[mean, mean], "red", linestyle='dashed', label="mean") # 平均値を示す補助線
#     plt.xlabel("time")
#     plt.ylabel("clusters")
#     plt.ylim((5,23))
#     plt.xlim((start, end))
#     plt.title("Relationship between time and n_clusters \n distance_threshold=%d \n min=%d , max=%d , mean=%1.2f" 
#               %(distance[i], min(clusters2[i]), max(clusters2[i]), mean))
#     plt.legend()
#     plt.show()

selected_time = [s/10 for s in range(start, end)]  # 単位を行から秒に修正

mean = np.mean(clusters2)
plt.figure(figsize=(20,6))
plt.plot(selected_time, clusters2)
plt.plot([start, end],[mean, mean], "red", linestyle='dashed', label="mean") # 平均値を示す補助線
plt.xlabel("time (s)")
plt.ylabel("clusters")
plt.ylim((5,23))
plt.xlim((start/10, end/10))
plt.title("Relationship between time and n_clusters \n distance_threshold=35 \n min=%d , max=%d , mean=%1.2f" 
          %(min(clusters2), max(clusters2), mean))
plt.legend()
plt.show()

# %% [markdown]
# #### クラスタ数の変化のフーリエ変換

# %%
fk1 = np.fft.fft(clusters2-np.mean(clusters2))
plt.plot((fk1.real)**2+(fk1.imag)**2)
plt.xlim(0,100)

# %% [markdown]
# ## クラスタサイズの時間変化（最大値）

# %%
'''
・時間ごとのクラスタ数の表示
'''
start1 = 0
end1 = 120000
list_of_maxclustersize = [] # クラスタサイズの最大値
list_of_minclustersize = [] # クラスタサイズの最小値


# for i in range(30, 36):
#     c_max = []
#     c_min = []
for j in range(start1, end1):
    labels = hierachical_clustering("single",j,35)[1] # 各点のラベル

    clusters_size = np.bincount(labels) # ラベル別のクラスタサイズ
    except_size1 = [x for x in clusters_size if x > 1]

    maxnum = max(except_size1) # クラスタサイズが最大のものを抽出
    list_of_maxclustersize.append(maxnum)

    minimum = min(except_size1)
    list_of_minclustersize.append(minimum)

# %%
# 描画
# distance = [s for s in range(30, 35)]
# for i in range(len(distance)):

selected_time1 = [s/10 for s in range(start1, end1)]
mean = np.mean(list_of_maxclustersize) # 最大クラスタサイズの平均値

# plt.fistart_pointigsize=(15,4)
plt.figure(figsize=(20, 6))
plt.plot(selected_time1, list_of_maxclustersize, label="maximum")
plt.plot(selected_time1, list_of_minclustersize, c="black", label="minimum")
plt.plot([start1, end1],[mean, mean], "red", linestyle='dashed', label="mean of maximum") # 平均値を示す補助線
plt.xlabel("time")
plt.ylabel("cluster size")
plt.ylim((0,20))
plt.xlim((start1, end1/10))
plt.title("Relationship between time, maximum clustersize and minimum clustersize \n distance_threshold=35 \n mean of maximum=%1.2f" 
          # %mean)
plt.legend()
plt.show()

# %% [markdown]
# ### 最大クラスタサイズをフーリエ変換

# %%
fk_maxclustersize = np.fft.fft(list_of_maxclustersize-np.mean(list_of_maxclustersize))
plt.plot((fk_maxclustersize.real)**2+(fk_maxclustersize.imag)**2)
plt.xlim(0,100)

# %% [markdown]
# ### クラスタサイズ別時間変化  

# %%
start_point = 0
end_point = 120000
sizecount = []
for i in range(start_point, end_point+1):
    labels = hierachical_clustering("single",i,35)[1] # 各点のラベル

    clusters_size = np.bincount(labels) # ラベル別のクラスタサイズ
    except_size1 = [x for x in clusters_size if x > 1]
    
    count_clustersize = np.bincount(except_size1) # クラスタサイズが同一のものをカウント
    if len(count_clustersize) < 10:
        for j in range(10 - len(count_clustersize)):
            count_clustersize = np.append(count_clustersize,0)
    sizecount.append(count_clustersize) 

# %%
plt.figure(figsize=(20,20))
plt.subplots_adjust(hspace=1)
for i in range(2, 10):
    plt.subplot(8,1,i-1)
    plt.plot([x/10 for x in range(start_point, end_point+1)],[s[i] for s in sizecount])
    plt.ylim(0,15)
    plt.title("clustersize = %d" %i)
    plt.ylabel("the number of clusters")
    plt.xlabel("time")

# %%
# %%time
maxclustersize = 20
start_time = 0
end_time = 120000
span = 10000
for i in range(start_time, end_time, span):
    start = i
    end = i + span
    
    count_by_clustersize = np.zeros(shape=(maxclustersize), dtype=int)
    for j in range(start, end):
        labels = hierachical_clustering("single",j,35)[1] # 各点のラベル

        clusters_size = np.bincount(labels) # ラベル別のクラスタサイズ
        clusters_size_without_size1 = clusters_size[clusters_size != 1]

        count = np.bincount(clusters_size_without_size1)
        # -----0で足りない要素分を埋め合わせ(array)-----
        if len(count) < maxclustersize:
            for j in range(maxclustersize-len(count)):
                count = np.append(count, 0)

        # -----サイズ別に個数のカウント-----
        count_by_clustersize = np.add(count_by_clustersize, count)
        # count_by_clustersize += count

    # print(count_by_clustersize)
    
    # -----plot-----
    plt.figure(figsize=(16,6))
    plt.bar([x for x in range(2, maxclustersize)],count_by_clustersize[2:])
    plt.title("Count clusters by clustersize \n time=%d - %d (s)" %(start / 10, end/10))
    plt.xlabel("clustersize")
    plt.ylim(0, 80000)
    plt.xticks([x for x in range(2, maxclustersize)])

# %% [markdown]
# ## アニメーション

# %%
# %%time
from matplotlib.animation import FuncAnimation, ArtistAnimation
from IPython.display import HTML
import pathlib
import matplotlib

matplotlib.rcParams['animation.embed_limit'] = 2**128

# Initialize scatter plot
fig, ax = plt.subplots(figsize=(6,6))
scatter = ax.scatter([], [])
center = ax.scatter([], [])
ax.set_title("Hierachical Clustering (single) \n distance threshold=35")
# Set plot properties
ax.set_xlim(0, 450)
ax.set_ylim(0, 450)
ax.set_xlabel('X')
ax.set_ylabel('Y')

timestart = 108000
# -----Define function to update scatter position-----
def update_scatter(i):
    # どこからスタートさせるか
    i = i+1+timestart
    ax.cla()
    ax.set_xlim(0, 430)
    ax.set_ylim(0, 430)
    model = hierachical_clustering("single", i, 35)[3]
    labels, data = hierachical_clustering("single",i,35)[1], hierachical_clustering("single", i,35)[2]
    x, y = zip(*data)

    scatter = ax.scatter(x[:], y[:], color="black")
    
    clusters_size = np.bincount(labels) # <-- ラベル別のクラスタサイズ
    
    # クラスタの中心座標
    centers = calculate_cluster_centers(labels, data, clusters_size)[0]    
    cnt = np.array(centers)
    center = ax.scatter(cnt[:,0], cnt[:,1], s=100, marker="x", color="red")
    
    # クラスタ半径
    radius = calculate_cluster_centers(labels, data, clusters_size)[1]
    maxradius = max(radius)
    for center, r in zip(centers, radius):
        if r == maxradius:
            color = "red"
            c = patches.Circle(center, r, alpha=0.2,facecolor=color, edgecolor='black')
            ax.add_patch(c)
        else:
            c = patches.Circle(center, r, alpha=0.2,facecolor='blue', edgecolor='black')
            ax.add_patch(c)
    
    text = ax.text(20, 400, "%1.1f (s)" %(i/10))
    return scatter, center, text

# Animate scatter plot
# フレーム数を変化させることで表示範囲を変更
ani = FuncAnimation(fig, update_scatter, frames=18000, interval=100)
# HTML(ani.to_jshtml())
# ani.save("outputs/animation_of_droplets_from%d.mp4" %(timestart/10))

# %% [markdown]
# ## 樹形図

# %%
# 樹形図　描画用関数
def plot_dendrogram(model, **kwargs):
   # Create linkage matrix and then plot the dendrogram

   # create the counts of samples under each node
   counts = np.zeros(model.children_.shape[0])
   n_samples = len(model.labels_)
   for i, merge in enumerate(model.children_):
      current_count = 0
      for child_idx in merge:
         if child_idx < n_samples:
            current_count += 1 # leaf node
         else:
            current_count += counts[child_idx - n_samples]
      counts[i] = current_count

   linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

   # Plot the corresponding dendrogram
   dendrogram(linkage_matrix, leaf_font_size=12, **kwargs)


# %%
for i in range(0, 130000, 20000):
    model = hierachical_clustering("single", i, 35)[3]
    # 樹形図の描画
    fig = plt.figure(figsize=(20,10), facecolor="white")
    ax = fig.add_subplot(111)
    
    ax.set_title('Dendrogram of Hierarchical Clustering \n Euclidean, linkage=single \n distance threshold=35, time=%d (s)' %(i/10))
    # plot the top three levels of the dendrogram
    plot_dendrogram(model, truncate_mode='level', p=10)
    ax.set_xlabel("Number of points in node (or index of point if no parenthesis).")
    ax.set_ylabel("Distance")
    

    ax.plot([0, 1000],[35, 35], "red", linestyle='dashed', label="distance threshold") 
    ax.legend()
    # fig.savefig("outputs/dendrogram/dendrogram_at_%d.png" %(i/10), transparent=False)
    plt.show()

# %%
