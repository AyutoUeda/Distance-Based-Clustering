# Hierarchical Clustering

## modules.py

階層クラスタリングに使用する基本的な関数群

|name of function | function|
| --------------- | ------- |
| hierarchical_clustering | 指定された行数（時間）における座標データとクラスタを計算|
| exception_size1  | ラスタサイズが1のものを除いた際のクラスタ数を返す関数|
| calculate_radius | 半径を計算する関数|
| calculate_center | クラスタの中心座標を計算する関数|
| calculate_cluster_centers | ```calculate_radius```と```calculate_centers```を使いクラスタの中心と半径をすべてのクラスタに対して計算|

**Example**
  
```modules.py
# データフレームを入力とする
data = pd.DataFrame([[1, 1, 2, 2, 3, 3, 4, 4, 5, 5], 
                     [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]])

# ある時間における「クラスタ数」、「そのクラスタのラベル」、「座標」を返す
n_clusters, labels, coordinate_data, model = hierarchical_clustering(data, time_analyze=0, threshold=2, "single")

# 上で求めたラベルと座標をもとにクラスタの中心を計算
centers, list_of_radius = calculate_cluster_centers(labels, coordinate_data, np.bincount(labels))

```

## distance_based_clustering.py
modules.pyの関数を用いてデータの各時間におけるクラスタリングを実行。  
その際の図も保存。
```distance_based_clustering.py
data = pd.read_csv("all_result.csv")
# 0行目から10000行目までのデータに対して、1000行ごとに閾値35を用いてクラスタリング
# save_figで図を保存するかを指定
for i in range(0, 10000, 1000):
   plot_coordinate(data, 35, 35, i, "single", save_fig=True)
```

## time_variation_analysis.py
すべての時間に対して、クラスタ数、最大クラスタサイズ、最小クラスタサイズ、最大半径、最小半径を調べる。
クラスタ数はクラスタサイズが1より大きいものの数を表す。  
戻り値は辞書型。  
Returns (```dict```):
- n_clusters (```list```): クラスタ数  
- max_clustersize (```list```): 最大クラスタサイズ  
- min_clustersize (```list```): 最小クラスタサイズ(size1を除く)  
- max_radius (```list```): 最大半径  
- min_radius (```list```): 最小半径

## plot_timevariation.py
time_variation_analysis.pyで求めた結果をプロットするプログラム。
