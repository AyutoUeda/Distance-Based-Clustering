# Hierarchical Clustering

## modules.py

階層クラスタリングに使用する基本的な関数群

|name of function | function|
| --------------- | ------- |
| hierarchical_clustering | 指定された行数（時間）における座標データとクラスタをプロットする関数|
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

## ■関数

・hierachical_clustering

戻り値：

    model.n_clusters_ --> クラスタ数（サイズ1を含む）
    
    model.labels_ --> 各点のラベル（要素数60）
    
    coordinate_data --> 引数で指定された時間における各点の座標(二次元配列、要素数60)
    
    model --> 引数で指定されたlinkage, distance_thresholdをモデルに組み込む
