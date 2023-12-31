# Hierarchical Clustering

## ```modules.py```

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

data = pd.DataFrame([[1, 1, 2, 2, 3, 3, 4, 4, 5, 5], 
                     [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]])

n_clusters, labels, coordinate_data, model = hierarchical_clustering(data, time_analyze=0, threshold=2, "single")

centers, list_of_radius = calculate_cluster_centers(labels, coordinate_data, np.bincount(labels))

```


## ■進捗

- 6月12日

円半径を求める

- 6月19日

最大半径の時間変化をフーリエ変換し、振動を見る

最大半径を持つクラスタの中心のプロットを他のクラスタでも実行

計算する範囲を拡大　(行の最大は13394）

- 6.28

全データのインポート（行数12万）--> all_result.csv

- 7.4

クラスタサイズ別に数をカウント（1000秒区切り）

クラスタに含まれない粒子（サイズが1）をプロット

## ■関数

・hierachical_clustering

戻り値：

    model.n_clusters_ --> クラスタ数（サイズ1を含む）
    
    model.labels_ --> 各点のラベル（要素数60）
    
    coordinate_data --> 引数で指定された時間における各点の座標(二次元配列、要素数60)
    
    model --> 引数で指定されたlinkage, distance_thresholdをモデルに組み込む
