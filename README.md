6月12日
円半径を求める

6月19日
最大半径の時間変化をフーリエ変換し、振動を見る
最大半径を持つクラスタの中心のプロットを他のクラスタでも実行
計算する範囲を拡大　(行の最大は13394）

6.28
全データのインポート（行数12万）--> all_result.csv

■関数
・hierachical_clustering
戻り値：
    model.n_clusters_ --> クラスタ数（サイズ1を含む）
    model.labels_ --> 各点のラベル（要素数60）
    coordinate_data --> 引数で指定された時間における各点の座標(二次元配列、要素数60)
    model --> 引数で指定されたlinkage, distance_thresholdをモデルに組み込む
