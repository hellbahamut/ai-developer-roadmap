---
title: 模块六：无监督学习
description: K-Means聚类与PCA降维
order: 6
---

# 模块六：无监督学习

## 6.1 聚类 (Clustering)

### 6.1.1 K-Means ⭐⭐⭐⭐⭐
*   **算法**:
    1.  随机初始化K个中心。
    2.  将每个点分配给最近的中心。
    3.  更新中心为簇内均值。
    4.  重复直到收敛。
*   **K的选择**: 肘部法则 (Elbow Method)。

```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
labels = kmeans.labels_
```

### 6.1.2 DBSCAN
基于密度的聚类，不需要指定K，能发现任意形状的簇，且能识别噪声。

## 6.2 降维 (Dimensionality Reduction)

### 6.2.1 PCA (主成分分析) ⭐⭐⭐⭐⭐
*   **原理**: 将数据投影到方差最大的方向（主成分），去除相关性，保留主要信息。
*   **应用**: 可视化、去噪、加速训练。

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95) # 保留95%的方差
X_pca = pca.fit_transform(X)
```

### 6.2.2 t-SNE
用于高维数据的可视化（降维到2D或3D），能很好地保留局部结构。
