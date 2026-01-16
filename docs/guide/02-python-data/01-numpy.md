---
title: 模块一：NumPy进阶
description: 多维数组、广播机制与向量化计算
order: 1
---

# 模块一：NumPy进阶

NumPy (Numerical Python) 是Python科学计算的核心库，提供了高性能的多维数组对象和工具。

## 1.1 核心知识点

### 1.1.1 ndarray 多维数组 ⭐⭐⭐⭐⭐

**创建数组**：
```python
import numpy as np

# 从列表创建
a = np.array([1, 2, 3, 4])
b = np.array([[1, 2], [3, 4]])  # 2维

# 内置函数
zeros = np.zeros((3, 4))        # 全0
ones = np.ones((2, 3))          # 全1
eye = np.eye(3)                 # 单位矩阵
rand = np.random.randn(2, 3)    # 标准正态分布

# 序列
arange = np.arange(0, 10, 2)    # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 10, 5) # [0, 2.5, 5, 7.5, 10]
```

**数组属性**：
*   `arr.shape`: 维度元组，如 `(2, 3)`
*   `arr.ndim`: 维数
*   `arr.dtype`: 数据类型 (e.g., `float64`, `int32`)

### 1.1.2 索引与切片 ⭐⭐⭐⭐⭐

**基础索引**：
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr[0, 1])    # 2 (第0行第1列)
print(arr[:, 1])    # [2, 5] (第1列所有元素)
print(arr[0:2, :])  # 前两行
```

**布尔索引** (非常重要)：
```python
arr = np.array([1, 2, 3, 4, 5])
mask = arr > 3          # [False, False, False, True, True]
print(arr[mask])        # [4, 5]
arr[arr % 2 == 0] = 0   # 将偶数置为0
```

### 1.1.3 数组运算与广播机制 (Broadcasting) ⭐⭐⭐⭐⭐

**广播规则**：
当两个数组进行算术运算时，NumPy会对照它们的形状。如果满足以下条件之一，则认为它们是兼容的：
1.  维度相同。
2.  其中一个维度为1。

**示例**：
```python
A = np.array([[1, 2, 3], 
              [4, 5, 6]])   # shape (2, 3)
b = np.array([10, 20, 30])  # shape (3,) -> 自动补全为 (1, 3) -> 广播为 (2, 3)

print(A + b)
# [[11, 22, 33],
#  [14, 25, 36]]
```

**实际应用**：数据标准化
```python
data = np.random.rand(100, 3)
mean = data.mean(axis=0)    # (3,)
std = data.std(axis=0)      # (3,)
# (100, 3) - (3,) -> 广播机制自动处理
normalized = (data - mean) / std
```

### 1.1.4 统计与聚合

```python
arr = np.random.randn(3, 4)

print(arr.sum())            # 所有元素和
print(arr.mean(axis=0))     # 按列求均值 (压缩行)
print(arr.max(axis=1))      # 按行求最大值 (压缩列)
print(arr.argmax())         # 最大值的索引
```

### 1.1.5 性能优化：向量化

**核心原则**：尽量避免使用Python `for` 循环处理数组，而是使用NumPy的向量化操作。

**Bad**:
```python
res = []
for x in arr:
    res.append(x ** 2)
```

**Good**:
```python
res = arr ** 2
```

## 1.2 实战练习

### 练习1：图像处理基础
编写函数 `process_image(image)`，输入一张RGB图像 `(H, W, 3)`：
1.  转换为灰度图（加权平均：0.299R + 0.587G + 0.114B）。
2.  归一化到 [0, 1]。

### 练习2：欧氏距离矩阵
给定点集 `A` (shape `N, D`) 和点集 `B` (shape `M, D`)，计算它们之间的距离矩阵 `dist` (shape `N, M`)，其中 `dist[i, j]` 是 `A[i]` 和 `B[j]` 的欧氏距离。
*提示：利用公式 $(a-b)^2 = a^2 + b^2 - 2ab$ 和广播机制。*

### 练习3：K-Means简化版
实现一个简单的 K-Means 聚类算法的一个迭代步：
1.  给定数据 `X` 和中心点 `centers`。
2.  计算每个点到每个中心的距离。
3.  将每个点分配给最近的中心。
4.  更新中心点为各类别的均值。
