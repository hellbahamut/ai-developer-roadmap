---
title: 综合项目与实战
description: 线性回归与高斯混合模型实战
order: 4
---

# 综合项目与实战

本模块通过两个经典项目，将线性代数、微积分和概率统计的知识串联起来。

## 项目一：从零实现线性回归 (Linear Regression)

**目标**：不使用 sklearn，仅使用 NumPy 实现一元线性回归。

### 1. 任务描述
给定一组数据点 $(x, y)$，假设它们满足 $y = wx + b + \epsilon$，其中 $\epsilon$ 是噪声。请找到最佳的 $w$ 和 $b$。

### 2. 数学原理
*   **模型**：$\hat{y} = wx + b$
*   **损失函数 (MSE)**：$L(w, b) = \frac{1}{2N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$
*   **梯度计算**：
    *   $\frac{\partial L}{\partial w} = \frac{1}{N} \sum (y_i - \hat{y}_i)(-x_i)$
    *   $\frac{\partial L}{\partial b} = \frac{1}{N} \sum (y_i - \hat{y}_i)(-1)$

### 3. 代码框架

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成数据
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1) # y = 4 + 3x + noise

# 2. 初始化参数
w = np.random.randn(1)
b = np.random.randn(1)
lr = 0.1
iterations = 1000

# 3. 梯度下降
for i in range(iterations):
    # 前向传播
    y_pred = w * X + b
    
    # 计算梯度
    dw = - (1/len(X)) * np.sum((y - y_pred) * X)
    db = - (1/len(X)) * np.sum(y - y_pred)
    
    # 更新参数
    w = w - lr * dw
    b = b - lr * db

print(f"结果: w={w}, b={b}")
# 预期结果应接近 w=3, b=4

# 4. 可视化
plt.scatter(X, y)
plt.plot(X, w*X + b, color='red')
plt.show()
```

## 项目二：高斯混合模型 (GMM) 理解

**目标**：理解 EM 算法（Expectation-Maximization）的思想。

### 1. 任务描述
数据是由两个不同的正态分布混合生成的，你需要把它们分开（聚类）。

### 2. 核心概念
*   **隐变量**：我们不知道每个数据点具体属于哪个分布。
*   **E步 (Expectation)**：固定参数，猜测每个数据点属于每个分布的概率（软分类）。
*   **M步 (Maximization)**：固定分类概率，更新每个分布的参数（均值和方差）。

### 3. 挑战
尝试使用 `sklearn.mixture.GaussianMixture` 对生成的数据进行聚类，并可视化结果。

```python
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# 生成数据
X, y_true = make_blobs(n_samples=400, centers=2, cluster_std=0.60, random_state=0)

# 训练GMM
gmm = GaussianMixture(n_components=2)
gmm.fit(X)
labels = gmm.predict(X)

# 可视化
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
plt.show()
```

## 进阶挑战

如果你完成了上述项目，可以尝试：
1.  推导线性回归的**正规方程 (Normal Equation)** 解：$\theta = (X^T X)^{-1} X^T y$。
2.  使用 PyTorch 自动求导机制重新实现线性回归，感受框架的便利。
