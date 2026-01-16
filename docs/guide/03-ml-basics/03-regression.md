---
title: 模块三：监督学习 - 回归
description: 线性回归、正则化与决策树回归
order: 3
---

# 模块三：监督学习 - 回归

回归问题的目标是预测一个连续值。

## 3.1 线性回归 (Linear Regression)

*   **模型**: $y = w_1x_1 + ... + w_nx_n + b$
*   **损失函数**: 均方误差 (MSE) $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$
*   **求解**: 最小二乘法 (OLS) 或 梯度下降。

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
print("权重:", model.coef_)
print("截距:", model.intercept_)
```

## 3.2 正则化回归 ⭐⭐⭐⭐⭐

为了防止过拟合，在损失函数中加入正则项。

### 3.2.1 Ridge (L2正则化)
*   **特点**: 系数趋向于0但不会等于0。
*   **适用**: 特征之间存在共线性，防止过拟合。

### 3.2.2 Lasso (L1正则化)
*   **特点**: 可以将系数压缩为0。
*   **适用**: 特征选择（自动去除不重要的特征）。

### 3.2.3 ElasticNet
*   结合了 L1 和 L2。

```python
from sklearn.linear_model import Ridge, Lasso

ridge = Ridge(alpha=1.0) # alpha是正则化强度
ridge.fit(X_train, y_train)

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
```

## 3.3 非线性回归

### 3.3.1 多项式回归
通过增加高次项特征，使线性模型能拟合非线性数据。

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
```

### 3.3.2 决策树回归
能够拟合复杂的非线性关系，但容易过拟合。

```python
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(max_depth=3)
```

### 3.3.3 SVR (支持向量回归)
```python
from sklearn.svm import SVR
svr = SVR(kernel='rbf') # 需要特征缩放
```
