---
title: 模块一：机器学习基础概念
description: 监督学习、过拟合与工作流程
order: 1
---

# 模块一：机器学习基础概念

## 1.1 什么是机器学习

**机器学习 (Machine Learning)** 是从数据中自动分析获得模型，并利用模型对未知数据进行预测的技术。

*   **传统编程**：规则 + 数据 → 答案
*   **机器学习**：数据 + 答案 → 规则

## 1.2 机器学习类型

### 1.2.1 监督学习 (Supervised Learning) ⭐⭐⭐⭐⭐
使用**有标签**的数据训练模型。
*   **回归 (Regression)**：预测连续值（如房价、温度）。
*   **分类 (Classification)**：预测离散类别（如垃圾邮件、猫狗分类）。

### 1.2.2 无监督学习 (Unsupervised Learning)
使用**无标签**的数据发现模式。
*   **聚类 (Clustering)**：客户细分。
*   **降维 (Dimensionality Reduction)**：特征压缩、可视化。

### 1.2.3 强化学习 (Reinforcement Learning)
通过与环境交互获得奖励信号来学习策略（如AlphaGo、自动驾驶）。

## 1.3 核心概念

### 1.3.1 数据集划分
*   **训练集 (Training Set)**：用于训练模型参数。
*   **验证集 (Validation Set)**：用于调整超参数，选择最佳模型。
*   **测试集 (Test Set)**：用于最终评估，**严禁**在训练过程中使用。

### 1.3.2 过拟合与欠拟合 ⭐⭐⭐⭐⭐

| 现象 | 表现 | 原因 | 解决方案 |
| :--- | :--- | :--- | :--- |
| **欠拟合 (Underfitting)** | 训练误差大，测试误差大 | 模型太简单 | 增加特征、使用更复杂模型 |
| **过拟合 (Overfitting)** | 训练误差小，测试误差大 | 模型太复杂，记住了噪声 | 更多数据、正则化、简化模型 |

### 1.3.3 偏差-方差权衡 (Bias-Variance Tradeoff)
*   **高偏差**：欠拟合。
*   **高方差**：过拟合。
*   **目标**：找到偏差和方差的平衡点。

## 1.4 Scikit-learn 标准工作流

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. 准备数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. 预处理
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. 训练
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 4. 预测与评估
y_pred = model.predict(X_test_scaled)
print(accuracy_score(y_test, y_pred))
```
