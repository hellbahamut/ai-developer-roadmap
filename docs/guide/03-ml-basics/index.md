---
title: 第三阶段：机器学习基础
description: 掌握经典ML算法与Scikit-learn框架
order: 3
---

# 第三阶段：机器学习基础

## 阶段目标
掌握经典机器学习算法原理，熟练使用 **Scikit-learn** 框架，能够独立完成从数据处理到模型部署的端到端机器学习项目。

## 学习时间规划
*   **总时长**： 3-4个月
*   **每周投入**： 15-20小时
*   **建议节奏**： 每个子模块2-3周

## 核心模块导航

| 模块 | 内容重点 | 预计时长 |
| :--- | :--- | :--- |
| **[01. 基础概念](./01-concepts.md)** | 监督/无监督、过拟合/欠拟合、偏差/方差 | 1周 |
| **[02. 数据预处理](./02-preprocessing.md)** | 清洗、缩放、编码、特征选择 | 2周 |
| **[03. 监督学习-回归](./03-regression.md)** | 线性回归、决策树回归、正则化 | 3周 |
| **[04. 监督学习-分类](./04-classification.md)** | 逻辑回归、SVM、随机森林、XGBoost | 4周 |
| **[05. 评估与调优](./05-evaluation.md)** | 交叉验证、网格搜索、ROC/AUC | 2周 |
| **[06. 无监督学习](./06-unsupervised.md)** | K-Means聚类、PCA降维 | 2周 |
| **[07. 综合项目](./07-projects.md)** | 泰坦尼克生存预测、房价预测 | 2周 |

## 环境配置

### 安装 Scikit-learn
推荐使用 conda 安装：
```bash
conda install scikit-learn
# 或者 pip
pip install scikit-learn
```

验证安装：
```python
import sklearn
print(sklearn.__version__)
```

### 推荐工具栈
*   **Scikit-learn**: 核心算法库
*   **XGBoost / LightGBM**: 高性能梯度提升树
*   **Category Encoders**: 高级类别编码
*   **MLflow**: 实验跟踪（进阶）

## 阶段验收与总结

### 验收标准

#### 1. 基础知识
*   [ ] 深刻理解监督学习 vs 无监督学习
*   [ ] 能清晰解释过拟合、欠拟合及其解决方案
*   [ ] 理解偏差-方差权衡 (Bias-Variance Tradeoff)

#### 2. 数据工程
*   [ ] 熟练处理缺失值和异常值
*   [ ] 掌握特征缩放 (StandardScaler) 和编码 (OneHot)
*   [ ] 能够使用 Scikit-learn 的 `Pipeline` 构建处理流

#### 3. 算法掌握
*   [ ] **回归**：线性回归、正则化 (L1/L2)
*   [ ] **分类**：逻辑回归、决策树、随机森林、XGBoost
*   [ ] **聚类**：K-Means
*   [ ] **降维**：PCA

#### 4. 评估与调优
*   [ ] 熟练使用交叉验证 (Cross-Validation)
*   [ ] 掌握精确率、召回率、F1、AUC 等指标
*   [ ] 能使用 `GridSearchCV` 进行超参数搜索

### 常见问题 (Q&A)

**Q1: 如何选择合适的算法？**
**A**:
*   **Baseline**: 先用简单的逻辑回归/线性回归。
*   **结构化数据**: 首选树模型 (Random Forest, XGBoost)。
*   **高维稀疏数据**: 尝试线性模型或 SVM。
*   **无标签**: K-Means 或 DBSCAN。

**Q2: 特征工程重要吗？**
**A**: 极其重要。业界名言：“数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限。”

**Q3: 模型不收敛怎么办？**
**A**:
1.  检查特征是否进行了**缩放** (Scaling)。
2.  调整**学习率** (Learning Rate)。
3.  检查数据是否存在严重异常值。
