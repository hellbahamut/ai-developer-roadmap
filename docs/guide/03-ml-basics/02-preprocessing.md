---
title: 模块二：数据预处理与特征工程
description: 清洗、缩放、编码与特征选择
order: 2
---

# 模块二：数据预处理与特征工程

## 2.1 数据预处理

### 2.1.1 缺失值处理

```python
from sklearn.impute import SimpleImputer

# 均值填充
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# KNN填充（更智能）
from sklearn.impute import KNNImputer
knn_imputer = KNNImputer(n_neighbors=5)
X_knn = knn_imputer.fit_transform(X)
```

### 2.1.2 特征缩放 (Scaling) ⭐⭐⭐⭐⭐

**为什么需要缩放？**
许多算法（如SVM, KNN, 神经网络, 线性回归）基于距离或梯度，如果特征量纲不一致（如年龄 vs 收入），会导致模型收敛慢或性能差。

*   **StandardScaler (标准化)**: $\frac{x - \mu}{\sigma}$。结果均值为0，方差为1。最常用。
*   **MinMaxScaler (归一化)**: $\frac{x - min}{max - min}$。结果在 [0, 1] 之间。
*   **RobustScaler**: 使用中位数和四分位数，对异常值鲁棒。

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # 注意：测试集只能transform！
```

### 2.1.3 类别编码

*   **LabelEncoder**: 标签编码（0, 1, 2...）。用于目标变量 $y$ 或有序特征。
*   **OneHotEncoder**: 独热编码。用于无序特征（如颜色：红、绿、蓝）。

```python
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_encoded = enc.fit_transform(X_cat)
```

## 2.2 特征工程 ⭐⭐⭐⭐⭐

### 2.2.1 特征创建
*   **多项式特征**: $x_1, x_2 \rightarrow x_1^2, x_1x_2, x_2^2$。捕捉非线性关系。
*   **业务特征**: 如 `FamilySize = SibSp + Parch + 1`。

### 2.2.2 特征选择
*   **过滤法 (Filter)**: 基于统计指标（如相关系数、卡方检验）。
*   **包装法 (Wrapper)**: 递归特征消除 (RFE)。
*   **嵌入法 (Embedded)**: 基于模型（如Lasso回归的系数、随机森林的特征重要性）。

```python
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=10)
X_new = selector.fit_transform(X, y)
```

## 2.3 Scikit-learn Pipeline

Pipeline 将预处理和模型串联起来，防止数据泄漏，简化代码。

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
```
