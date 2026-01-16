---
title: 模块四：监督学习 - 分类
description: 逻辑回归、SVM、随机森林与XGBoost
order: 4
---

# 模块四：监督学习 - 分类

分类问题的目标是预测离散的类别标签。

## 4.1 逻辑回归 (Logistic Regression) ⭐⭐⭐⭐⭐

虽然名字叫回归，但其实是分类算法。
*   **原理**: 使用 Sigmoid 函数将线性输出映射到 [0, 1] 概率区间。
*   **适用**: 二分类问题的基准模型 (Baseline)。

```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train, y_train)
probs = clf.predict_proba(X_test) # 获取概率
```

## 4.2 支持向量机 (SVM)

*   **原理**: 寻找一个超平面，最大化不同类别之间的间隔 (Margin)。
*   **核技巧 (Kernel Trick)**: 将低维不可分数据映射到高维空间使其可分 (RBF核)。
*   **注意**: 对特征缩放非常敏感。

```python
from sklearn.svm import SVC
svm = SVC(kernel='rbf', C=1.0, probability=True)
```

## 4.3 决策树与随机森林

### 4.3.1 决策树 (Decision Tree)
*   **原理**: 通过一系列规则（如 `if age > 30`）划分数据。
*   **优点**: 可解释性强，无需缩放。
*   **缺点**: 极易过拟合。

### 4.3.2 随机森林 (Random Forest) ⭐⭐⭐⭐⭐
*   **原理**: 集成学习 (Bagging)。训练多棵决策树，每棵树使用随机的数据子集和特征子集，最后投票决定结果。
*   **优点**: 鲁棒性强，不易过拟合，准确率高。

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, max_depth=5)
rf.fit(X_train, y_train)
# 特征重要性
print(rf.feature_importances_)
```

## 4.4 梯度提升树 (GBDT)

目前竞赛和工业界最强的传统ML算法。

### 4.4.1 XGBoost / LightGBM ⭐⭐⭐⭐⭐
*   **原理**: Boosting。串行训练树，每棵树都在修正前一棵树的错误（残差）。
*   **特点**: 速度快，精度高，支持缺失值。

```python
from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=100, learning_rate=0.1)
xgb.fit(X_train, y_train)
```

## 4.5 朴素贝叶斯 (Naive Bayes)
*   **原理**: 基于贝叶斯定理，假设特征之间相互独立。
*   **适用**: 文本分类 (垃圾邮件过滤)。

```python
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
```

## 4.6 实战案例：泰坦尼克号生存预测

我们使用经典的泰坦尼克号数据集，演示一个完整的机器学习分类任务流程。

**任务**：根据乘客的年龄、性别、票价等信息，预测其是否幸存。

**代码实现**：

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. 加载数据 (这里使用模拟数据，实际使用 pd.read_csv('titanic.csv'))
# 假设列: ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']
np.random.seed(42)
n_samples = 200
data = {
    'Pclass': np.random.choice([1, 2, 3], n_samples),
    'Sex': np.random.choice(['male', 'female'], n_samples),
    'Age': np.random.randint(1, 80, n_samples),
    'SibSp': np.random.randint(0, 3, n_samples),
    'Fare': np.random.uniform(10, 500, n_samples),
    'Survived': np.random.randint(0, 2, n_samples) # 标签
}
df = pd.DataFrame(data)

# 人为制造一点规律（女性、头等舱生存率高）
mask = (df['Sex'] == 'female') | (df['Pclass'] == 1)
df.loc[mask, 'Survived'] = np.random.choice([0, 1], mask.sum(), p=[0.3, 0.7])

# 2. 数据预处理
# 填充缺失值 (Age通常有缺失)
df['Age'].fillna(df['Age'].median(), inplace=True)

# 编码分类变量 (Sex)
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex']) # male/female -> 0/1

# 特征与标签
X = df.drop('Survived', axis=1)
y = df['Survived']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征缩放 (这对LR/SVM很重要，对随机森林影响不大)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. 模型训练与评估

# 模型1: 逻辑回归 (Baseline)
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
print("逻辑回归准确率:", accuracy_score(y_test, y_pred_lr))

# 模型2: 随机森林
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("随机森林准确率:", accuracy_score(y_test, y_pred_rf))

# 4. 查看详细报告
print("\n随机森林分类报告:")
print(classification_report(y_test, y_pred_rf))

# 5. 查看特征重要性
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
print("\n特征重要性:")
print(importances.sort_values(ascending=False))
```

**关键步骤总结**：
1.  **数据清洗**：处理缺失值是第一步，不能把NaN丢给模型。
2.  **特征编码**：计算机只认识数字，文本类别（如Sex）必须转换。
3.  **特征缩放**：不同特征量纲不同（如Age是0-80，Fare是0-500），缩放有助于模型收敛。
4.  **模型对比**：通常先用简单模型（LR）定基准，再尝试复杂模型（RF/XGBoost）。
