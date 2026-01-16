---
title: 综合项目：端到端机器学习实战
description: 泰坦尼克生存预测与Pipeline实战
order: 7
---

# 综合项目：端到端机器学习实战

## 项目一：泰坦尼克号生存预测 (进阶版)

**目标**：构建一个完整的机器学习 Pipeline，使用随机森林进行预测，并进行超参数调优。

### 代码框架

```python
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. 加载数据
df = pd.read_csv('train.csv')
X = df.drop(['Survived', 'Name', 'Ticket', 'Cabin'], axis=1) # 简化特征
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 定义预处理 Pipeline
numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['Pclass', 'Sex', 'Embarked']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 3. 定义完整 Pipeline (预处理 + 模型)
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier())])

# 4. 训练
clf.fit(X_train, y_train)

# 5. 评估
print("Test Accuracy: %.3f" % clf.score(X_test, y_test))

# 6. 调优
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [5, 10, None]
}
grid = GridSearchCV(clf, param_grid, cv=5)
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)
```

## 项目二：加利福尼亚房价预测 (回归)

**目标**：使用 XGBoost 预测房价。
1.  加载 `sklearn.datasets.fetch_california_housing` 数据。
2.  进行 EDA，发现特征之间的关系。
3.  构建 XGBoost 回归模型。
4.  使用 RMSE 评估模型性能。
5.  绘制特征重要性图。

## 进阶挑战
尝试使用 **Stacking** (堆叠泛化) 技术，结合 逻辑回归、随机森林 和 XGBoost 的预测结果，看能否进一步提升泰坦尼克号的预测准确率。
