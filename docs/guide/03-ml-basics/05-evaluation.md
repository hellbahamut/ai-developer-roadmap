---
title: 模块五：模型评估与调优
description: 评估指标、交叉验证与超参数搜索
order: 5
---

# 模块五：模型评估与调优

## 5.1 评估指标

### 5.1.1 分类指标
*   **准确率 (Accuracy)**: 预测对的比例。类别不平衡时失效。
*   **精确率 (Precision)**: 预测为正的样本中有多少是真的正样本。 (宁缺毋滥)
*   **召回率 (Recall)**: 真实正样本中有多少被预测出来了。 (宁可错杀不可放过)
*   **F1-Score**: 精确率和召回率的调和平均。
*   **AUC-ROC**: 衡量模型排序能力的指标，对阈值不敏感。

```python
from sklearn.metrics import classification_report, roc_auc_score

print(classification_report(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))
```

### 5.1.2 回归指标
*   **MSE (均方误差)**: 对大误差敏感。
*   **MAE (平均绝对误差)**: 鲁棒性较好。
*   **R² Score**: 拟合优度，1表示完美，0表示和均值模型一样差。

## 5.2 交叉验证 (Cross-Validation) ⭐⭐⭐⭐⭐

为了更可靠地评估模型性能，避免一次划分的偶然性。

*   **K-Fold**: 将数据分成K份，轮流做验证集。
*   **Stratified K-Fold**: 保持每折中类别比例一致（分类问题常用）。

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print("平均准确率:", scores.mean())
```

## 5.3 超参数调优

### 5.3.1 网格搜索 (Grid Search)
穷举搜索所有参数组合。

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, None]
}

grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)
print("最佳参数:", grid.best_params_)
```

### 5.3.2 随机搜索 (Random Search)
在参数空间随机采样，效率通常比网格搜索高。
