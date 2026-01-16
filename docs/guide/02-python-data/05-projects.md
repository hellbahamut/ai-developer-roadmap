---
title: 综合项目：探索性数据分析 (EDA)
description: 完整的Kaggle级别数据分析项目实战
order: 5
---

# 综合项目：探索性数据分析 (EDA)

## 项目描述
**泰坦尼克号生存预测 (Titanic Survival Prediction)** 是Kaggle上最经典的入门竞赛。本项目不要求你训练复杂的模型，而是重点在于**数据分析**和**特征工程**。

## 任务清单

### 1. 数据加载与初步概览 (Data Overview)
*   加载 `train.csv`。
*   查看数据维度、列名、数据类型。
*   使用 `df.describe()` 查看统计摘要。
*   检查缺失值比例 (`df.isnull().mean()`)。

### 2. 单变量分析 (Univariate Analysis)
*   **目标变量 (Survived)**：生存率是多少？样本是否平衡？
*   **数值变量 (Age, Fare)**：分布是怎样的？是否有偏态？是否有异常值？
*   **分类变量 (Pclass, Sex, Embarked)**：各类别占比如何？

### 3. 双变量分析 (Bivariate Analysis)
*   **Sex vs Survived**: 女性生存率是否显著高于男性？
*   **Pclass vs Survived**: 头等舱乘客是否更容易获救？
*   **Age vs Survived**: 小孩子和老人的生存率如何？（可以使用分箱分析）
*   **Fare vs Pclass**: 票价和舱位是否强相关？

### 4. 数据清洗与特征工程 (Cleaning & Feature Engineering)
*   **缺失值处理**：
    *   `Age`: 可以用中位数填充，或者根据 `Title` (称谓) 分组填充。
    *   `Embarked`: 填充众数。
    *   `Cabin`: 缺失太多，可以提取"是否有客舱号"作为一个新特征，或者直接丢弃。
*   **特征提取**：
    *   从 `Name` 中提取 `Title` (Mr, Mrs, Miss, Master 等)。
    *   从 `SibSp` 和 `Parch` 计算 `FamilySize`。
    *   创造 `IsAlone` 特征。

### 5. 总结与洞察
*   总结哪些特征对生存预测最重要。
*   输出一份清洗后、特征丰富的数据集 `titanic_cleaned.csv`。

## 示例代码片段

```python
# 提取称谓
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# 查看称谓与生存率的关系
df.groupby('Title')['Survived'].mean().sort_values(ascending=False)
```

## 交付物
1.  一个结构清晰、包含代码和分析文字的 `.ipynb` 文件。
2.  生成的 `titanic_cleaned.csv` 数据集。
3.  (可选) 尝试使用逻辑回归模型对测试集进行预测，并提交到 Kaggle 查看排名。
