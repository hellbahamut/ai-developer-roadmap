---
title: 模块二：Pandas数据分析
description: DataFrame操作、数据清洗与分组聚合
order: 2
---

# 模块二：Pandas数据分析

Pandas 是基于 NumPy 构建的数据分析工具，提供了高效地操作大型数据集所需的工具。

## 2.1 核心知识点

### 2.1.1 Series 与 DataFrame

*   **Series**: 带标签的一维数组。
*   **DataFrame**: 带标签的二维表格（最常用）。

```python
import pandas as pd

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['NY', 'LA', 'SF']
})
```

### 2.1.2 数据读取与存储

```python
# 读取
df = pd.read_csv('data.csv')
df = pd.read_excel('data.xlsx')
df = pd.read_sql('SELECT * FROM table', conn)

# 存储
df.to_csv('output.csv', index=False)
```

### 2.1.3 数据选择与过滤 ⭐⭐⭐⭐⭐

*   **`loc`**: 基于标签（Label）的选择。
*   **`iloc`**: 基于位置（Integer Position）的选择。

```python
# 选择列
df['age']
df[['name', 'age']]

# 选择行
df.loc[0]           # 第1行（假设索引是0,1,2...）
df.iloc[0]          # 第1行（绝对位置）

# 条件过滤
adults = df[df['age'] >= 18]
ny_residents = df.query('city == "NY"')
```

### 2.1.4 数据清洗

**处理缺失值**：
```python
df.isnull().sum()           # 统计缺失值
df.dropna()                 # 删除包含缺失值的行
df.fillna(0)                # 填充0
df['age'].fillna(df['age'].mean(), inplace=True) # 填充均值
```

**处理重复值**：
```python
df.drop_duplicates(subset=['id'], keep='first')
```

**类型转换**：
```python
df['price'] = df['price'].astype(float)
df['date'] = pd.to_datetime(df['date'])
```

### 2.1.5 数据变换 (`apply` 与 `map`)

```python
# map: 用于Series，字典映射
df['gender_code'] = df['gender'].map({'Male': 0, 'Female': 1})

# apply: 用于Series或DataFrame，应用函数
df['name_len'] = df['name'].apply(len)
df.apply(lambda row: row['a'] + row['b'], axis=1) # 按行操作
```

### 2.1.6 分组与聚合 (`groupby`) ⭐⭐⭐⭐⭐

这是Pandas中最强大的功能之一，类似于SQL的 `GROUP BY`。

```python
# 模式：Split-Apply-Combine
grouped = df.groupby('city')

# 简单聚合
print(grouped['age'].mean())

# 多种聚合
agg_df = grouped.agg({
    'age': ['mean', 'min', 'max'],
    'salary': 'sum'
})
```

### 2.1.7 数据合并

*   **`pd.merge()`**: 类似SQL Join (inner, left, right, outer)。
*   **`pd.concat()`**: 物理拼接（按行或按列）。

```python
# 连接
merged = pd.merge(users, orders, on='user_id', how='left')

# 拼接
combined = pd.concat([df_2023, df_2024], axis=0) # 上下拼接
```

## 2.2 实战项目：销售数据清洗Pipeline

编写一个函数 `clean_sales_data(df)`，执行以下步骤：
1.  将 `date` 列转换为日期时间类型。
2.  删除 `order_id` 重复的行。
3.  填充 `customer_age` 的缺失值为平均年龄。
4.  创建一个新列 `month`，从 `date` 中提取月份。
5.  过滤掉 `amount` 小于等于0的异常订单。
6.  返回清洗后的 DataFrame。

```python
def clean_sales_data(df):
    # 1. 时间转换
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # 2. 去重
    df = df.drop_duplicates(subset=['order_id'])
    
    # 3. 缺失值
    mean_age = df['customer_age'].mean()
    df['customer_age'] = df['customer_age'].fillna(mean_age)
    
    # 4. 特征提取
    df['month'] = df['date'].dt.month
    
    # 5. 异常值处理
    df = df[df['amount'] > 0]
    
    return df

## 2.3 实战案例：电商销售数据综合分析

我们来模拟一个真实的电商数据分析场景。

**场景**：
你拿到了一份包含订单信息的CSV数据，需要回答以下问题：
1.  每个月的总销售额是多少？
2.  哪个产品的销量最高？
3.  不同城市用户的平均客单价是多少？
4.  销售额随时间的变化趋势如何？

**代码实现**：

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 模拟生成数据 (实际场景通常是 pd.read_csv('orders.csv'))
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=100)
data = {
    'order_id': range(1001, 1101),
    'date': dates,
    'product': np.random.choice(['Laptop', 'Mouse', 'Keyboard', 'Monitor'], 100),
    'city': np.random.choice(['Beijing', 'Shanghai', 'Shenzhen', 'Guangzhou'], 100),
    'amount': np.random.randint(50, 2000, 100),
    'quantity': np.random.randint(1, 5, 100)
}
df = pd.DataFrame(data)

# 2. 数据概览
print("前5行数据:")
print(df.head())
print("\n数据基本信息:")
print(df.info())

# 3. 数据分析

# Q1: 每月总销售额
# 将日期设置为索引，按月重采样
monthly_sales = df.set_index('date').resample('M')['amount'].sum()
print("\n每月销售额:")
print(monthly_sales)

# Q2: 产品销量排行
product_sales = df.groupby('product')['quantity'].sum().sort_values(ascending=False)
print("\n产品销量排行:")
print(product_sales)

# Q3: 城市客单价 (总金额 / 订单数)
city_metrics = df.groupby('city').agg({
    'amount': 'sum',
    'order_id': 'count'
})
city_metrics['avg_order_value'] = city_metrics['amount'] / city_metrics['order_id']
print("\n城市客单价:")
print(city_metrics.sort_values('avg_order_value', ascending=False))

# 4. 可视化
plt.figure(figsize=(12, 5))

# 子图1: 每月销售趋势
plt.subplot(1, 2, 1)
monthly_sales.plot(kind='line', marker='o', title='Monthly Sales Trend')
plt.grid(True)

# 子图2: 产品销量分布
plt.subplot(1, 2, 2)
product_sales.plot(kind='bar', color='skyblue', title='Product Sales Quantity')

plt.tight_layout()
plt.show()
```

**关键点解析**：
*   **`resample('M')`**: 处理时间序列数据的神器，比手动提取月份再groupby更方便。
*   **`agg()`**: 一次性计算多种统计指标。
*   **数据可视化**: 分析结果必须配合图表才能直观呈现趋势。

---

## 2.4 学习资源
```
