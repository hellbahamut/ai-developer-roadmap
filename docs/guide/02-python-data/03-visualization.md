---
title: 模块三：数据可视化
description: Matplotlib与Seaborn绘图实战
order: 3
---

# 模块三：数据可视化

## 3.1 Matplotlib 基础

Matplotlib 是Python绘图的始祖，功能强大但API相对底层。

### 3.1.1 基础绘图流程

```python
import matplotlib.pyplot as plt
import numpy as np

# 1. 准备数据
x = np.linspace(0, 10, 100)
y = np.sin(x)

# 2. 创建画布和坐标轴
fig, ax = plt.subplots(figsize=(10, 6))

# 3. 绘图
ax.plot(x, y, label='sin(x)', color='blue', linewidth=2)
ax.scatter(x[::10], y[::10], color='red', label='Samples')

# 4. 装饰
ax.set_title("Sine Wave Example")
ax.set_xlabel("Time")
ax.set_ylabel("Amplitude")
ax.legend()
ax.grid(True, alpha=0.3)

# 5. 显示或保存
plt.show()
# plt.savefig('plot.png', dpi=300)
```

### 3.1.2 子图布局 (`subplots`)

```python
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].plot(x, y)
axes[0, 0].set_title('Line Plot')

axes[0, 1].hist(np.random.randn(1000))
axes[0, 1].set_title('Histogram')

plt.tight_layout()
plt.show()
```

## 3.2 Seaborn 进阶

Seaborn 建立在 Matplotlib 之上，提供了更高级的统计图表接口和更美观的默认样式。

```python
import seaborn as sns
import pandas as pd

# 加载内置数据集
tips = sns.load_dataset('tips')
```

### 3.2.1 常用统计图表

**1. 分布图 (Distplot/Histplot)**
```python
sns.histplot(data=tips, x='total_bill', hue='sex', kde=True)
```

**2. 箱线图与小提琴图 (Boxplot/Violinplot)**
用于查看数据分布和异常值。
```python
sns.boxplot(data=tips, x='day', y='total_bill')
sns.violinplot(data=tips, x='day', y='total_bill', hue='smoker')
```

**3. 热力图 (Heatmap)**
常用于展示相关性矩阵。
```python
corr = tips.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
```

**4. 关系图 (Relplot/Scatterplot)**
```python
sns.scatterplot(data=tips, x='total_bill', y='tip', hue='time', size='size')
```

**5. 成对关系图 (Pairplot)**
快速查看多变量两两之间的关系。
```python
sns.pairplot(tips, hue='sex')
```

## 3.3 可视化最佳实践

1.  **选择正确的图表**：
    *   **比较**：柱状图 (Bar Chart)
    *   **分布**：直方图 (Histogram)、箱线图 (Boxplot)
    *   **关系**：散点图 (Scatter Plot)
    *   **趋势**：折线图 (Line Chart)
    *   **构成**：饼图 (Pie Chart - 慎用)、堆叠柱状图
2.  **Less is More**：避免过多的颜色和杂乱的元素，突出核心信息。
3.  **标注清晰**：务必包含标题、轴标签、单位和图例。
4.  **色彩使用**：使用感知均匀的色图（如 Viridis, Magma），注意色盲友好。
