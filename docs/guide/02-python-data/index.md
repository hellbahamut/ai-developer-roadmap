---
title: 第二阶段：Python数据处理
description: 熟练使用NumPy, Pandas, Matplotlib进行数据分析
order: 2
---

# 第二阶段：Python数据处理能力

## 阶段目标
熟练使用Python进行数据清洗、分析和可视化，为机器学习打下坚实的数据处理基础。在AI项目中，80%的时间通常花在数据准备上，因此本阶段至关重要。

## 学习时间规划
*   **总时长**： 1-2个月
*   **每周投入**： 15-20小时
*   **建议节奏**： 每个子模块1-2周

## 核心模块导航

| 模块 | 内容重点 | 预计时长 |
| :--- | :--- | :--- |
| **[01. NumPy进阶](./01-numpy.md)** | 多维数组操作、广播机制、向量化运算 | 1-2周 |
| **[02. Pandas数据分析](./02-pandas.md)** | DataFrame操作、数据清洗、分组聚合 | 2-3周 |
| **[03. 数据可视化](./03-visualization.md)** | Matplotlib基础、Seaborn统计绘图 | 1-2周 |
| **[04. Jupyter最佳实践](./04-jupyter.md)** | 魔法命令、快捷键、高效工作流 | 1周 |
| **[05. 综合项目](./05-projects.md)** | 完整的探索性数据分析 (EDA) | 1周 |

## 环境配置

### 1. 安装Anaconda（推荐）
Anaconda 是最流行的数据科学Python发行版，预装了所有必要的库。

*   **下载**：访问 [Anaconda官网](https://www.anaconda.com/download) 下载Python 3.x版本。
*   **安装**：Windows用户建议勾选 "Add to PATH"（或安装后手动配置环境变量）。
*   **验证**：
    ```bash
    conda --version
    python --version
    ```

### 2. 核心库版本检查
建议使用以下版本（或更新）：
*   **Python**: 3.10+
*   **NumPy**: 1.24+
*   **Pandas**: 2.0+
*   **Matplotlib**: 3.7+
*   **Seaborn**: 0.12+

安装/更新命令：
```bash
conda install numpy pandas matplotlib seaborn jupyter
# 或
pip install numpy pandas matplotlib seaborn jupyter
```

### 3. 启动 Jupyter Lab
Jupyter Lab 是比 Notebook 更现代化的开发环境。
```bash
jupyter lab
```

## 阶段验收与总结

### 验收标准

#### 1. NumPy操作 (30分)
*   [ ] 熟练创建和操作 `ndarray`
*   [ ] 深刻理解并能运用 **广播机制 (Broadcasting)**
*   [ ] 熟练掌握花式索引与切片
*   [ ] 能使用向量化运算替代显式循环

#### 2. Pandas数据处理 (40分)
*   [ ] 熟练读取 CSV, Excel, SQL 等格式数据
*   [ ] 掌握缺失值、重复值处理等数据清洗技术
*   [ ] 熟练使用 `groupby` 进行分组聚合
*   [ ] 能够灵活进行多表合并 (`merge`, `concat`)

#### 3. 数据可视化 (30分)
*   [ ] 能绘制基础图表（折线、柱状、散点、直方图）
*   [ ] 掌握 Matplotlib 的子图布局 (`subplots`)
*   [ ] 能使用 Seaborn 绘制统计图表（箱线图、热力图、PairPlot）
*   [ ] 图表清晰美观，包含必要的标题、标签和图例

#### 4. 项目能力
*   [ ] 完成至少2个完整的数据分析项目
*   [ ] 能独立进行探索性数据分析 (EDA)
*   [ ] 能从数据中提取有价值的商业洞察

## 常见问题 (Q&A)

**Q1: Pandas和NumPy的关系？**
**A**: NumPy是底层库，提供高性能的数组运算；Pandas基于NumPy构建，提供了更高级的表格数据结构（DataFrame）和数据分析工具。Pandas的底层数据存储在NumPy数组中。

**Q2: 什么时候用Matplotlib vs Seaborn？**
**A**:
*   **Matplotlib**：底层绘图库，灵活性极高，适合精细定制图表的每一个细节。
*   **Seaborn**：基于Matplotlib的高级库，默认样式美观，适合快速绘制统计类图表（如热力图、小提琴图）。
*   **建议**：用Seaborn进行快速探索，用Matplotlib进行最终报告的精细调整。

**Q3: 如何处理大数据（如超过内存大小）？**
**A**:
1.  使用 `chunksize` 参数分块读取。
2.  只加载需要的列 (`usecols`)。
3.  转换数据类型（如 `float64` -> `float32`，字符串 -> `category`）以节省内存。
4.  使用 Polars 或 Dask 等大数据处理库。
