---
title: 模块四：Jupyter Notebook最佳实践
description: 魔法命令、快捷键与高效工作流
order: 4
---

# 模块四：Jupyter Notebook最佳实践

Jupyter Notebook 是数据科学家的标准IDE，它支持代码、Markdown文档、数学公式和可视化图表混合排版。

## 4.1 常用快捷键

**命令模式 (Esc)**:
*   `A`: 在上方插入单元格
*   `B`: 在下方插入单元格
*   `D, D`: 删除当前单元格
*   `M`: 切换为 Markdown 模式
*   `Y`: 切换为 Code 模式
*   `Shift + M`: 合并选中单元格

**编辑模式 (Enter)**:
*   `Shift + Enter`: 运行当前单元格并选中下一个
*   `Ctrl + Enter`: 运行当前单元格
*   `Tab`: 代码补全
*   `Shift + Tab`: 查看函数文档（非常有用！）

## 4.2 魔法命令 (Magic Commands)

以 `%` 开头的特殊命令。

*   `%timeit`: 测量代码运行时间（多次运行取平均）。
    ```python
    %timeit [x**2 for x in range(1000)]
    ```
*   `%time`: 测量单次运行时间。
*   `%pwd`: 显示当前工作目录。
*   `%ls`: 列出文件。
*   `%matplotlib inline`: 确保图表在Notebook中显示（现在的版本通常默认支持，但加上保险）。
*   `%load_ext autoreload`: 自动重载外部模块，开发库时非常有用。
    ```python
    %load_ext autoreload
    %autoreload 2
    ```

## 4.3 Notebook 结构规范

一个优秀的分析 Notebook 应该像一篇文章一样清晰：

1.  **标题与元数据**：项目名称、作者、日期。
2.  **导入库**：所有 `import` 放在第一个单元格。
3.  **配置**：设置绘图风格、pandas显示选项等。
    ```python
    pd.set_option('display.max_columns', None)
    plt.style.use('seaborn')
    ```
4.  **数据加载**：读取数据并展示 `head()`。
5.  **数据清洗与探索**：包含必要的 Markdown 注释，解释每一步的意图。
6.  **分析与可视化**：每个图表前应有分析假设，图表后应有结论洞察。
7.  **总结**：总结核心发现。

## 4.4 导出与分享

*   **导出为 HTML/PDF**：方便非技术人员阅读。
*   **清理 Output**：在提交到 Git 之前，建议 `Kernel -> Restart & Clear Output`，以减小文件体积并避免冲突。
