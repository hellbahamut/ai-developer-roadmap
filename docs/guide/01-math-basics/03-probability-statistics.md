---
title: 模块三：概率论与统计
description: 概率分布、贝叶斯定理、假设检验与信息论
order: 3
---

# 模块三：概率论与统计

## 3.1 核心知识点

### 3.1.1 概率分布

**离散型分布**：
*   **伯努利分布 (Bernoulli)**：0/1分布，抛一次硬币。应用：二分类。
*   **二项分布 (Binomial)**：$n$次伯努利试验中成功的次数。
*   **泊松分布 (Poisson)**：单位时间内事件发生的次数。

**连续型分布**：
*   **均匀分布 (Uniform)**：在区间内概率相等。应用：参数初始化。
*   **正态分布 (Normal / Gaussian) ⭐⭐⭐⭐⭐**：
    *   符号：$X \sim N(\mu, \sigma^2)$。
    *   **中心极限定理**：大量独立随机变量之和趋于正态分布。
    *   **AI应用**：误差假设、权重初始化、变分自编码器(VAE)。

### 3.1.2 贝叶斯定理 (Bayes' Theorem)

$$ P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)} $$

*   **$P(A)$ 先验概率 (Prior)**：在看到证据之前对A的信念。
*   **$P(B|A)$ 似然 (Likelihood)**：假设A成立，证据B出现的概率。
*   **$P(A|B)$ 后验概率 (Posterior)**：看到证据B后，对A信念的更新。
*   **AI应用**：贝叶斯推断、朴素贝叶斯分类器、在线学习（不断更新后验）。

### 3.1.3 统计推断

*   **最大似然估计 (MLE)**：
    *   思想：找到一组参数，使得观测数据出现的概率（似然）最大。
    *   应用：深度学习中大多数损失函数（如交叉熵）的推导基础。
*   **最大后验估计 (MAP)**：
    *   思想：在MLE基础上加入了先验知识（正则化）。

### 3.1.4 信息论基础

*   **熵 (Entropy)**：衡量分布的不确定性。$H(X) = -\sum p(x) \log p(x)$。
*   **交叉熵 (Cross-Entropy)**：衡量两个分布的差异。
    *   **AI应用**：分类任务的标准损失函数。
*   **KL散度 (Relative Entropy)**：$D_{KL}(P||Q)$，衡量分布P和Q的距离（非对称）。

## 3.2 Python实现

```python
import numpy as np
from scipy import stats

# 1. 正态分布
mu, sigma = 0, 1
s = np.random.normal(mu, sigma, 1000)

# 2. 贝叶斯定理计算
def bayes_theorem(p_a, p_b_given_a, p_b):
    return (p_b_given_a * p_a) / p_b

# 示例：疾病检测
# P(病) = 0.01
# P(阳|病) = 0.99
# P(阳|健康) = 0.05
# 求 P(病|阳)
p_disease = 0.01
p_pos_given_disease = 0.99
p_pos_given_healthy = 0.05
p_pos = p_pos_given_disease * p_disease + p_pos_given_healthy * (1 - p_disease)
result = bayes_theorem(p_disease, p_pos_given_disease, p_pos)
print(f"患病概率: {result:.4f}") # 约 16.7%

# 3. 熵与交叉熵
def entropy(p):
    return -np.sum(p * np.log2(p + 1e-10))

def cross_entropy(p, q):
    return -np.sum(p * np.log2(q + 1e-10))

p = np.array([1, 0, 0])      # 真实标签 (one-hot)
q = np.array([0.7, 0.2, 0.1]) # 预测概率
print(f"交叉熵损失: {cross_entropy(p, q):.4f}")
```

## 3.3 练习题

1.  **概率陷阱**：为什么“检测结果为阳性，你实际患病的概率可能很低”？（参考贝叶斯代码示例）。
2.  **MLE推导**：假设数据服从正态分布，推导均值 $\mu$ 的最大似然估计值就是样本均值。
3.  **朴素贝叶斯**：使用 `sklearn.naive_bayes` 对简单的文本数据（如垃圾邮件分类）进行建模。

## 3.4 学习资源

*   **视频**：[StatQuest with Josh Starmer](https://space.bilibili.com/23910356) (非常生动有趣，强烈推荐)。
*   **教材**：《程序员的数学2：概率统计》。
*   **可视化**：[Seeing Theory](https://seeing-theory.brown.edu/) (交互式概率统计)。
