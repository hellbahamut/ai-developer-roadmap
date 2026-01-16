---
title: 模块二：微积分
description: 导数、梯度、链式法则与优化理论
order: 2
---

# 模块二：微积分

## 2.1 核心知识点

### 2.1.1 导数 (Derivative)

*   **定义**：函数在某一点的切线斜率，描述了函数的**变化率**。
    *   $f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$
*   **常用公式**：
    *   $(x^n)' = nx^{n-1}$
    *   $(e^x)' = e^x$
    *   $(\ln x)' = 1/x$
    *   $(\sin x)' = \cos x$
*   **求导法则**：
    *   **链式法则 (Chain Rule)**：$[f(g(x))]' = f'(g(x)) \cdot g'(x)$。
    *   **AI应用**：**反向传播算法 (Backpropagation)** 的核心，用于计算神经网络中深层参数的梯度。

### 2.1.2 偏导数与梯度 (Gradient)

*   **偏导数 (Partial Derivative)**：多元函数对其中一个变量求导，其余变量视为常数。
    *   符号：$\frac{\partial f}{\partial x}$
*   **梯度 (Gradient)**：由所有偏导数组成的**向量**。
    *   符号：$\nabla f = [\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, ..., \frac{\partial f}{\partial x_n}]^T$
*   **几何意义**：
    *   梯度方向是函数**增长最快**的方向。
    *   负梯度方向 ($-\nabla f$) 是函数**下降最快**的方向。
*   **AI应用**：梯度下降算法，用于最小化损失函数。

### 2.1.3 雅可比矩阵与海森矩阵

*   **雅可比矩阵 (Jacobian)**：向量值函数的一阶偏导数矩阵。用于处理输入输出都是向量的函数变换。
*   **海森矩阵 (Hessian)**：二阶偏导数矩阵。用于判断极值点类型（极大值、极小值、鞍点）以及牛顿法优化。

## 2.2 优化理论

### 2.2.1 梯度下降算法 (Gradient Descent)

这是AI中最核心的优化算法。

*   **目标**：找到参数 $\theta$，使得损失函数 $J(\theta)$ 最小。
*   **迭代公式**：
    $$ \theta_{t+1} = \theta_t - \alpha \nabla J(\theta_t) $$
    *   $\alpha$：**学习率 (Learning Rate)**，控制步长的大小。
    *   $\nabla J(\theta_t)$：当前的梯度。

### 2.2.2 梯度下降的变体

1.  **批量梯度下降 (BGD)**：每次更新使用**所有**样本。计算准但慢。
2.  **随机梯度下降 (SGD)**：每次更新只使用**一个**样本。快但震荡。
3.  **小批量梯度下降 (Mini-batch GD)**：每次使用一小批（如32, 64个）样本。**最常用**，平衡了速度和稳定性。

### 2.2.3 高级优化器

*   **Momentum**：引入动量，加速收敛并抑制震荡。
*   **Adam**：结合了动量和自适应学习率，是目前**最流行**的优化器。

## 2.3 Python实现

### 2.3.1 数值梯度计算

```python
import numpy as np

def numerical_gradient(f, x, eps=1e-5):
    """
    计算多元函数 f 在 x 处的梯度
    """
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        # 中心差分公式
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad

# 测试
f = lambda x: x[0]**2 + x[1]**2  # z = x^2 + y^2
x_start = np.array([3.0, 4.0])
grad = numerical_gradient(f, x_start)
print(f"梯度: {grad}")  # 应接近 [6.0, 8.0]
```

### 2.3.2 简单梯度下降

```python
def gradient_descent(f, init_x, lr=0.1, steps=100):
    x = init_x.copy()
    history = [x.copy()]
    
    for _ in range(steps):
        grad = numerical_gradient(f, x)
        x = x - lr * grad
        history.append(x.copy())
        
    return x, np.array(history)

# 寻找最小值
final_x, path = gradient_descent(f, np.array([10.0, 10.0]))
print(f"最终位置: {final_x}") # 应接近 [0, 0]
```

## 2.4 练习题

### 基础练习
1.  **手动求导**：$f(x) = e^x \sin(x)$，求 $f'(x)$。
2.  **偏导数**：$f(x, y) = x^2y + y^3$，求 $\partial f/\partial x$ 和 $\partial f/\partial y$。

### 编程练习
1.  **可视化**：使用 Matplotlib 绘制 $z=x^2+y^2$ 的3D图像或等高线图，并画出梯度下降的轨迹。
2.  **线性回归**：实现一元线性回归 $y = wx + b$，定义均方误差损失函数，使用梯度下降求解 $w$ 和 $b$。

## 2.5 学习资源

*   **视频**：[3Blue1Brown - 微积分的本质](https://space.bilibili.com/88461692/channel/seriesdetail?sid=1528931)。
*   **工具**：[Desmos Graphing Calculator](https://www.desmos.com/) (可视化函数)。
*   **可视化**：[Gradient Descent Visualization](https://losslandscape.com/)。
