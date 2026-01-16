---
title: 模块一：线性代数
description: 向量、矩阵、特征值与SVD分解
order: 1
---

# 模块一：线性代数

## 1.1 核心知识点

### 1.1.1 标量、向量、矩阵

*   **标量 (Scalar)**
    *   **定义**：只有大小，没有方向的量。
    *   **表示**：单个数字，如 $x = 5$。
    *   **AI应用**：损失函数值、学习率、正则化系数。

*   **向量 (Vector)**
    *   **定义**：有大小和方向的量。
    *   **表示**：$n$维数组，如 $\mathbf{x} = [x_1, x_2, ..., x_n]$。
    *   **几何意义**：空间中的点或从原点出发的有向线段。
    *   **AI应用**：特征向量（Feature Vector）、权重向量、梯度。

*   **矩阵 (Matrix)**
    *   **定义**：$m \times n$ 的数字矩形阵列。
    *   **表示**：$A = [a_{ij}]$ ($i=1..m, j=1..n$)。
    *   **AI应用**：数据集（行是样本，列是特征）、权重矩阵、注意力矩阵。

### 1.1.2 向量运算

**基础运算**：
*   **向量加法**：$\mathbf{z} = \mathbf{x} + \mathbf{y}$，对应元素相加。几何上遵循平行四边形法则。
*   **标量乘法**：$\mathbf{z} = \alpha \cdot \mathbf{x}$，每个元素乘以标量，改变向量长度（或方向）。
*   **向量点积 (Dot Product)**：$\mathbf{x} \cdot \mathbf{y} = \sum x_i y_i$。
    *   **几何意义**：$|\mathbf{x}| |\mathbf{y}| \cos\theta$（衡量两个向量的**相似度**）。
    *   **AI应用**：注意力机制、余弦相似度、投影。

**向量的范数 (Norm)**：
*   **L1范数**：$||\mathbf{x}||_1 = \sum |x_i|$（曼哈顿距离）。
*   **L2范数**：$||\mathbf{x}||_2 = \sqrt{\sum x_i^2}$（欧几里得距离）。
*   **L∞范数**：$||\mathbf{x}||_\infty = \max |x_i|$。
*   **AI应用**：正则化（L1/L2正则）防止过拟合，损失函数定义。

### 1.1.3 矩阵运算

**基础运算**：
*   **矩阵加法**：$C = A + B$（对应元素相加）。
*   **矩阵乘法**：$C = AB$。
    *   **维度要求**：$A(m \times n) \times B(n \times p) = C(m \times p)$。
    *   **计算规则**：$c_{ij} = \sum_k (a_{ik} \cdot b_{kj})$（行乘以列）。
    *   **重要性质**：矩阵乘法**不满足交换律** ($AB \neq BA$)。
*   **转置 (Transpose)**：$A^T$，行列互换。
    *   $(AB)^T = B^T A^T$。
    *   **AI应用**：调整数据维度以适配模型接口。
*   **逆矩阵 (Inverse)**：$A^{-1}$，满足 $AA^{-1} = I$。
    *   条件：只有**方阵**且**行列式≠0**时才有逆。

### 1.1.4 特殊矩阵

*   **单位矩阵 $I$**：主对角线为1，其余为0。
*   **对角矩阵**：仅对角线非零。
*   **对称矩阵**：$A = A^T$。
*   **正交矩阵**：$A^T A = I$（列向量两两正交且为单位向量）。
    *   **AI应用**：正交初始化，保持梯度的稳定性。

### 1.1.5 特征值与特征向量 (Eigenvalues & Eigenvectors)

*   **定义**：对于方阵 $A$，若存在标量 $\lambda$ 和非零向量 $\mathbf{v}$ 使得 $A\mathbf{v} = \lambda \mathbf{v}$。
    *   $\lambda$ 称为**特征值**。
    *   $\mathbf{v}$ 称为**特征向量**。
*   **几何意义**：
    *   特征向量表示变换中的**不变方向**。
    *   特征值表示在该方向上的**伸缩倍数**。
*   **AI应用**：
    *   **PCA (主成分分析)**：利用特征值分解进行数据降维。
    *   **PageRank**：利用特征向量计算网页排名。

### 1.1.6 矩阵分解

*   **特征分解 (Eigendecomposition)**：
    *   $A = Q\Lambda Q^{-1}$ ($Q$为特征向量矩阵，$\Lambda$为特征值对角矩阵)。
    *   限制：$A$必须是方阵。
*   **奇异值分解 (SVD)**：
    *   $A = U\Sigma V^T$。
    *   $U, V$ 为正交矩阵，$\Sigma$ 为奇异值对角矩阵。
    *   **优点**：适用于**任意形状**的矩阵。
    *   **AI应用**：
        *   LSA (潜在语义分析)。
        *   图像压缩 / 降噪。
        *   推荐系统（矩阵补全）。

## 1.2 Python实现 (NumPy)

```python
import numpy as np

# 1. 基础定义
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 2. 向量运算
v_add = v1 + v2             # 加法
v_dot = np.dot(v1, v2)      # 点积: 32
l2_norm = np.linalg.norm(v1) # L2范数

# 3. 矩阵运算
C = np.dot(A, B)            # 矩阵乘法 (或 A @ B)
A_T = A.T                   # 转置
try:
    A_inv = np.linalg.inv(A)    # 逆矩阵
except np.linalg.LinAlgError:
    print("矩阵不可逆")

# 4. 特征值与特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"特征值: {eigenvalues}")

# 5. SVD分解
U, S, Vt = np.linalg.svd(A)
print(f"奇异值: {S}")

# 6. 求解线性方程组 Ax = b
b = np.array([9, 8])
x = np.linalg.solve(A, b)
print(f"方程组解: {x}")
```

## 1.3 常见陷阱

::: warning 注意事项
1.  **维度检查**：进行矩阵乘法前，务必检查维度是否匹配 $(m \times n) \times (n \times p)$。
2.  **广播机制 (Broadcasting)**：NumPy会自动扩展维度，这很方便但也容易导致意外的bug。如果不确定，使用 `reshape` 显式指定维度。
3.  **数值稳定性**：尽量避免直接求逆矩阵（`inv`），而是使用 `solve` 求解方程组，或者使用伪逆（`pinv`）。
:::

## 1.4 练习题

### 基础练习
1.  **手动计算**：给定 $A = [[1, 2], [3, 4]]$，计算 $A^2$ 和 $A^T$。
2.  **几何理解**：两个向量的点积为0意味着什么？（正交/垂直）。

### 编程练习
1.  **余弦相似度**：编写函数 `cosine_similarity(v1, v2)`。
2.  **PCA简化版**：使用 `np.linalg.eig` 对二维数据进行降维。
3.  **图片压缩**：读取一张灰度图片，对其进行SVD分解，保留前 $k$ 个奇异值重构图片，观察清晰度变化。

## 1.5 实战案例：使用SVD进行图像压缩

为了直观理解SVD（奇异值分解）在数据压缩中的作用，我们来实现一个简单的图像压缩器。

**原理**：
任何图像矩阵 $A$ 都可以分解为 $A = U \Sigma V^T$。其中 $\Sigma$ 中的奇异值按从大到小排列。较大的奇异值包含了图像的主要信息（轮廓、主要色块），而较小的奇异值通常对应噪声或细节。通过只保留前 $k$ 个最大的奇异值，我们可以用更少的数据来近似重构原始图像。

**代码实现**：

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def compress_image(image_path, k_values):
    """
    使用SVD压缩图像
    :param image_path: 图片路径
    :param k_values: 保留的奇异值数量列表
    """
    # 1. 读取图片并转换为灰度图
    img = Image.open(image_path).convert('L')
    img_mat = np.array(img)
    
    print(f"原始图像尺寸: {img_mat.shape}")
    
    # 2. SVD分解
    # U: (m, m), S: (min(m,n),), Vt: (n, n)
    U, S, Vt = np.linalg.svd(img_mat, full_matrices=False)
    
    # 3. 绘制结果
    plt.figure(figsize=(12, 6))
    
    # 显示原图
    plt.subplot(1, len(k_values) + 1, 1)
    plt.imshow(img_mat, cmap='gray')
    plt.title("Original")
    plt.axis('off')
    
    # 显示不同压缩率的结果
    for i, k in enumerate(k_values):
        # 重构矩阵: U[:, :k] @ diag(S[:k]) @ Vt[:k, :]
        compressed_img = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        
        # 计算压缩比 (近似)
        original_size = img_mat.size
        compressed_size = k * (img_mat.shape[0] + k + img_mat.shape[1])
        ratio = compressed_size / original_size * 100
        
        plt.subplot(1, len(k_values) + 1, i + 2)
        plt.imshow(compressed_img, cmap='gray')
        plt.title(f"k={k}\n({ratio:.1f}%)")
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

# 模拟运行（请替换为本地实际图片路径）
# compress_image('test_image.jpg', k_values=[10, 50, 100])
```

**实验观察**：
*   当 `k` 很小时（如10），图像非常模糊，只能看到大致轮廓。
*   随着 `k` 增加（如50），图像逐渐清晰，但仍有噪点。
*   当 `k` 达到一定程度（如100），图像与原图几乎无异，但数据量可能只有原图的20%-30%。

---

## 1.6 学习资源

*   **视频**：[3Blue1Brown - 线性代数的本质](https://space.bilibili.com/88461692/channel/seriesdetail?sid=1528931) (强烈推荐，建立几何直觉)。
*   **教材**：《线性代数导论》- Gilbert Strang (MIT 18.06)。
*   **文档**：[NumPy 官方文档](https://numpy.org/doc/stable/)。
