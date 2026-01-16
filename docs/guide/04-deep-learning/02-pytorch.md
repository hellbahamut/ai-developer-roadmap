---
title: 模块二：PyTorch基础
description: Tensor操作、自动求导与模型构建
order: 2
---

# 模块二：PyTorch基础

## 2.1 Tensor (张量)

PyTorch 的核心数据结构，类似于 NumPy 的 ndarray，但支持 GPU 加速。

```python
import torch

# 创建
x = torch.randn(2, 3)       # 标准正态分布
y = torch.ones(2, 3)

# 运算
z = x + y
z = torch.matmul(x, y.T)    # 矩阵乘法

# GPU支持
if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    z = x + y # 在GPU上运算
```

## 2.2 自动求导 (Autograd) ⭐⭐⭐⭐⭐

PyTorch 会自动记录 Tensor 的操作历史，从而自动计算梯度。

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1

y.backward() # 反向传播
print(x.grad) # 2x + 3 = 7.0
```

## 2.3 nn.Module 模型构建

推荐使用类继承的方式构建模型。

```python
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

model = SimpleNet()
```

## 2.4 数据加载 (DataLoader)

处理批量数据、打乱、多进程加载。

```python
from torch.utils.data import DataLoader, TensorDataset

dataset = TensorDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch_X, batch_y in loader:
    # 训练逻辑
    pass
```

## 2.5 完整训练循环 (Template)

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

for epoch in range(100):
    for X, y in loader:
        # 1. 清零梯度
        optimizer.zero_grad()
        
        # 2. 前向传播
        output = model(X)
        
        # 3. 计算损失
        loss = criterion(output, y)
        
        # 4. 反向传播
        loss.backward()
        
        # 5. 更新参数
        optimizer.step()
```
