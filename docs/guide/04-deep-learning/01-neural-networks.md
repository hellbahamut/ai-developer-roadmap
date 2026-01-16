---
title: 模块一：神经网络基础
description: 感知机、MLP、反向传播与优化技巧
order: 1
---

# 模块一：神经网络基础

## 1.1 从感知机到多层感知机 (MLP)

### 1.1.1 感知机 (Perceptron)
最简单的神经网络单元。
$$ y = \text{step}(w \cdot x + b) $$
局限性：只能解决线性可分问题（无法解决 XOR）。

### 1.1.2 多层感知机 (MLP)
通过引入**隐藏层**和**非线性激活函数**，具备了拟合任意函数的能力（通用近似定理）。

### 1.1.3 激活函数 ⭐⭐⭐⭐⭐

| 函数 | 公式 | 特点 | 适用场景 |
| :--- | :--- | :--- | :--- |
| **Sigmoid** | $\frac{1}{1+e^{-x}}$ | 输出(0,1)，容易梯度消失 | 二分类输出层 |
| **Tanh** | $\frac{e^x-e^{-x}}{e^x+e^{-x}}$ | 输出(-1,1)，零中心 | RNN |
| **ReLU** | $\max(0, x)$ | 计算快，解决梯度消失，但在x<0时梯度为0 (Dead ReLU) | 隐藏层首选 |
| **Leaky ReLU** | $\max(\alpha x, x)$ | 解决 Dead ReLU | 深层网络 |

## 1.2 反向传播 (Backpropagation) ⭐⭐⭐⭐⭐

神经网络训练的核心算法。
基于**链式法则 (Chain Rule)**，将损失函数的梯度从输出层向输入层传播。

$$ \frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial z} \cdot \frac{\partial z}{\partial w} $$

## 1.3 训练技巧

### 1.3.1 权重初始化
*   **Xavier Initialization**: 适用于 Sigmoid/Tanh。
*   **He Initialization**: 适用于 ReLU。保持每一层输出的方差一致。

### 1.3.2 批量归一化 (Batch Normalization) ⭐⭐⭐⭐⭐
在每一层的激活函数之前，对输入进行标准化（减均值，除方差）。
*   **优点**：加速收敛、允许更大学习率、轻微正则化效果。

### 1.3.3 Dropout
训练时随机“丢弃”一部分神经元（置零）。
*   **作用**：防止过拟合，相当于训练了多个子网络的集成。
*   **注意**：测试时需关闭 Dropout，并缩放权重（PyTorch自动处理）。

## 1.4 实战案例：设计一个识别手写数字的神经网络

MNIST数据集是深度学习界的"Hello World"。包含60000张训练图片和10000张测试图片，每张是28x28像素的灰度手写数字（0-9）。

**任务**：搭建一个简单的全连接神经网络（MLP），识别图片中的数字。

**网络结构设计**：
1.  **输入层**：784个节点 (28*28拉平)。
2.  **隐藏层1**：128个神经元，ReLU激活。
3.  **隐藏层2**：64个神经元，ReLU激活。
4.  **输出层**：10个神经元（对应0-9十个类别），Softmax输出概率。

**PyTorch代码实现**：

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. 准备数据
transform = transforms.Compose([
    transforms.ToTensor(), # 转为Tensor并归一化到[0,1]
    transforms.Normalize((0.1307,), (0.3081,)) # 标准化 (均值, 标准差)
])

# 下载数据集 (首次运行需要联网)
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 2. 定义模型
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # 定义层
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # x shape: [batch, 1, 28, 28] -> [batch, 784]
        x = x.view(-1, 28 * 28) 
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x) # 输出层通常不需要激活函数(CrossEntropyLoss自带Softmax)
        return x

model = SimpleNet()

# 3. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练循环
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad() # 梯度清零
        output = model(data)  # 前向传播
        loss = criterion(output, target) # 计算损失
        loss.backward()       # 反向传播
        optimizer.step()      # 更新参数
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]\tLoss: {loss.item():.6f}')

# 5. 测试循环
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True) # 获取最大概率的索引
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')

# 运行
# for epoch in range(1, 4):
#     train(epoch)
#     test()
```

**关键点解析**：
*   **`view(-1, 28*28)`**: 图片是二维的，全连接层需要一维输入，所以要"拉平" (Flatten)。
*   **`CrossEntropyLoss`**: 多分类任务的标准损失函数，它内部结合了 LogSoftmax 和 NLLLoss，所以模型输出层不需要手动加 Softmax。
*   **`model.train()` / `model.eval()`**: 切换模式，这会影响 Dropout 和 BatchNormalization 的行为。
