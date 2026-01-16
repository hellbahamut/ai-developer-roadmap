---
title: 第四阶段：深度学习
description: 掌握神经网络、PyTorch、CNN/RNN/Transformer
order: 4
---

# 第四阶段：深度学习

## 阶段目标
掌握神经网络核心原理，熟练使用 **PyTorch** 深度学习框架，能够构建、训练和优化深度神经网络（CNN, RNN, Transformer）。

## 学习时间规划
*   **总时长**： 3-4个月
*   **每周投入**： 15-20小时
*   **建议节奏**： 每个子模块2-3周

## 核心模块导航

| 模块 | 内容重点 | 预计时长 |
| :--- | :--- | :--- |
| **[01. 神经网络基础](./01-neural-networks.md)** | 感知机、MLP、反向传播、激活函数 | 3-4周 |
| **[02. PyTorch基础](./02-pytorch.md)** | Tensor、自动求导、nn.Module、DataLoader | 2-3周 |
| **[03. 卷积神经网络](./03-cnn.md)** | 卷积层、池化、ResNet、迁移学习 | 3-4周 |
| **[04. 循环神经网络](./04-rnn.md)** | RNN、LSTM、GRU、序列预测 | 2-3周 |
| **[05. Transformer](./05-transformer.md)** | Attention机制、BERT、GPT | 2-3周 |
| **[06. 综合项目](./06-projects.md)** | 图像分类、文本情感分析 | 2周 |

## 环境配置

### 1. 安装 PyTorch (推荐)
PyTorch 是目前研究界和工业界最流行的深度学习框架。

**访问官网获取命令**：[pytorch.org](https://pytorch.org/get-started/locally/)

示例（带有CUDA支持）：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

示例（CPU版本）：
```bash
pip install torch torchvision torchaudio
```

### 2. 验证安装
```python
import torch
print(torch.__version__)
print(f"CUDA Available: {torch.cuda.is_available()}")
```

### 3. 其他依赖
```bash
pip install tensorboard tqdm albumentations
```

## 阶段验收与总结

### 验收标准

#### 1. 理论基础
*   [ ] 能手推 **反向传播 (Backpropagation)** 算法
*   [ ] 理解 **梯度消失/爆炸** 及其解决方案 (ReLU, BatchNorm, ResNet)
*   [ ] 掌握 **卷积 (Convolution)** 和 **池化 (Pooling)** 的计算细节
*   [ ] 理解 **注意力机制 (Attention)** 的数学原理

#### 2. 框架能力
*   [ ] 熟练使用 PyTorch 的 `Tensor` 操作
*   [ ] 能自定义 `nn.Module` 构建复杂网络
*   [ ] 熟练使用 `DataLoader` 加载自定义数据集
*   [ ] 能编写完整的训练循环 (Training Loop)

#### 3. 模型架构
*   [ ] **CNN**: 能复现 ResNet 或 VGG
*   [ ] **RNN**: 能使用 LSTM 处理时间序列
*   [ ] **Transformer**: 理解 Encoder/Decoder 结构

#### 4. 调试与优化
*   [ ] 能看懂 Loss 曲线并进行调优
*   [ ] 掌握过拟合处理 (Dropout, Weight Decay, Data Augmentation)
*   [ ] 能够保存和加载模型检查点 (Checkpoint)

## 常见问题 (Q&A)

**Q1: 为什么选择 PyTorch 而不是 TensorFlow?**
**A**: PyTorch 采用动态计算图 (Dynamic Computation Graph)，调试更方便，Python风格更浓，是目前学术界的主流。TensorFlow 2.0 虽然也引入了动态图，但在易用性上 PyTorch 仍有优势。

**Q2: 没有 GPU 怎么学？**
**A**:
1.  对于简单的 MLP 和 CNN（如 MNIST/CIFAR-10），CPU 足够。
2.  使用免费的云端 GPU 资源：**Google Colab** 或 **Kaggle Kernels**。

**Q3: 梯度消失是什么？**
**A**: 在深层网络中，梯度在反向传播时经过多次连乘，如果激活函数导数小于1（如Sigmoid），梯度会指数级衰减，导致浅层参数无法更新。解决方案：使用 **ReLU** 激活函数、**BatchNorm**、**残差连接 (ResNet)**。
