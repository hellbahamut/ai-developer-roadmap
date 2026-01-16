---
title: 模块三：卷积神经网络 (CNN)
description: 卷积层、池化层与经典架构
order: 3
---

# 模块三：卷积神经网络 (CNN)

CNN 是计算机视觉领域的基石。

## 3.1 核心组件

### 3.1.1 卷积层 (Conv2d) ⭐⭐⭐⭐⭐
提取局部特征。
*   **Kernel (Filter)**: 卷积核，在图像上滑动。
*   **Stride**: 步长。
*   **Padding**: 填充，保持输出尺寸。

```python
nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
```

### 3.1.2 池化层 (Pooling)
降维，减少参数，扩大感受野。
*   **MaxPool**: 取窗口内最大值（提取显著特征）。
*   **AvgPool**: 取平均值。

```python
nn.MaxPool2d(kernel_size=2, stride=2)
```

## 3.2 经典架构

### 3.2.1 LeNet-5
最早的 CNN，用于手写数字识别。

### 3.2.2 AlexNet
深度学习爆发的起点 (2012)，使用了 ReLU 和 Dropout。

### 3.2.3 VGG
通过重复堆叠 3x3 卷积核构建深层网络，结构规整。

### 3.2.4 ResNet (残差网络) ⭐⭐⭐⭐⭐
通过 **Skip Connection (残差连接)** 解决了深层网络的梯度消失问题，使得训练成百上千层的网络成为可能。
$$ y = F(x) + x $$

```python
# 使用预训练模型
import torchvision.models as models
resnet = models.resnet18(pretrained=True)
```

## 3.3 迁移学习 (Transfer Learning)

利用在 ImageNet 上预训练好的模型，微调 (Fine-tune) 到自己的数据集上。
1.  加载预训练模型。
2.  冻结前几层的参数。
3.  替换最后一层全连接层（适应新的类别数）。
4.  训练最后几层。
