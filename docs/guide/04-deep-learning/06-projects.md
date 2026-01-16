---
title: 综合项目：深度学习实战
description: 图像分类与文本情感分析
order: 6
---

# 综合项目：深度学习实战

## 项目一：CIFAR-10 图像分类

**目标**：从零构建一个 ResNet 模型（或使用预训练模型），在 CIFAR-10 数据集上达到 90%+ 的准确率。

### 关键步骤
1.  **数据增强**: RandomCrop, RandomHorizontalFlip, Normalize。
2.  **模型构建**: 使用 `torchvision.models.resnet18`。
3.  **训练策略**:
    *   Optimizer: SGD with Momentum。
    *   Scheduler: CosineAnnealingLR。
    *   Loss: CrossEntropyLoss。
4.  **可视化**: 使用 TensorBoard 记录 Loss 和 Accuracy。

## 项目二：IMDB 电影评论情感分析

**目标**：使用 LSTM 或 Transformer 对影评进行正负面分类。

### 关键步骤
1.  **文本预处理**: Tokenization, Padding, 构建词表。
2.  **Embedding**: 使用 `nn.Embedding` 或预训练的 GloVe 词向量。
3.  **模型构建**:
    *   Bi-LSTM + Linear。
    *   或 Fine-tune BERT。
4.  **评估**: Accuracy, F1-Score。

## 进阶挑战
尝试实现 **Style Transfer (风格迁移)**，将一张图片的风格迁移到另一张图片上。这需要理解 CNN 的特征图如何表示内容和风格。
