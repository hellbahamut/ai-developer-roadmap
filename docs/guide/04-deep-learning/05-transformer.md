---
title: 模块五：Transformer
description: Attention机制与现代NLP基石
order: 5
---

# 模块五：Transformer

Transformer 彻底改变了 NLP 领域，现在也正在改变 CV 领域。

## 5.1 自注意力机制 (Self-Attention) ⭐⭐⭐⭐⭐

核心公式：
$$ \text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V $$

*   **Query (Q)**: 查询向量。
*   **Key (K)**: 键向量。
*   **Value (V)**: 值向量。
*   **多头注意力 (Multi-Head Attention)**: 让模型关注不同子空间的信息。

## 5.2 Transformer 架构

*   **Encoder**: 编码器，用于理解输入（如 BERT）。
*   **Decoder**: 解码器，用于生成输出（如 GPT）。
*   **Positional Encoding**: 因为 Transformer 并行处理，需要注入位置信息。

## 5.3 预训练模型

### 5.3.1 BERT (Encoder-only)
*   **任务**: Masked Language Model (MLM), Next Sentence Prediction (NSP)。
*   **特点**: 双向上下文，适合理解任务（分类、问答）。

### 5.3.2 GPT (Decoder-only)
*   **任务**: 自回归语言建模 (预测下一个词)。
*   **特点**: 单向上下文，适合生成任务。

## 5.4 Hugging Face Transformers 库

工业界标准库。

```python
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
```
