# 5.1 大语言模型基础

## 1. 什么是大语言模型

### 1.1 基本概念

**定义**：
- 基于Transformer架构的大规模预训练模型
- 通过在海量文本数据上自监督学习
- 能够理解、生成、推理自然语言

**代表性模型**：

| 模型 | 公司 | 参数量 | 特点 |
|------|------|--------|------|
| GPT-4 | OpenAI | ~1.8T | 最强通用能力 |
| Claude | Anthropic | ~175B | 长文本、安全性 |
| 文心一言 | 百度 | - | 中文优化 |
| 通义千问 | 阿里 | - | 中文能力强 |
| DeepSeek | 深度求索 | - | 开源、数学强 |
| LLaMA | Meta | 8B-70B | 开源基准 |

### 1.2 LLM的能力与局限

**能做什么**：
- 文本生成（文章、代码、创意内容）
- 文本理解（摘要、分类、情感分析）
- 问答（知识查询、解释说明）
- 推理（数学、逻辑、因果）
- 翻译（多语言互译）
- 代码生成与解释

**不能做什么**：
- 实时信息获取（训练截止日期后的数据）
- 复杂数学计算（容易出错）
- 长期记忆（上下文窗口限制）
- 物理世界交互
- 保证100%准确（幻觉问题）

## 2. 快速开始

### 2.1 OpenAI API基础

```python
import openai
from dotenv import load_dotenv
import os

load_dotenv()

# 设置API密钥
client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")  # 可选
)

# 1. 简单对话
response = client.chat.completions.create(
    model="gpt-3.5-turbo",  # 或 gpt-4
    messages=[
        {"role": "system", "content": "你是一个有用的AI助手。"},
        {"role": "user", "content": "你好！请介绍一下你自己。"}
    ],
    temperature=0.7,  # 控制随机性（0-2）
    max_tokens=1000   # 最大输出长度
)

print(response.choices[0].message.content)

# 2. 流式输出
stream = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "写一首关于AI的诗"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")

# 3. 多轮对话
messages = [
    {"role": "system", "content": "你是一个专业的Python导师。"}
]

while True:
    user_input = input("\n你: ")
    if user_input.lower() in ["退出", "exit", "quit"]:
        break
    
    messages.append({"role": "user", "content": user_input})
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    
    assistant_reply = response.choices[0].message.content
    print(f"AI: {assistant_reply}")
    
    messages.append({"role": "assistant", "content": assistant_reply})
```

### 2.2 使用兼容接口（省钱方案）

```python
# 使用国产LLM（兼容OpenAI接口）
# SiliconFlow、DeepSeek、智谱等

# 方法1：直接替换base_url
client = openai.OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1"
)

# 使用模型
response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V3",  # 或其他模型
    messages=[{"role": "user", "content": "你好"}]
)

# 方法2：使用LangChain
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

response = llm.invoke("你好")
print(response.content)
```

## 3. Token计算与成本

```python
import tiktoken

# Token计算
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

text = "Hello, world!"
tokens = encoding.encode(text)
print(f"Token数量: {len(tokens)}")
print(f"Tokens: {tokens}")
print(f"解码: {encoding.decode(tokens)}")

# 计算成本
def calculate_cost(model, input_tokens, output_tokens):
    """计算API调用成本"""
    prices = {
        "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},  # 每1M token价格(美元)
        "gpt-4": {"input": 30, "output": 60},
        "gpt-4-turbo": {"input": 10, "output": 30},
    }
    
    if model not in prices:
        return "未知模型"
    
    input_cost = (input_tokens / 1_000_000) * prices[model]["input"]
    output_cost = (output_tokens / 1_000_000) * prices[model]["output"]
    total_cost = input_cost + output_cost
    
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost
    }

# 使用示例
cost = calculate_cost("gpt-3.5-turbo", 1000, 500)
print(f"成本: ${cost['total_cost']:.4f}")
```
