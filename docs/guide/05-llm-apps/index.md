# 第五阶段：大语言模型（LLM）应用

## 阶段目标
掌握大语言模型的应用开发技术，能够使用LLM构建智能应用，包括Prompt Engineering、RAG系统、Agent开发等。

## 学习时间规划
- **总时长**： 2-3个月
- **每周投入**： 15-20小时
- **建议节奏**： 每个子模块1-2周

---

## 环境配置

### 安装核心库

```bash
# OpenAI API（或兼容接口）
pip install openai

# LangChain框架 ⭐⭐⭐⭐⭐
pip install langchain langchain-openai langchain-community

# 向量数据库
pip install chromadb faiss-cpu

# 文档处理
pip install pypdf unstructured python-docx

# 嵌入模型
pip install sentence-transformers

# Web爬取
pip install beautifulsoup4 playwright

# 其他工具
pip install tiktoken  # Token计算
pip install dotenv    # 环境变量管理
```

### API密钥配置

1. 创建 `.env` 文件：

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_API_BASE=https://api.openai.com/v1  # 或使用兼容接口

# 或使用国内服务
# SILICONFLOW_API_KEY=your-key
# ZHIPU_API_KEY=your-key
# DEEPSEEK_API_KEY=your-key
```

2. 加载环境变量：

```python
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
print(f"API Key已设置: {api_key[:10]}...")
```
