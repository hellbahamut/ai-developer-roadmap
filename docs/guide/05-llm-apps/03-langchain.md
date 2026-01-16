# 5.3 LangChain框架

## 1. LangChain基础

### 1.1 核心概念

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

load_dotenv()

# 1. LLM（大语言模型）
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)

# 简单调用
response = llm.invoke("你好，请介绍一下你自己")
print(response.content)

# 2. Prompt Template（提示模板）
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的{role}。"),
    ("user", "{input}")
])

# 填充模板
prompt_value = prompt.format_messages(
    role="Python程序员",
    input="如何用Python读取CSV文件？"
)

# 3. Chain（链）
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(
    role="Python程序员",
    input="如何用Python读取CSV文件？"
)
print(result)

# 4. 使用LCEL（LangChain Expression Language）
from langchain.schema.runnable import RunnablePassthrough

chain = prompt | llm
result = chain.invoke({
    "role": "Python程序员",
    "input": "如何用Python读取CSV文件？"
})
print(result.content)
```

### 1.2 常用Chain类型

```python
from langchain.chains import (
    SimpleSequentialChain,
    SequentialChain
)

# 1. SimpleSequentialChain（简单顺序链）
# 输出 → 输入

prompt1 = ChatPromptTemplate.from_template(
    "为{product}写一个简短的广告标语。"
)
chain1 = LLMChain(llm=llm, prompt=prompt1)

prompt2 = ChatPromptTemplate.from_template(
    "将以下标语翻译成英文：\n{标语}"
)
chain2 = LLMChain(llm=llm, prompt=prompt2)

# 串联
combined_chain = SimpleSequentialChain(chains=[chain1, chain2], verbose=True)
result = combined_chain.run("智能手表")

# 2. SequentialChain（复杂顺序链）
# 多个输入输出

prompt_title = ChatPromptTemplate.from_template(
    "为以下主题写一个吸引人的标题：\n{topic}"
)
chain_title = LLMChain(llm=llm, prompt=prompt_title, output_key="title")

prompt_content = ChatPromptTemplate.from_template(
    "基于标题'{title}'，写一篇博客文章大纲，主题是：{topic}"
)
chain_content = LLMChain(llm=llm, prompt=prompt_content, output_key="outline")

prompt_summary = ChatPromptTemplate.from_template(
    "总结以下大纲：\n{outline}"
)
chain_summary = LLMChain(llm=llm, prompt=prompt_summary, output_key="summary")

# 组合
overall_chain = SequentialChain(
    chains=[chain_title, chain_content, chain_summary],
    input_variables=["topic"],
    output_variables=["title", "outline", "summary"],
    verbose=True
)

result = overall_chain({"topic": "人工智能的未来"})
```

### 1.3 Router Chain（路由链）

根据输入内容自动选择处理链。

```python
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.prompts import PromptTemplate

# 定义多个专家Prompt...
# (此处省略详细定义，核心思想是根据input内容选择最适合的PromptTemplate进行回答)
```

## 2. Memory（记忆）

```python
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory
)
from langchain.chains import ConversationChain

# 1. ConversationBufferMemory（保存所有对话）
memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

conversation.predict(input="我叫小明")
conversation.predict(input="我的爱好是什么？")
print(memory.buffer)  # 查看所有对话历史

# 2. ConversationBufferWindowMemory（只保存最近k轮）
memory_window = ConversationBufferWindowMemory(k=2)

# 3. ConversationSummaryMemory（摘要记忆）
memory_summary = ConversationSummaryMemory(llm=llm)

# 4. 自定义Memory（使用Redis等）
# from langchain.memory import RedisEntityMemory
```

## 3. Agent（智能体）⭐⭐⭐⭐⭐

### 3.1 ReAct Agent

```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub

# 1. 定义工具
def calculator(expression):
    """计算数学表达式"""
    try:
        result = eval(expression)
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算错误：{str(e)}"

def search_wikipedia(query):
    """搜索维基百科"""
    return f"这是关于'{query}'的维基百科搜索结果..."

tools = [
    Tool(
        name="Calculator",
        func=calculator,
        description="用于数学计算，输入应该是数学表达式字符串"
    ),
    Tool(
        name="Wikipedia",
        func=search_wikipedia,
        description="用于搜索知识，输入应该是搜索关键词"
    )
]

# 2. 获取Prompt模板
prompt = hub.pull("hwchase17/react")

# 3. 创建Agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# 4. 创建Agent执行器
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# 5. 运行Agent
result = agent_executor.invoke({
    "input": "如果我有100元，买了3本书每本15元，还剩多少钱？"
})
print(result["output"])
```

### 3.2 OpenAI Functions Agent

```python
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.tools import tool

# 定义工具（使用装饰器）
@tool
def search_database(query: str) -> str:
    """搜索产品数据库"""
    return f"未找到包含'{query}'的产品"

@tool
def get_product_price(product_name: str) -> str:
    """获取产品价格"""
    return "999元"

tools = [search_database, get_product_price]

# 创建Agent
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的购物助手。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_openai_functions_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

result = agent_executor.invoke({
    "input": "我想买iPhone，多少钱？"
})
```
