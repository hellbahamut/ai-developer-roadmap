# 5.5 Agent高级应用

## 1. 自定义工具开发

### 1.1 使用Pydantic定义输入

```python
from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field

# 1. 定义工具输入Schema
class CalculatorInput(BaseModel):
    expression: str = Field(description="要计算的数学表达式")

# 2. 定义工具
class CalculatorTool(BaseTool):
    name = "calculator"
    description = "用于计算数学表达式，支持加减乘除和括号"
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(self, expression: str) -> str:
        """同步运行"""
        try:
            result = eval(expression)
            return f"计算结果：{result}"
        except Exception as e:
            return f"计算错误：{str(e)}"

    async def _arun(self, expression: str) -> str:
        """异步运行"""
        return self._run(expression)

# 3. 使用工具
calculator = CalculatorTool()
result = calculator.run("2 * (3 + 4)")
```

## 2. 结构化输出Agent

强制Agent输出特定格式的数据（如JSON）。

```python
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

# 1. 定义输出模型
class ProductReview(BaseModel):
    """产品评论分析"""
    product_name: str = Field(description="产品名称")
    sentiment: str = Field(description="情感类别：positive/negative/neutral")
    rating: int = Field(description="评分：1-5", ge=1, le=5)
    key_points: List[str] = Field(description="关键点列表")
    summary: str = Field(description="一句话总结")

# 2. 创建解析器
parser = PydanticOutputParser(pydantic_object=ProductReview)

# 3. 创建Prompt并注入指令
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的产品评论分析师。"),
    ("user", """
分析以下产品评论：

{format_instructions}

评论内容：
{review}
""")
])

prompt = prompt.partial(
    format_instructions=parser.get_format_instructions()
)

# 4. 创建链
chain = prompt | llm | parser
```

## 3. 多Agent协作

通过多个专门的Agent协作完成复杂任务（如：一个负责研究，一个负责写作）。

```python
# 示例架构：
# Research Agent: 拥有搜索工具，负责搜集资料
# Writing Agent: 拥有写作和修改工具，负责生成内容

# 协作流程函数
def create_article(topic):
    """研究→写作→修改"""
    # 1. 研究阶段
    # research_result = research_executor.invoke(...)
    
    # 2. 写作阶段
    # writing_result = writing_executor.invoke(...)
    
    # 3. 修改阶段
    # final_result = writing_executor.invoke(...)
    
    return # final_result
```
