# 5.2 Prompt Engineering（提示工程）

## 1. Prompt基础结构

### 1.1 标准Prompt模板

```python
# 基础结构
prompt_template = """
# 角色设定
你是一个{role}，具有{expertise}专业背景。

# 任务描述
你需要完成以下任务：{task}

# 输入信息
{input}

# 要求
1. {requirement_1}
2. {requirement_2}

# 输出格式
{output_format}

请开始：
"""

# 使用示例
prompt = prompt_template.format(
    role="Python程序员",
    expertise="5年Web开发经验",
    task="编写一个Flask API接口",
    input="用户注册功能，包含邮箱和密码验证",
    requirement_1="代码需要有详细的注释",
    requirement_2="包含错误处理",
    output_format="请以Markdown代码块形式输出"
)
```

### 1.2 CO-STAR框架

CO-STAR：Context, Objective, Style, Tone, Audience, Response

```python
def create_costar_prompt(context, objective, style, tone, audience, response):
    prompt = f"""
# Context (背景)
{context}

# Objective (目标)
{objective}

# Style (风格)
{style}

# Tone (语气)
{tone}

# Audience (受众)
{audience}

# Response Format (响应格式)
{response}
"""
    return prompt

# 使用示例
prompt = create_costar_prompt(
    context="我是一名初学者，正在学习Python类和对象的概念。",
    objective="请解释Python中的类和对象，并提供简单易懂的示例。",
    style="教学性、循序渐进",
    tone="友好、鼓励",
    audience="编程初学者",
    response="请使用类比和代码示例，最后给出练习建议。"
)
```

## 2. 高级Prompt技巧

### 2.1 Few-Shot Learning（少样本学习）

```python
# Zero-shot（无示例）
prompt_zero_shot = """
将以下文本分类为正面、负面或中性：
这部电影太精彩了！
分类：
"""

# One-shot（一个示例）
prompt_one_shot = """
将以下文本分类为正面、负面或中性：

示例1：
文本：这个产品简直是垃圾，浪费钱！
分类：负面

现在请分类：
这部电影太精彩了！
分类：
"""

# Few-shot（多个示例）
prompt_few_shot = """
将以下文本分类为正面、负面或中性：

示例1：
文本：这个产品简直是垃圾，浪费钱！
分类：负面

示例2：
文本：还可以，但不是很满意。
分类：中性

示例3：
文本：非常喜欢，强烈推荐！
分类：正面

现在请分类：
这部电影太精彩了！
分类：
"""
```

### 2.2 思维链（Chain-of-Thought）⭐⭐⭐⭐⭐

```python
# 标准Prompt（可能出错）
standard_prompt = """
小明有10个苹果，他给了小红3个，又买了5个，然后吃了2个。
请问小明现在有几个苹果？
"""

# CoT Prompt（让模型逐步思考）
cot_prompt = """
小明有10个苹果，他给了小红3个，又买了5个，然后吃了2个。
请问小明现在有几个苹果？

请一步步思考：
1. 开始：小明有10个苹果
2. 给小红3个后：
3. 买了5个后：
4. 吃了2个后：
5. 最终数量：

答案：
"""

# 自动CoT
auto_cot_prompt = """
小明有10个苹果，他给了小红3个，又买了5个，然后吃了2个。
请问小明现在有几个苹果？

让我们一步步思考这个问题：
"""
```

### 2.3 自我一致性（Self-Consistency）

```python
import openai

def self_consistency_solver(problem, num_samples=5):
    """通过多次采样获得一致答案"""
    
    # 生成多个推理路径
    responses = []
    for _ in range(num_samples):
        prompt = f"""
{problem}

让我们一步步思考这个问题：
"""
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7  # 增加随机性
        )
        responses.append(response.choices[0].message.content)
    
    # 投票选择最常见答案
    from collections import Counter
    # 注意：这里需要一个提取答案的辅助函数 extract_answer
    # answers = [extract_answer(r) for r in responses]
    # most_common = Counter(answers).most_common(1)[0][0]
    
    return responses # 简化返回
```

### 2.4 结构化输出

```python
import json

# 方法1：明确要求JSON格式
prompt_json = """
请分析以下电影评论的情感：

评论："这部电影太棒了！剧情紧凑，演员演技出色，强烈推荐！"

请以JSON格式输出，包含以下字段：
{
  "sentiment": "情感类别（正面/负面/中性）",
  "confidence": "置信度（0-1之间的浮点数）",
  "keywords": ["关键词列表"],
  "reasoning": "判断理由"
}
"""

# 方法2：使用Function Calling（推荐）
def analyze_sentiment(review):
    """使用Function Calling进行结构化分析"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"分析这条评论的情感：{review}"}
        ],
        functions=[{
            "name": "analyze_review",
            "description": "分析电影评论的情感",
            "parameters": {
                "type": "object",
                "properties": {
                    "sentiment": {
                        "type": "string",
                        "enum": ["positive", "negative", "neutral"],
                        "description": "情感类别"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "置信度（0-1）"
                    },
                    "key_points": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "关键点"
                    }
                },
                "required": ["sentiment", "confidence", "key_points"]
            }
        }],
        function_call="auto"
    )
    
    # 解析函数调用
    function_call = response.choices[0].message.function_call
    return json.loads(function_call.arguments)
```

## 3. Prompt优化策略

### 3.1 迭代优化流程

```python
# 版本1（初始）
prompt_v1 = "写一个Python函数来计算斐波那契数列"

# 版本2（增加要求）
prompt_v2 = """
写一个Python函数来计算斐波那契数列。
要求：
1. 使用递归实现
2. 添加类型提示
3. 包含文档字符串
"""

# 版本3（优化后）
prompt_v3 = """
# 任务
编写一个高效的斐波那契数列计算函数。

# 要求
1. 考虑性能问题（不要用纯递归）
2. 使用动态规划或记忆化
3. 添加完整的类型提示（Python 3.9+）
4. 包含详细的docstring（Google风格）
5. 添加示例代码和使用说明

# 输出格式
请以Markdown代码块形式输出，包含：
- 函数实现
- 时间复杂度分析
- 使用示例
"""
```

### 3.2 A/B测试Prompt

```python
def ab_test_prompts(prompt_a, prompt_b, test_cases):
    """A/B测试两个Prompt的效果"""
    
    results_a = []
    results_b = []
    
    for test_case in test_cases:
        # 测试Prompt A
        response_a = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt_a + "\n" + test_case}]
        )
        results_a.append(response_a.choices[0].message.content)
        
        # 测试Prompt B
        response_b = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt_b + "\n" + test_case}]
        )
        results_b.append(response_b.choices[0].message.content)
    
    return results_a, results_b
```
