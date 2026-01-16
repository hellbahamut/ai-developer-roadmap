# 6.1 模型服务化

## 1. Flask部署示例

### 1.1 基础Flask API

```python
from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

# 全局加载模型（避免重复加载）
MODEL_PATH = "./models/sentiment_model"
model = None
tokenizer = None

def load_model():
    """加载模型"""
    global model, tokenizer
    if model is None:
        print("Loading model...")
        # 实际使用时需确保模型文件存在
        # tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        # model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        # model.eval()
        print("Model loaded!")

# 启动时加载
load_model()

@app.route('/predict', methods=['POST'])
def predict():
    """预测接口"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
            
        # 模拟预测
        result = {
            "text": text,
            "sentiment": "positive",
            "confidence": 0.95
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

## 2. FastAPI（推荐）⭐⭐⭐⭐⭐

### 2.1 为什么选择FastAPI？
- **高性能**：基于Starlette和Pydantic，性能接近Go/Node.js。
- **自动文档**：自动生成Swagger UI和ReDoc。
- **类型安全**：利用Python类型提示进行验证。
- **异步支持**：原生支持`async/await`。

### 2.2 FastAPI示例

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import uvicorn

app = FastAPI(
    title="Sentiment Analysis API",
    description="情感分析服务",
    version="1.0.0"
)

# 定义请求模型
class PredictionRequest(BaseModel):
    text: str = Field(..., description="待分析文本", min_length=1, max_length=5000)
    return_probabilities: bool = Field(default=False, description="是否返回所有类别概率")

class PredictionResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    probabilities: dict = None

# 启动事件
@app.on_event("startup")
async def startup_event():
    print("Loading model...")
    # load_model()
    print("Model loaded!")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """单条预测"""
    try:
        # 模拟预测逻辑
        return PredictionResponse(
            text=request.text,
            sentiment="positive",
            confidence=0.98
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## 3. LLM服务化

### 3.1 简单LLM API

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from typing import Optional

app = FastAPI(title="LLM Chat API")

# 存储多个会话
sessions = {}

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"

@app.post("/chat")
async def chat(request: ChatRequest):
    """聊天接口"""
    session_id = request.session_id
    
    # 获取或创建会话
    if session_id not in sessions:
        # llm = ChatOpenAI(...)
        # memory = ConversationBufferMemory()
        # chain = ConversationChain(...)
        sessions[session_id] = "mock_chain"
    
    return {"response": f"Echo: {request.message}", "session_id": session_id}
```

### 3.2 RAG服务

构建一个支持文件上传和问答的RAG服务。
（代码略，见上一阶段RAG部分，核心是将RAG流程封装为FastAPI接口）

## 4. 实战案例：生产级FastAPI项目结构

单文件的FastAPI脚本适合Demo，但在生产环境中，我们需要更清晰的目录结构来管理路由、模型、业务逻辑和配置。

**推荐目录结构**：

```
my_ai_service/
├── app/
│   ├── __init__.py
│   ├── main.py              # 入口文件
│   ├── core/                # 核心配置
│   │   ├── config.py        # 环境变量配置
│   │   └── security.py      # 认证逻辑
│   ├── api/                 # API路由
│   │   ├── __init__.py
│   │   ├── deps.py          # 依赖注入
│   │   └── v1/
│   │       ├── endpoints/
│   │       │   ├── chat.py
│   │       │   └── sentiment.py
│   │       └── api.py
│   ├── schemas/             # Pydantic模型 (DTO)
│   │   ├── chat.py
│   │   └── item.py
│   ├── services/            # 业务逻辑 (AI模型加载与推理)
│   │   ├── llm_service.py
│   │   └── model_loader.py
│   └── utils/               # 工具函数
├── tests/                   # 测试用例
├── .env                     # 环境变量
├── requirements.txt
└── Dockerfile
```

**核心代码示例**：

1.  **`app/core/config.py`** (配置管理)

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Model Service"
    API_V1_STR: str = "/api/v1"
    OPENAI_API_KEY: str

    class Config:
        env_file = ".env"

settings = Settings()
```

2.  **`app/schemas/chat.py`** (数据验证)

```python
from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    temperature: float = 0.7

class ChatResponse(BaseModel):
    response: str
    tokens_used: int
```

3.  **`app/api/v1/endpoints/chat.py`** (路由逻辑)

```python
from fastapi import APIRouter, Depends, HTTPException
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.llm_service import LLMService

router = APIRouter()

# 依赖注入方式获取Service实例
def get_llm_service():
    return LLMService()

@router.post("/completions", response_model=ChatResponse)
async def chat_completion(
    request: ChatRequest,
    service: LLMService = Depends(get_llm_service)
):
    try:
        result = await service.generate_response(request.message, request.temperature)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

4.  **`app/main.py`** (程序入口)

```python
from fastapi import FastAPI
from app.core.config import settings
from app.api.v1.api import api_router

app = FastAPI(title=settings.PROJECT_NAME)

# 注册路由
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/health")
def health_check():
    return {"status": "healthy"}
```

**为什么这样做？**
1.  **解耦**：路由(Router)、数据定义(Schema)和业务逻辑(Service)分离，互不干扰。
2.  **可维护性**：新增一个API只需要增加对应的文件，不会让 `main.py` 变得臃肿。
3.  **环境隔离**：通过 `.env` 管理配置，方便在开发、测试和生产环境切换。

