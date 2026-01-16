# 6.2 容器化部署

## 1. Docker基础

### 1.1 Dockerfile编写

```dockerfile
# 使用Python官方镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 8000

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 1.2 构建和运行

```bash
# 构建镜像
docker build -t my-llm-api:latest .

# 运行容器
docker run -d \
  --name llm-api \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your-key \
  -v $(pwd)/models:/app/models \
  my-llm-api:latest
```

## 2. Docker Compose（多服务）

### 2.1 docker-compose.yml

```yaml
version: '3.8'

services:
  # API服务
  api:
    build: .
    container_name: llm-api
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DATABASE_URL=postgresql://user:pass@db:5432/llm_db
    volumes:
      - ./models:/app/models
    depends_on:
      - db
      - redis
    restart: unless-stopped

  # PostgreSQL数据库
  db:
    image: postgres:15-alpine
    container_name: llm-db
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=llm_db
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  # Redis缓存
  redis:
    image: redis:7-alpine
    container_name: llm-redis
    ports:
      - "6379:6379"
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

### 2.2 常用命令

```bash
# 启动所有服务
docker-compose up -d

# 查看日志
docker-compose logs -f api

# 停止所有服务
docker-compose down
```

## 3. 多阶段构建（优化镜像大小）

```dockerfile
# 第一阶段：构建
FROM python:3.10-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# 第二阶段：运行
FROM python:3.10-slim
WORKDIR /app
# 只复制必要的文件
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```
