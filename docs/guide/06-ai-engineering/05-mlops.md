# 6.5 MLOps基础

## 1. 实验跟踪（MLflow）

记录训练参数、指标和模型文件，方便复现和对比。

```python
import mlflow

mlflow.set_experiment("iris-classification")

with mlflow.start_run():
    mlflow.log_params({"n_estimators": 100})
    # 训练...
    mlflow.log_metric("accuracy", 0.95)
    mlflow.sklearn.log_model(model, "model")
```

## 2. CI/CD（GitHub Actions）

自动化测试和部署流程。

```yaml
# .github/workflows/ml-pipeline.yml
jobs:
  train-and-evaluate:
    steps:
    - name: Train model
      run: python train.py
    - name: Evaluate
      run: python evaluate.py

  deploy:
    needs: train-and-evaluate
    steps:
    - name: Deploy
      run: ./deploy.sh
```

## 3. A/B测试

同时运行两个版本的模型，根据用户反馈决定使用哪个版本。

```python
@app.post("/predict")
async def predict(user_id: str):
    # 根据user_id哈希值分流
    if hash(user_id) % 2 == 0:
        model = model_v1
    else:
        model = model_v2
    
    return model.predict(...)
```
